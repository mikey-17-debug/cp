"""项目入口：训练与推理命令行。

用法示例（在项目根目录 Bert_model 下执行）：

    # 训练单标签与多标签模型
    python main.py train --task all

    # 只训练多标签模型，并临时覆盖超参
    python main.py train --task multi --epochs 2 --batch-size 8

    # 冒烟测试：只用前 200 条样本、1 个 epoch
    python main.py train --task single --limit 200 --epochs 1

    # 推理
    python main.py predict --task multi --text "一名银行家蒙冤入狱，用二十年时间挖通墙壁越狱。"

    # 导出数据集逐条预测结果（含压缩变体）
    python main.py predict-set --task all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 允许从任意工作目录调用本脚本
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.benchmark import predict_logits  # noqa: E402
from src.config import (  # noqa: E402
    DEFAULT_CONFIG_FILE,
    MULTI_LABEL,
    SINGLE_LABEL,
    TASK_KEYS,
    load_config,
)
from src.data import prepare_data  # noqa: E402
from src.optimization import load_variant  # noqa: E402
from src.predictions import PREDICTIONS_FILE, save_predictions  # noqa: E402
from src.predictor import GenreClassifier  # noqa: E402
from src.trainer import train_all  # noqa: E402
from src.utils import device_label, setup_logging  # noqa: E402

logger = logging.getLogger("main")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于 chinese-bert-wwm 的电影类型分类（单标签 / 多标签）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_FILE),
        help="配置文件路径（相对路径按当前工作目录解析）",
    )
    parser.add_argument(
        "--log-file",
        help="同时把日志写入该文件（相对路径按当前工作目录解析），便于另开窗口训练时回看进度",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="训练模型", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train.add_argument("--task", choices=[*TASK_KEYS, "all"], default="all", help="训练哪个任务")
    train.add_argument("--epochs", type=float, help="覆盖 training.num_train_epochs")
    train.add_argument("--batch-size", type=int, help="覆盖 training.per_device_train_batch_size")
    train.add_argument("--eval-batch-size", type=int, help="覆盖 training.per_device_eval_batch_size")
    train.add_argument("--learning-rate", type=float, help="覆盖 training.learning_rate")
    train.add_argument("--max-length", type=int, help="覆盖 model.max_length")
    train.add_argument("--num-workers", type=int, help="覆盖 training.dataloader_num_workers")
    train.add_argument("--output-root", help="覆盖 training.output_root（相对路径按项目根目录解析）")
    train.add_argument("--seed", type=int, help="覆盖 training.seed 与 split.seed")
    train.add_argument("--limit", type=int, help="覆盖 data.limit，仅用前 N 条样本（调试）")
    train.add_argument("--cpu", action="store_true", help="强制使用 CPU 训练")

    predict = sub.add_parser("predict", help="用已训练模型推理", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    predict.add_argument("--task", choices=list(TASK_KEYS), help="按任务自动定位模型目录")
    predict.add_argument("--model-dir", help="模型目录（指定后忽略 --task 的默认路径）")
    predict.add_argument("--text", nargs="+", required=True, help="待预测的剧情简介，可传多条")
    predict.add_argument("--top-k", type=int, help="单标签任务返回的候选数")
    predict.add_argument("--threshold", type=float, help="多标签任务的判定阈值")
    predict.add_argument("--device", help="推理设备，如 cpu / cuda:0")
    predict.add_argument("--json", action="store_true", help="以 JSON 输出结果")

    predict_set = sub.add_parser(
        "predict-set",
        help="导出数据集逐条预测结果（含压缩变体）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    predict_set.add_argument("--task", choices=[*TASK_KEYS, "all"], default="all", help="导出哪个任务")
    predict_set.add_argument(
        "--split", choices=["train", "val", "test"], default="test", help="导出哪个切分"
    )
    predict_set.add_argument("--model-dir", help="只导出指定模型目录")
    predict_set.add_argument("--batch-size", type=int, help="覆盖 optimization.eval_batch_size")
    predict_set.add_argument("--device", help="推理设备，如 cpu / cuda:0")
    predict_set.add_argument(
        "--overwrite", action="store_true", help="已存在预测文件时强制重新计算"
    )

    return parser


def _train_overrides(parsed: argparse.Namespace) -> Dict[str, Any]:
    """把命令行参数整理成嵌套的配置覆盖项。"""
    training: Dict[str, Any] = {}
    model: Dict[str, Any] = {}
    data: Dict[str, Any] = {}
    split: Dict[str, Any] = {}

    if parsed.epochs is not None:
        training["num_train_epochs"] = parsed.epochs
    if parsed.batch_size is not None:
        training["per_device_train_batch_size"] = parsed.batch_size
    if parsed.eval_batch_size is not None:
        training["per_device_eval_batch_size"] = parsed.eval_batch_size
    if parsed.learning_rate is not None:
        training["learning_rate"] = parsed.learning_rate
    if parsed.num_workers is not None:
        training["dataloader_num_workers"] = parsed.num_workers
    if parsed.output_root is not None:
        training["output_root"] = parsed.output_root
    if parsed.cpu:
        training["use_cpu"] = True
        training["bf16"] = False
    if parsed.max_length is not None:
        model["max_length"] = parsed.max_length
    if parsed.limit is not None:
        data["limit"] = parsed.limit
    if parsed.seed is not None:
        training["seed"] = parsed.seed
        split["seed"] = parsed.seed

    overrides: Dict[str, Any] = {}
    for key, value in (("training", training), ("model", model), ("data", data), ("split", split)):
        if value:
            overrides[key] = value
    return overrides


def _command_train(parsed: argparse.Namespace) -> int:
    config = load_config(parsed.config, _train_overrides(parsed))
    task_keys = list(TASK_KEYS) if parsed.task == "all" else [parsed.task]

    logger.info("配置文件：%s", config.config_file)
    logger.info("训练数据：%s", config.data_path)
    logger.info("预训练模型：%s", config.pretrained_path)
    logger.info("输出目录：%s", config.output_root)
    logger.info("设备：%s", device_label())

    train_all(config, task_keys)
    logger.info("全部训练完成，模型目录：%s", config.output_root)
    return 0


def _command_predict(parsed: argparse.Namespace) -> int:
    config = load_config(parsed.config)

    if parsed.model_dir:
        model_dir = Path(parsed.model_dir).expanduser().resolve()
    elif parsed.task:
        model_dir = config.best_model_dir(parsed.task)
    else:
        raise SystemExit("请指定 --task 或 --model-dir 以定位模型")

    classifier = GenreClassifier(
        model_dir=model_dir,
        device=parsed.device,
        max_length=config.model.max_length,
    )
    results = classifier.predict(parsed.text, top_k=parsed.top_k, threshold=parsed.threshold)

    if parsed.json:
        print(json.dumps({"model_dir": str(model_dir), "results": results}, ensure_ascii=False, indent=2))
    else:
        for text, result in zip(parsed.text, results):
            logger.info("文本: %s", text[:40] + ("..." if len(text) > 40 else ""))
            if classifier.is_multi_label:
                logger.info("  预测类型: %s", result["labels"] or ["<无>"] )
                logger.info("  对应概率: %s", result["scores"])
            else:
                logger.info("  预测类型: %s (%.4f)", result["label"], result["score"])
                logger.info("  Top-%d: %s", len(result["top_k"]), result["top_k"])
    return 0


def _predict_targets(config, task_key: str) -> List[Path]:
    """列出该任务的待导出目录：最优模型 + 全部压缩变体。"""
    targets = [config.best_model_dir(task_key)]
    optimized_dir = config.optimized_dir(task_key)
    if optimized_dir.exists():
        targets.extend(
            sorted(path for path in optimized_dir.iterdir() if (path / "optimization.json").exists())
        )
    return targets


def _export_one(
    config,
    task_key: str,
    model_dir: Path,
    texts,
    labels,
    batch_size: int,
    device_arg: str | None,
    overwrite: bool,
) -> None:
    """对单个模型目录导出逐条预测。"""
    labels_file = model_dir / "labels.json"
    if not labels_file.exists():
        logger.warning("跳过 %s：缺少 labels.json", model_dir)
        return
    if not overwrite and (model_dir / PREDICTIONS_FILE).exists():
        logger.info("跳过 %s：预测结果已存在（--overwrite 可强制重算）", model_dir)
        return

    labels_payload = json.loads(labels_file.read_text(encoding="utf-8"))
    classes = labels_payload["labels"]
    max_length = int(labels_payload.get("max_length", config.model.max_length))
    threshold = float(labels_payload.get("threshold", config.task(task_key).threshold))
    top_k = int(labels_payload.get("top_k", config.task(task_key).top_k))

    meta_file = model_dir / "optimization.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        model = load_variant(model_dir)
        # 动态 INT8 只支持 CPU
        default_device = "cpu" if meta["method"] == "int8" else ("cuda" if torch.cuda.is_available() else "cpu")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        default_device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_arg or default_device)
    model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    logits = predict_logits(
        model,
        tokenizer,
        texts,
        max_length=max_length,
        device=device,
        batch_size=batch_size,
    )
    save_predictions(
        target_dir=model_dir,
        texts=texts,
        true_labels=labels,
        logits=logits,
        classes=classes,
        task_key=task_key,
        threshold=threshold,
        top_k=top_k,
        encoding=config.data.export_encoding,
    )


def _command_predict_set(parsed: argparse.Namespace) -> int:
    config = load_config(parsed.config)
    batch_size = parsed.batch_size or config.optimization.eval_batch_size

    if parsed.model_dir:
        target = Path(parsed.model_dir).expanduser().resolve()
        task_key = parsed.task if parsed.task in TASK_KEYS else SINGLE_LABEL
        texts, labels = prepare_data(config, task_key)["splits"][parsed.split]
        _export_one(config, task_key, target, texts, labels, batch_size, parsed.device, parsed.overwrite)
        return 0

    task_keys = list(TASK_KEYS) if parsed.task == "all" else [parsed.task]
    for task_key in task_keys:
        texts, labels = prepare_data(config, task_key)["splits"][parsed.split]
        for model_dir in _predict_targets(config, task_key):
            logger.info("[%s] 导出预测：%s", task_key, model_dir)
            _export_one(
                config, task_key, model_dir, texts, labels, batch_size, parsed.device, parsed.overwrite
            )
    logger.info("逐条预测结果导出完成")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    parsed = parser.parse_args(argv)
    setup_logging(log_file=parsed.log_file)

    if parsed.command == "train":
        return _command_train(parsed)
    if parsed.command == "predict":
        return _command_predict(parsed)
    if parsed.command == "predict-set":
        return _command_predict_set(parsed)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
