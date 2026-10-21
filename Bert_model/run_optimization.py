"""压缩实验入口：对已训练好的单标签 / 多标签模型做量化、剪枝、低秩分解。

用法（在 Bert_model 目录下执行）：

    python run_optimization.py --task all
    python run_optimization.py --task single --report ../report.md

产物：
    * 压缩模型：<output_root>/<task>/optimized/<variant>/
    * 结果 JSON：config.optimization.results_path
    * Markdown 报告：config.optimization.report_path
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.benchmark import (  # noqa: E402
    directory_size_bytes,
    evaluate_accuracy,
    measure_latency,
    predict_logits,
)
from src.config import MULTI_LABEL, SINGLE_LABEL, TASK_KEYS, load_config  # noqa: E402
from src.data import prepare_data  # noqa: E402
from src.optimization import (  # noqa: E402
    VariantSpec,
    apply_variant,
    count_parameters,
    linear_weight_sparsity,
    load_variant,
    make_variants,
    save_variant,
)
from src.predictions import save_predictions  # noqa: E402
from src.report import build_markdown  # noqa: E402
from src.utils import device_label, setup_logging  # noqa: E402

logger = logging.getLogger("optimization")


def _environment(config) -> Dict[str, Any]:
    gpu = device_label()
    cpu = platform.processor() or platform.machine()
    return {
        "gpu": gpu,
        "cpu": cpu,
        "cpu_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "latency_batch_size": config.optimization.latency_batch_size,
    }


def _measure_variant(
    model,
    tokenizer,
    texts,
    max_length,
    device: torch.device,
    batch_size: int,
    samples: int,
    rounds: int,
) -> Dict[str, float] | None:
    sample_texts = texts[:samples]
    if not sample_texts:
        return None
    return measure_latency(
        model,
        tokenizer,
        sample_texts,
        max_length=max_length,
        device=device,
        batch_size=batch_size,
        rounds=rounds,
    )


def _verify_saved(target_dir: Path, expected_logits: int) -> bool:
    """确认落盘产物可以被重新加载并正常前向。"""
    try:
        meta = json.loads((target_dir / "optimization.json").read_text(encoding="utf-8"))
        # 动态 INT8 模型只支持 CPU，不能迁移到 GPU
        device = torch.device("cpu")
        if meta["method"] != "int8" and torch.cuda.is_available():
            device = torch.device("cuda")

        model = load_variant(target_dir).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(str(target_dir), use_fast=True)
        encoded = tokenizer(["测试文本"], truncation=True, max_length=16, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
        return int(logits.shape[-1]) == expected_logits
    except Exception as error:  # pragma: no cover - 仅用于产物校验
        logger.warning("产物校验失败 %s：%s", target_dir.name, error)
        return False


def run_task(config, task_key: str) -> Dict[str, Any]:
    optimization = config.optimization
    base_dir = config.best_model_dir(task_key)
    if not base_dir.exists():
        raise FileNotFoundError(f"未找到 {task_key} 任务的基线模型：{base_dir}，请先执行训练")

    labels_payload = json.loads((base_dir / "labels.json").read_text(encoding="utf-8"))
    label_count = len(labels_payload["labels"])
    max_length = int(labels_payload.get("max_length", config.model.max_length))
    threshold = float(labels_payload.get("threshold", 0.5))
    top_k = int(labels_payload.get("top_k", 3))
    # 训练阶段已在验证集上逐类校准阈值，压缩实验沿用同一套阈值
    tuned_thresholds = labels_payload.get("thresholds") or {}
    ordered_thresholds = (
        [tuned_thresholds.get(name, threshold) for name in labels_payload["labels"]]
        if tuned_thresholds
        else None
    )

    prepared = prepare_data(config, task_key)
    test_texts, test_labels = prepared["splits"]["test"]
    if optimization.max_test_samples:
        test_texts = test_texts[: optimization.max_test_samples]
        test_labels = test_labels[: optimization.max_test_samples]
        logger.warning("[%s] 调试模式：仅使用前 %d 条测试样本", task_key, len(test_texts))

    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), use_fast=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(str(base_dir)).eval()
    base_params = count_parameters(base_model)

    use_cuda = torch.cuda.is_available() and not config.training.use_cpu
    gpu = torch.device("cuda" if use_cuda else "cpu")
    cpu = torch.device("cpu")

    logger.info(
        "[%s] 基线模型 %s | 测试样本 %d | 类别 %d | 参数 %.1fM",
        task_key,
        base_dir,
        len(test_texts),
        label_count,
        base_params / 1e6,
    )

    variants = make_variants(config)
    results: List[Dict[str, Any]] = []
    base_size = directory_size_bytes(base_dir)

    for spec in variants:
        logger.info("-" * 60)
        logger.info("[%s] 变体 %s：%s", task_key, spec.name, spec.description)

        model = apply_variant(base_model, spec, config)
        target_device = cpu if spec.method == "int8" or not use_cuda else gpu
        model.to(target_device).eval()

        params = base_params if spec.method == "int8" else count_parameters(model)
        sparsity = linear_weight_sparsity(model) if spec.method == "unstructured" else 0.0

        saved_path: Path | None = None
        if spec.name == "fp32":
            size_bytes = base_size
        else:
            saved_path = save_variant(
                model=model,
                spec=spec,
                target_dir=config.optimized_dir(task_key) / spec.name,
                base_model_dir=base_dir,
                tokenizer=tokenizer,
                labels_payload=labels_payload,
            )
            size_bytes = directory_size_bytes(saved_path)

        metrics = evaluate_accuracy(
            model,
            tokenizer,
            test_texts,
            test_labels,
            task_key=task_key,
            max_length=max_length,
            device=target_device,
            batch_size=optimization.eval_batch_size,
            threshold=threshold,
            top_k=top_k,
            thresholds=ordered_thresholds,
        )

        # 导出逐条样本的预测结果（fp32 基线写到任务目录，其余写到变体目录）
        prediction_logits = predict_logits(
            model,
            tokenizer,
            test_texts,
            max_length=max_length,
            device=target_device,
            batch_size=optimization.eval_batch_size,
        )
        prediction_dir = saved_path if saved_path else config.task_output_dir(task_key)
        save_predictions(
            target_dir=prediction_dir,
            texts=test_texts,
            true_labels=test_labels,
            logits=prediction_logits,
            classes=labels_payload["labels"],
            task_key=task_key,
            threshold=threshold,
            top_k=top_k,
            encoding=config.data.export_encoding,
            thresholds=tuned_thresholds or None,
        )

        cpu_latency = None
        gpu_latency = None
        gpu_throughput = None

        if use_cuda and spec.method != "int8":
            single = _measure_variant(
                model, tokenizer, test_texts, max_length, gpu,
                batch_size=1,
                samples=optimization.latency_single_samples,
                rounds=optimization.latency_rounds,
            )
            gpu_latency = single["latency_ms"] if single else None
            throughput = _measure_variant(
                model, tokenizer, test_texts, max_length, gpu,
                batch_size=optimization.latency_batch_size,
                samples=optimization.latency_samples,
                rounds=optimization.latency_rounds,
            )
            gpu_throughput = throughput["throughput"] if throughput else None

        # CPU 延迟/吞吐：除 INT8（只测 CPU）外的变体都测；半精度在 CPU 上通常更慢，如实记录
        cpu_latency = None
        cpu_throughput = None
        if spec.method in {"baseline", "dtype", "int8", "unstructured", "structured", "low_rank"}:
            try:
                model.to(cpu).eval()
                cpu_measure = _measure_variant(
                    model, tokenizer, test_texts, max_length, cpu,
                    batch_size=1,
                    samples=optimization.latency_single_samples,
                    rounds=optimization.latency_rounds,
                )
                cpu_latency = cpu_measure["latency_ms"] if cpu_measure else None
                # 吞吐只跑一轮，避免 CPU 测量耗时过长
                cpu_batch = _measure_variant(
                    model, tokenizer, test_texts, max_length, cpu,
                    batch_size=optimization.latency_batch_size,
                    samples=optimization.latency_samples,
                    rounds=1,
                )
                cpu_throughput = cpu_batch["throughput"] if cpu_batch else None
            except RuntimeError as error:
                logger.warning("[%s] %s 的 CPU 测量不可用：%s", task_key, spec.name, error)

        verified = _verify_saved(saved_path, label_count) if saved_path else True

        record = {
            "name": spec.name,
            "method": spec.method,
            "description": spec.description,
            "param": spec.param,
            "params": params,
            "size_bytes": size_bytes,
            "size_ratio": size_bytes / base_size,
            "sparsity": sparsity,
            "metrics": metrics,
            "cpu_latency_ms": cpu_latency,
            "cpu_throughput": cpu_throughput,
            "gpu_latency_ms": gpu_latency,
            "gpu_throughput": gpu_throughput,
            "saved": saved_path is not None,
            "path": str(saved_path) if saved_path else None,
            "load_ok": verified,
        }
        results.append(record)
        logger.info(
            "[%s] %-22s size=%.1fMB ratio=%.1f%% cpu=%sms gpu=%sms %s",
            task_key,
            spec.name,
            size_bytes / 1024 ** 2,
            record["size_ratio"] * 100,
            f"{cpu_latency:.1f}" if cpu_latency else "-",
            f"{gpu_latency:.1f}" if gpu_latency else "-",
            {k: round(v, 4) for k, v in metrics.items() if k in ("accuracy", "macro_f1", "micro_f1", "top3_accuracy", "subset_accuracy", "hamming_loss")},
        )

        del model
        if use_cuda:
            torch.cuda.empty_cache()

    return {
        "baseline_dir": str(base_dir),
        "label_count": label_count,
        "test_size": len(test_texts),
        "threshold": threshold,
        "threshold_mode": "per_class" if ordered_thresholds else "global",
        "variants": results,
    }


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对训练好的模型执行量化 / 剪枝 / 低秩分解并生成报告",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    parser.add_argument("--task", choices=[*TASK_KEYS, "all"], default="all")
    parser.add_argument("--report", help="覆盖 optimization.report_path")
    parser.add_argument("--results", help="覆盖 optimization.results_path")
    parser.add_argument("--log-file", help="日志文件路径")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging(log_file=args.log_file)
    config = load_config(args.config)

    report_path = Path(args.report).expanduser().resolve() if args.report else config.report_path
    results_path = Path(args.results).expanduser().resolve() if args.results else config.results_path

    task_keys = list(TASK_KEYS) if args.task == "all" else [args.task]
    logger.info("开始压缩实验：%s", task_keys)
    logger.info("设备：%s", device_label())

    payload: Dict[str, Any] = {
        "generated_at": None,
        "project": {
            "model_name": Path(config.model.pretrained_path).name,
            "dataset_name": config.data.display_name or Path(config.data_path).name,
            "dataset_path": str(config.data_path),
            "output_root": str(config.output_root),
            "output_root_name": Path(config.output_root).name,
            "config_file": Path(args.config).name,
        },
        "environment": _environment(config),
        "tasks": {},
    }
    for task_key in task_keys:
        payload["tasks"][task_key] = run_task(config, task_key)
    payload["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_markdown(payload), encoding="utf-8")

    logger.info("结果已写入 %s", results_path)
    logger.info("报告已写入 %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
