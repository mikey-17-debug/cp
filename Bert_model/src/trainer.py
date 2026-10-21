"""训练层：把配置、数据、模型、指标装配成一次完整训练，并落盘最优模型。

输出目录结构（<output_root>/<tasks.*.output_dir>）：
    best/               最优模型（config.json / model.safetensors / 分词器文件）
    labels.json         标签空间与推理参数
    metrics.json        验证集 / 测试集指标与配置快照
    checkpoint-*        训练中间检查点（默认不保留）
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Callable, Dict

import numpy as np
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments

from .config import MULTI_LABEL, PROBLEM_TYPES, SINGLE_LABEL, ProjectConfig
from .data import DynamicPadCollator, GenreDataset, build_tokenizer, prepare_data
from .metrics import (
    make_multi_label_metrics,
    multi_label_scores,
    single_label_metrics,
    single_label_scores,
    tune_thresholds,
)
from .model import build_model
from .predictions import save_predictions
from .utils import device_label, set_seed

logger = logging.getLogger(__name__)


class ProgressLogCallback(TrainerCallback):
    """把训练步进度同步写进项目日志，便于在日志文件里跟踪训练进展。"""

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
        if not logs or "loss" not in logs:
            return
        logger.info(
            "[进度] step %d/%d | epoch %.2f | loss %.4f | lr %.3g",
            state.global_step,
            state.max_steps,
            state.epoch or 0.0,
            logs["loss"],
            logs.get("learning_rate", 0.0),
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: D401
        if not metrics:
            return
        readable = " | ".join(
            f"{key}={value:.4f}" for key, value in metrics.items() if isinstance(value, (int, float))
        )
        logger.info("[验证] epoch %.2f | %s", state.epoch or 0.0, readable)


def _build_training_arguments(
    config: ProjectConfig,
    task_key: str,
    output_dir: Path,
    warmup_steps: int,
) -> TrainingArguments:
    """按配置生成 TrainingArguments（不含任何硬编码超参）。"""
    training = config.training
    task = config.task(task_key)
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=training.per_device_train_batch_size,
        per_device_eval_batch_size=training.per_device_eval_batch_size,
        learning_rate=training.learning_rate,
        num_train_epochs=training.num_train_epochs,
        weight_decay=training.weight_decay,
        optim=training.optim,
        warmup_steps=warmup_steps,
        max_grad_norm=training.max_grad_norm,
        bf16=training.bf16,
        fp16=training.fp16,
        use_cpu=training.use_cpu,
        seed=training.seed,
        data_seed=training.seed,
        logging_steps=training.logging_steps,
        eval_strategy=training.eval_strategy,
        save_strategy=training.save_strategy,
        save_total_limit=training.save_total_limit,
        load_best_model_at_end=training.load_best_model_at_end,
        metric_for_best_model=task.metric_for_best_model,
        greater_is_better=task.greater_is_better,
        dataloader_num_workers=training.dataloader_num_workers,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        report_to=training.report_to,
        # 数据集返回的是张量字典，无需按列名裁剪
        remove_unused_columns=False,
    )


def _warmup_steps(config: ProjectConfig, train_size: int) -> int:
    """transformers 5.x 已移除 warmup_ratio，这里按比例换算成步数。"""
    training = config.training
    batch = max(
        1,
        training.per_device_train_batch_size * training.gradient_accumulation_steps,
    )
    steps_per_epoch = max(1, math.ceil(train_size / batch))
    total_steps = max(1, int(steps_per_epoch * training.num_train_epochs))
    return int(total_steps * training.warmup_ratio)


def _metrics_function(task_key: str, config: ProjectConfig) -> Callable[..., Dict[str, float]]:
    task = config.task(task_key)
    if task_key == SINGLE_LABEL:
        return lambda eval_pred: single_label_metrics(eval_pred, top_k=task.top_k)
    return make_multi_label_metrics(threshold=task.threshold)


def _save_outputs(
    config: ProjectConfig,
    task_key: str,
    trainer: Trainer,
    classes,
    metrics: Dict[str, float],
    thresholds: Dict[str, float] | None = None,
) -> Path:
    """保存最优模型、标签空间与训练指标，并清理中间检查点。"""
    task_dir = config.task_output_dir(task_key)
    best_dir = config.best_model_dir(task_key)
    trainer.save_model(str(best_dir))
    if trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(str(best_dir))

    task = config.task(task_key)
    labels_payload = {
        "task": task_key,
        "problem_type": PROBLEM_TYPES[task_key],
        "labels": list(classes),
        "id2label": {str(index): label for index, label in enumerate(classes)},
        "top_k": task.top_k,
        "threshold": task.threshold,
        "max_length": config.model.max_length,
    }
    if thresholds:
        labels_payload["thresholds"] = thresholds
    for target in (best_dir, task_dir):
        (target / "labels.json").write_text(
            json.dumps(labels_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    summary = {
        "task": task_key,
        "metrics": metrics,
        "label_count": len(classes),
        "config": config.snapshot(),
    }
    (task_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for checkpoint in task_dir.glob("checkpoint-*"):
        shutil.rmtree(checkpoint, ignore_errors=True)
    logger.info("[%s] 最优模型已保存到 %s", task_key, best_dir)
    return best_dir


def _save_test_predictions(
    config: ProjectConfig,
    task_key: str,
    texts,
    true_labels: np.ndarray,
    logits: np.ndarray,
    classes,
    thresholds: Dict[str, float] | None = None,
) -> None:
    """用训练后的最优模型导出测试集逐条预测结果。"""
    task = config.task(task_key)
    save_predictions(
        target_dir=config.task_output_dir(task_key),
        texts=texts,
        true_labels=true_labels,
        logits=logits,
        classes=classes,
        task_key=task_key,
        threshold=task.threshold,
        top_k=task.top_k,
        encoding=config.data.export_encoding,
        thresholds=thresholds,
    )


def _tune_multi_label_thresholds(
    config: ProjectConfig,
    trainer: Trainer,
    val_dataset: GenreDataset,
    classes,
) -> Dict[str, float] | None:
    """在验证集上逐类搜索阈值，返回 {标签: 阈值}；单标签任务返回 None。"""
    prediction = trainer.predict(val_dataset)
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(prediction.predictions, dtype=np.float32)))
    threshold = config.task(MULTI_LABEL).threshold
    tuned = tune_thresholds(probabilities, np.asarray(prediction.label_ids), default=threshold)
    payload = {label: value for label, value in zip(classes, tuned)}
    readable = ", ".join(f"{label}={value:.2f}" for label, value in payload.items())
    logger.info("[multi] 验证集逐类阈值校准：%s", readable)
    return payload


def run_training(
    config: ProjectConfig,
    task_key: str,
    resume_from_checkpoint: str | Path | None = None,
) -> Dict[str, float]:
    """执行单个任务的完整训练流程，返回测试集指标。"""
    set_seed(config.training.seed)
    task = config.task(task_key)
    logger.info("=" * 70)
    logger.info("[%s] 开始训练 | 设备 %s | 输出 %s", task_key, device_label(), config.task_output_dir(task_key))
    if resume_from_checkpoint:
        logger.info("[%s] 从检查点续训：%s", task_key, resume_from_checkpoint)
    logger.info("=" * 70)

    tokenizer = build_tokenizer(config)
    prepared = prepare_data(config, task_key)
    classes = prepared["classes"]
    splits = prepared["splits"]
    problem_type = PROBLEM_TYPES[task_key]

    datasets = {
        name: GenreDataset(texts, labels, tokenizer, config.model.max_length, problem_type)
        for name, (texts, labels) in splits.items()
    }

    model = build_model(config, task_key, classes)
    args = _build_training_arguments(
        config,
        task_key,
        config.task_output_dir(task_key),
        warmup_steps=_warmup_steps(config, len(datasets["train"])),
    )
    logger.info(
        "[%s] 训练参数：epochs=%s, batch=%d, lr=%s, warmup_steps=%d, max_length=%d",
        task_key,
        args.num_train_epochs,
        args.per_device_train_batch_size,
        args.learning_rate,
        args.warmup_steps,
        config.model.max_length,
    )

    callbacks = [ProgressLogCallback()]
    patience = config.training.early_stopping_patience
    if patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
        logger.info(
            "[%s] 早停已启用：连续 %d 轮 %s 无提升即停止",
            task_key,
            patience,
            task.metric_for_best_model,
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=DynamicPadCollator(tokenizer.pad_token_id),
        compute_metrics=_metrics_function(task_key, config),
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)

    metrics: Dict[str, float] = {}
    metrics.update(trainer.evaluate(datasets["val"], metric_key_prefix="val"))

    # 多标签任务：在验证集上逐类校准阈值，测试指标与预测都使用校准后的阈值
    thresholds = _tune_multi_label_thresholds(config, trainer, datasets["val"], classes) \
        if task_key == MULTI_LABEL else None

    prediction = trainer.predict(datasets["test"])
    test_logits = np.asarray(prediction.predictions)
    test_labels = np.asarray(prediction.label_ids)
    if task_key == SINGLE_LABEL:
        scores = single_label_scores(test_logits, test_labels, top_k=task.top_k)
    else:
        scores = multi_label_scores(
            test_logits,
            test_labels,
            threshold=task.threshold,
            thresholds=list(thresholds.values()) if thresholds else None,
        )
    metrics.update({f"test_{key}": value for key, value in scores.items()})
    if "test_loss" in prediction.metrics:
        metrics["test_loss"] = float(prediction.metrics["test_loss"])

    for name, value in sorted(metrics.items()):
        logger.info("[%s] %-24s %s", task_key, name, value)

    _save_outputs(config, task_key, trainer, classes, metrics, thresholds)
    _save_test_predictions(
        config,
        task_key,
        datasets["test"].texts,
        test_labels,
        test_logits,
        classes,
        thresholds,
    )
    logger.info("[%s] 完成，按 %s=%s 选取的最优模型已就绪", task_key, task.metric_for_best_model,
                metrics.get(f"test_{task.metric_for_best_model}"))
    return metrics


def train_all(
    config: ProjectConfig,
    task_keys,
    resume_checkpoints: Dict[str, str | Path] | None = None,
) -> Dict[str, Dict[str, float]]:
    """按顺序训练多个任务，可为每个任务指定续训检查点。"""
    resume_checkpoints = resume_checkpoints or {}
    results: Dict[str, Dict[str, float]] = {}
    for task_key in task_keys:
        results[task_key] = run_training(
            config,
            task_key,
            resume_from_checkpoint=resume_checkpoints.get(task_key),
        )
    return results
