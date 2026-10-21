"""评估指标：单标签多分类与多标签分类两套实现。

既提供给 transformers Trainer 的 compute_metrics，也供压缩实验直接调用。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


def _split(eval_pred) -> Tuple[np.ndarray, np.ndarray]:
    """从 Trainer 传入的 EvalPrediction 中取出 logits 与标签。"""
    logits = np.asarray(eval_pred.predictions)
    labels = np.asarray(eval_pred.label_ids)
    return logits, labels


def single_label_scores(logits: np.ndarray, labels: np.ndarray, top_k: int = 3) -> Dict[str, float]:
    """单标签（多分类）指标：准确率、Macro/Weighted F1、Top-K 命中率。"""
    logits = np.asarray(logits)
    labels = np.asarray(labels).astype(int)
    predictions = logits.argmax(axis=-1)

    k = int(min(top_k, logits.shape[-1]))
    top_k_indices = np.argsort(-logits, axis=-1)[:, :k]
    top_k_hit = np.any(top_k_indices == labels[:, None], axis=1)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        f"top{k}_accuracy": float(top_k_hit.mean()),
    }


def multi_label_scores(
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    thresholds: Sequence[float] | None = None,
) -> Dict[str, float]:
    """多标签指标：对每个类别独立做 sigmoid 阈值判定。

    Args:
        threshold: 统一的判定阈值（thresholds 为空时生效）。
        thresholds: 逐类别阈值（长度 = 类别数），优先级高于 threshold。
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels).astype(int)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    if thresholds is not None:
        cut = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
        predictions = (probabilities >= cut).astype(int)
        reported = float(np.mean(cut))
    else:
        predictions = (probabilities >= threshold).astype(int)
        reported = float(threshold)

    return {
        # 完全匹配率：一个样本的所有类型都预测正确才算对
        "subset_accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        # 汉明损失：错误判定的类别占比，越低越好
        "hamming_loss": float(hamming_loss(labels, predictions)),
        "threshold": reported,
    }


def tune_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    default: float = 0.5,
    grid: Sequence[float] | None = None,
) -> List[float]:
    """在验证集上逐类别搜索使该类 F1 最大的阈值。

    稀有类别在验证集上可能没有正样本，此时沿用 default，避免阈值被噪声带偏。
    """
    probabilities = np.asarray(probabilities, dtype=np.float32)
    labels = np.asarray(labels).astype(int)
    candidates = np.asarray(grid if grid is not None else np.arange(0.05, 0.96, 0.05), dtype=np.float32)

    tuned: List[float] = []
    for index in range(labels.shape[1]):
        target = labels[:, index]
        if target.sum() == 0:
            tuned.append(float(default))
            continue
        best_threshold, best_score = float(default), -1.0
        for candidate in candidates:
            predicted = (probabilities[:, index] >= candidate).astype(int)
            score = f1_score(target, predicted, zero_division=0)
            if score > best_score:
                best_score, best_threshold = float(score), float(candidate)
        tuned.append(best_threshold)
    return tuned


def single_label_metrics(eval_pred, top_k: int = 3) -> Dict[str, float]:
    """Trainer 适配层：单标签指标。"""
    logits, labels = _split(eval_pred)
    return single_label_scores(logits, labels, top_k=top_k)


def make_multi_label_metrics(threshold: float = 0.5) -> Callable[..., Dict[str, float]]:
    """Trainer 适配层：多标签指标。"""

    def _metrics(eval_pred) -> Dict[str, float]:
        logits, labels = _split(eval_pred)
        return multi_label_scores(logits, labels, threshold=threshold)

    return _metrics
