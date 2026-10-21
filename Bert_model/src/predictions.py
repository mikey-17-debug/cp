"""逐样本预测结果的解码与落盘。

保存内容（每个模型目录下）：
    test_predictions.csv   每条样本的真实标签、预测标签、置信度、是否正确
    test_probabilities.npy 形状 (样本数, 类别数) 的概率矩阵，便于后续阈值调优
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PREDICTIONS_FILE = "test_predictions.csv"
PROBABILITIES_FILE = "test_probabilities.npy"


def class_probabilities(logits: np.ndarray, task_key: str) -> np.ndarray:
    """把 logits 转成概率：单标签用 softmax，多标签逐类 sigmoid。"""
    logits = np.asarray(logits, dtype=np.float32)
    if task_key == "single":
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def _decode_single(
    probabilities: np.ndarray,
    classes: Sequence[str],
    top_k: int,
) -> Tuple[List[str], List[float], List[str]]:
    k = int(min(top_k, len(classes)))
    labels: List[str] = []
    scores: List[float] = []
    top_k_text: List[str] = []
    for row in probabilities:
        order = np.argsort(-row)[:k]
        labels.append(classes[int(order[0])])
        scores.append(float(row[order[0]]))
        top_k_text.append("|".join(f"{classes[int(i)]}:{row[i]:.4f}" for i in order))
    return labels, scores, top_k_text


def _decode_multi(
    probabilities: np.ndarray,
    classes: Sequence[str],
    threshold: float,
    thresholds: Dict[str, float] | None = None,
) -> Tuple[List[str], List[float]]:
    labels: List[str] = []
    scores: List[float] = []
    cut = np.asarray([thresholds.get(name, threshold) for name in classes]) if thresholds else None
    for row in probabilities:
        if cut is not None:
            selected = [index for index, value in enumerate(row) if value >= cut[index]]
        else:
            selected = [index for index, value in enumerate(row) if value >= threshold]
        labels.append("|".join(classes[index] for index in selected))
        scores.append(float(row.max()) if row.size else 0.0)
    return labels, scores


def _true_labels_text(
    labels: np.ndarray,
    classes: Sequence[str],
    task_key: str,
) -> List[str]:
    if task_key == "single":
        return [classes[int(index)] for index in labels]
    return [
        "|".join(classes[index] for index, flag in enumerate(row) if flag > 0.5)
        for row in labels
    ]


def save_predictions(
    target_dir: Path,
    texts: Sequence[str],
    true_labels: np.ndarray,
    logits: np.ndarray,
    classes: Sequence[str],
    task_key: str,
    threshold: float,
    top_k: int,
    encoding: str = "utf-8-sig",
    thresholds: Dict[str, float] | None = None,
) -> Path:
    """把逐条样本的预测结果与概率写入 target_dir。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    probabilities = class_probabilities(logits, task_key)
    truth = _true_labels_text(true_labels, classes, task_key)

    if task_key == "single":
        predicted, scores, top_k_text = _decode_single(probabilities, classes, top_k)
        frame = pd.DataFrame(
            {
                "row": np.arange(len(texts)),
                "summary": list(texts),
                "true_label": truth,
                "pred_label": predicted,
                "confidence": scores,
                "top_k": top_k_text,
                "correct": [t == p for t, p in zip(truth, predicted)],
            }
        )
    else:
        predicted, scores = _decode_multi(probabilities, classes, threshold, thresholds)
        frame = pd.DataFrame(
            {
                "row": np.arange(len(texts)),
                "summary": list(texts),
                "true_labels": truth,
                "pred_labels": predicted,
                "max_probability": scores,
                "num_pred": [len(item.split("|")) if item else 0 for item in predicted],
                "subset_correct": [t == p for t, p in zip(truth, predicted)],
            }
        )

    csv_path = target_dir / PREDICTIONS_FILE
    frame.to_csv(csv_path, index=False, encoding=encoding)
    np.save(target_dir / PROBABILITIES_FILE, probabilities)
    logger.info("预测结果已写入 %s（%d 条）", csv_path, len(frame))
    return csv_path


def load_predictions(target_dir: Path, encoding: str = "utf-8-sig") -> Dict[str, object]:
    """读取已保存的预测结果，便于后续分析。"""
    csv_path = target_dir / PREDICTIONS_FILE
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到预测结果文件：{csv_path}")
    frame = pd.read_csv(csv_path, encoding=encoding)
    probabilities_path = target_dir / PROBABILITIES_FILE
    probabilities = np.load(probabilities_path) if probabilities_path.exists() else None
    return {"frame": frame, "probabilities": probabilities}
