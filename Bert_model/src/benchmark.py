"""压缩实验的测量工具：精度、推理延迟、吞吐与体积。"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

from .metrics import multi_label_scores, single_label_scores

logger = logging.getLogger(__name__)


def directory_size_bytes(path: Path) -> int:
    """目录内所有文件的总字节数。"""
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _encode(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    max_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(texts),
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return {
        key: value.to(device)
        for key, value in encoded.items()
        if key in ("input_ids", "attention_mask")
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_logits(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    max_length: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """批量前向，返回拼接后的 logits。"""
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            batch = _encode(tokenizer, chunk, max_length, device)
            logits = model(**batch).logits
            outputs.append(logits.float().detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate_accuracy(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    labels: np.ndarray,
    task_key: str,
    max_length: int,
    device: torch.device,
    batch_size: int,
    threshold: float = 0.5,
    top_k: int = 3,
    thresholds: Sequence[float] | None = None,
) -> Dict[str, float]:
    """在测试集上计算任务对应指标。"""
    logits = predict_logits(model, tokenizer, texts, max_length, device, batch_size)
    if task_key == "single":
        return single_label_scores(logits, labels, top_k=top_k)
    return multi_label_scores(logits, labels, threshold=threshold, thresholds=thresholds)


def measure_latency(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    max_length: int,
    device: torch.device,
    batch_size: int,
    rounds: int,
) -> Dict[str, float]:
    """测量纯前向延迟（不含分词），返回单条延迟与吞吐。"""
    model.eval()
    batches = [
        _encode(tokenizer, texts[start : start + batch_size], max_length, device)
        for start in range(0, len(texts), batch_size)
    ]
    sample_count = sum(batch["input_ids"].shape[0] for batch in batches)

    with torch.inference_mode():
        for batch in batches[:1]:  # 预热
            model(**batch)
        _sync(device)
        start_time = time.perf_counter()
        for _ in range(rounds):
            for batch in batches:
                model(**batch)
        _sync(device)
        elapsed = time.perf_counter() - start_time

    total = sample_count * rounds
    return {
        "latency_ms": elapsed / total * 1000,
        "throughput": total / elapsed,
        "samples": sample_count,
        "rounds": rounds,
        "batch_size": batch_size,
        "device": str(device),
    }
