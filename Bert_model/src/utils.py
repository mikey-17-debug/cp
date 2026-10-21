"""通用工具：日志、随机种子、设备信息。"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    """初始化统一日志格式，并压低 transformers 的冗余输出。

    Args:
        level: 控制台日志级别。
        log_file: 可选的日志文件路径，便于另开窗口训练时回看进度。
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    # 仅屏蔽权重加载报告，保留 tqdm 进度条（另开窗口时可直观看到训练进度）
    try:
        import transformers

        transformers.logging.set_verbosity_error()
    except Exception:  # pragma: no cover - 仅影响日志观感
        pass


def set_seed(seed: int) -> None:
    """固定随机种子，保证切分与训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_label() -> str:
    """返回当前可用设备的可读描述，用于日志。"""
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        return f"cuda:{index} ({name})"
    return "cpu"
