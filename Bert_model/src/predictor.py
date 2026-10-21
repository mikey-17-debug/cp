"""推理层：加载已保存的模型目录并对文本做类型预测。

预测行为由保存下来的 labels.json 决定：
    * single_label_classification -> softmax + Top-K
    * multi_label_classification  -> sigmoid + 阈值过滤
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class GenreClassifier:
    """电影类型分类器。"""

    def __init__(
        self,
        model_dir: str | Path,
        device: str | None = None,
        max_length: int | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在：{self.model_dir}")

        meta_file = self.model_dir / "labels.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"缺少标签信息文件：{meta_file}")
        self.meta: Dict[str, Any] = json.loads(meta_file.read_text(encoding="utf-8"))

        self.labels: List[str] = list(self.meta["labels"])
        self.problem_type: str = self.meta["problem_type"]
        self.top_k: int = int(self.meta.get("top_k", 3))
        self.threshold: float = float(self.meta.get("threshold", 0.5))
        # 逐类别校准后的阈值（训练结束时在验证集上搜索得到）
        self.thresholds: Dict[str, float] = dict(self.meta.get("thresholds") or {})
        self.max_length: int = int(max_length or self.meta.get("max_length", 512))

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device).eval()
        logger.info(
            "已加载 %s（%s，%d 个标签）到 %s",
            self.model_dir,
            self.problem_type,
            len(self.labels),
            self.device,
        )

    @property
    def is_multi_label(self) -> bool:
        return self.problem_type == "multi_label_classification"

    def predict(
        self,
        texts: Union[str, Sequence[str]],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """预测类型。

        Args:
            texts: 单条文本或文本列表（通常是剧情简介 summary）。
            top_k: 单标签任务返回的候选数量，默认取 labels.json 中的配置。
            threshold: 多标签任务的判定阈值，默认取 labels.json 中的配置。

        Returns:
            单条文本传入时返回 dict，多条传入时返回等长 list。
        """
        single_input = isinstance(texts, str)
        batch = [texts] if single_input else list(texts)
        if not batch:
            return []

        encoded = self.tokenizer(
            batch,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits

        results = self._decode(logits.cpu().numpy(), top_k, threshold)
        return results[0] if single_input else results

    def _decode(
        self,
        logits: np.ndarray,
        top_k: int | None,
        threshold: float | None,
    ) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        if self.is_multi_label:
            if threshold is not None:
                cut: float | np.ndarray = threshold
            elif self.thresholds:
                cut = np.asarray([self.thresholds.get(label, self.threshold) for label in self.labels])
            else:
                cut = self.threshold
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            for row in probabilities:
                order = np.argsort(-row)
                if isinstance(cut, np.ndarray):
                    selected = [int(index) for index in order if row[index] >= cut[index]]
                else:
                    selected = [int(index) for index in order if row[index] >= cut]
                outputs.append(
                    {
                        "labels": [self.labels[index] for index in selected],
                        "scores": [round(float(row[index]), 6) for index in selected],
                    }
                )
        else:
            k = int(min(top_k or self.top_k, len(self.labels)))
            scores = np.exp(logits - logits.max(axis=-1, keepdims=True))
            scores = scores / scores.sum(axis=-1, keepdims=True)
            for row in scores:
                order = np.argsort(-row)[:k]
                outputs.append(
                    {
                        "label": self.labels[int(order[0])],
                        "score": round(float(row[order[0]]), 6),
                        "top_k": [
                            {"label": self.labels[int(index)], "score": round(float(row[index]), 6)}
                            for index in order
                        ],
                    }
                )
        return outputs
