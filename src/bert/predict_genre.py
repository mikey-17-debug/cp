"""独立预测脚本：只改 MODEL_DIR 即可，不依赖项目内其它模块。

加载方式与项目里的 api_predict 保持一致：模块导入时直接加载模型，
对外只暴露一个 predict(data: dict) 方法，data 只需要包含 text 字段。

用法一（代码调用）：
    from predict_genre import predict
    result = predict({"text": "一名银行家蒙冤入狱，用二十年时间挖通墙壁越狱。"})
    # result = {"text": ..., "pred_class": "犯罪", "score": ..., "top_k": [...], "duration": 12}

用法二（命令行）：
    python predict_genre.py --text "剧情简介"
    python predict_genre.py --text "文本1" "文本2"
    python predict_genre.py --file texts.txt --json

模型目录需包含 config.json、权重、分词器与 labels.json；
脚本会根据 labels.json 里的 problem_type 自动区分单标签 / 多标签。

默认自动选择设备；强制 CPU 两种方式：
    1) 环境变量：set GENRE_DEVICE=cpu
    2) 改脚本顶部：DEVICE_PREFERENCE = "cpu"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import LLMConfig, BertConfig

# ======== 只需要改这一行：指向包含 config.json / labels.json / 权重的模型目录 ========
config_obj = BertConfig()
MODEL_DIR = config_obj.model_dir
# ==================================================================================

# 运行设备："auto"（有 GPU 用 GPU）/ "cpu"（强制 CPU）/ "cuda:0" 等
# 也可以不改文件，通过环境变量覆盖：set GENRE_DEVICE=cpu
DEVICE_PREFERENCE = os.environ.get("GENRE_DEVICE", "auto").strip().lower()

MODEL_PATH = Path(MODEL_DIR)
LABELS_FILE = MODEL_PATH / "labels.json"
if not LABELS_FILE.exists():
    raise FileNotFoundError(f"模型目录缺少 labels.json：{LABELS_FILE}")

META: Dict[str, Any] = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
LABELS: List[str] = list(META["labels"])
MAX_LENGTH: int = int(META.get("max_length", 512))
IS_MULTI: bool = META.get("problem_type") == "multi_label_classification"
TOP_K: int = int(META.get("top_k", 3))
DEFAULT_THRESHOLD: float = float(META.get("threshold", 0.5))
THRESHOLDS: Dict[str, float] = dict(META.get("thresholds") or {})

DEVICE = torch.device(
    ("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE_PREFERENCE == "auto"
    else DEVICE_PREFERENCE
)
if DEVICE.type == "cuda" and not torch.cuda.is_available():
    DEVICE = torch.device("cpu")

TOKENIZER = AutoTokenizer.from_pretrained(str(MODEL_PATH))
MODEL = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH)).eval().to(DEVICE)


def predict(data: Dict[str, Any]) -> Dict[str, Any]:
    """接收包含 text 字段的字典，返回补充了预测结果的同一字典。

    Args:
        data: {"text": "剧情简介"}，可选 "top_k" / "threshold" 覆盖默认值。

    Returns:
        data 原字典，新增：
            pred_class  单标签为类型名，多标签为概率最高的类型
            score       对应概率
            top_k       单标签的 Top-K 列表
            pred_labels / scores  多标签命中的全部类型与概率
            duration    本次推理耗时（毫秒）
    """
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("data 必须包含非空的 text 字段")

    started = time.time()
    encoded = TOKENIZER(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = MODEL(**encoded).logits[0].float().cpu().numpy()

    if IS_MULTI:
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        threshold = float(data.get("threshold", DEFAULT_THRESHOLD))
        if THRESHOLDS:
            cut = np.asarray([THRESHOLDS.get(name, threshold) for name in LABELS])
            selected = [int(i) for i in np.argsort(-probabilities) if probabilities[i] >= cut[i]]
        else:
            selected = [int(i) for i in np.argsort(-probabilities) if probabilities[i] >= threshold]
        best = int(selected[0]) if selected else int(np.argmax(probabilities))
        data["pred_class"] = LABELS[best]
        data["score"] = round(float(probabilities[best]), 6)
        data["pred_labels"] = [LABELS[i] for i in selected]
        data["scores"] = [round(float(probabilities[i]), 6) for i in selected]
    else:
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        k = int(data.get("top_k", TOP_K))
        order = np.argsort(-probabilities)[:k]
        data["pred_class"] = LABELS[int(order[0])]
        data["score"] = round(float(probabilities[order[0]]), 6)
        data["top_k"] = [
            {"label": LABELS[int(i)], "score": round(float(probabilities[i]), 6)}
            for i in order
        ]

    data["duration"] = int((time.time() - started) * 1000)
    return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="电影类型分类（独立脚本）")
    parser.add_argument("--text", nargs="+", help="一条或多条剧情简介")
    parser.add_argument("--file", help="从文本文件读取，每行一条")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.text:
        texts = list(args.text)
    elif args.file:
        texts = [line.strip() for line in Path(args.file).read_text(encoding="utf-8").splitlines() if line.strip()]
    elif not sys.stdin.isatty():
        texts = [line.strip() for line in sys.stdin.read().splitlines() if line.strip()]
    else:
        raise SystemExit("请通过 --text 或 --file 提供待预测文本")

    results = [predict({"text": text}) for text in texts]
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    for result in results:
        text = result["text"]
        print(f"\n文本：{text[:60]}{'...' if len(text) > 60 else ''}")
        if IS_MULTI:
            pairs = "、".join(
                f"{label}({score:.4f})" for label, score in zip(result["pred_labels"], result["scores"])
            )
            print(f"  预测类型：{pairs or '<无>'} | 耗时 {result['duration']} ms")
        else:
            print(f"  预测类型：{result['pred_class']}（{result['score']:.4f}） | 耗时 {result['duration']} ms")
            for item in result["top_k"]:
                print(f"    - {item['label']:<6} {item['score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
