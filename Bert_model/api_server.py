"""电影类型分类模型的 HTTP API（Flask）。

启动（在 Bert_model 目录下）：

    python api_server.py --config config_zh_balance.yaml --task single --port 8000

接口：
    GET  /health   查看模型、标签与设备状态
    POST /predict  传入一条或多条剧情简介，返回预测类型与 Top-K

请求示例：
    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d "{\"texts\": [\"一名银行家蒙冤入狱，用二十年时间挖通墙壁越狱。\"], \"top_k\": 3}"

Python 调用示例：
    import requests
    resp = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"texts": ["一名银行家蒙冤入狱，用二十年时间挖通墙壁越狱。"], "top_k": 3},
        timeout=30,
    )
    print(resp.json())
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import TASK_KEYS, load_config  # noqa: E402
from src.predictor import GenreClassifier  # noqa: E402
from src.utils import setup_logging  # noqa: E402

logger = logging.getLogger("api")


def build_app(classifier: GenreClassifier, max_batch: int, api_key: str | None = None) -> Flask:
    """把分类器包装成 Flask 应用。

    Args:
        api_key: 非空时，/predict 需要携带 X-API-Key 或 Authorization: Bearer <key>。
    """
    app = Flask(__name__)
    lock = threading.Lock()

    @app.after_request
    def add_cors_headers(response):
        # 允许浏览器跨域调用（如需限制来源，可把 * 改成具体域名）
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.route("/", methods=["GET"])
    def index() -> Any:
        return jsonify(
            {
                "service": "movie-genre-classifier",
                "endpoints": {
                    "GET /health": "查看模型与标签状态",
                    "POST /predict": "预测类型，body: {\"texts\": [...], \"top_k\": 3}",
                },
                "auth": "请求头 X-API-Key 或 Authorization: Bearer <key>" if api_key else "未启用",
            }
        )

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        return jsonify(
            {
                "status": "ok",
                "model_dir": str(classifier.model_dir),
                "problem_type": classifier.problem_type,
                "labels": classifier.labels,
                "device": str(classifier.device),
                "max_length": classifier.max_length,
                "auth_required": bool(api_key),
            }
        )

    @app.route("/predict", methods=["POST"])
    def predict() -> Any:
        if api_key:
            provided = request.headers.get("X-API-Key", "").strip()
            authorization = request.headers.get("Authorization", "")
            if authorization.lower().startswith("bearer "):
                provided = authorization[7:].strip() or provided
            if provided != api_key:
                return jsonify({"error": "未授权，请在 X-API-Key 或 Authorization 中提供有效 key"}), 401

        payload: Dict[str, Any] = request.get_json(silent=True) or {}
        texts = payload.get("texts")
        if texts is None and isinstance(payload.get("text"), str):
            texts = [payload["text"]]
        if not isinstance(texts, list) or not texts:
            return jsonify({"error": "请提供 texts（字符串数组）或 text（字符串）"}), 400
        if len(texts) > max_batch:
            return jsonify({"error": f"单次最多支持 {max_batch} 条文本"}), 400
        if not all(isinstance(text, str) and text.strip() for text in texts):
            return jsonify({"error": "texts 中存在空文本或非字符串"}), 400

        top_k = payload.get("top_k")
        if top_k is not None:
            if not isinstance(top_k, int) or not 1 <= top_k <= len(classifier.labels):
                return jsonify({"error": f"top_k 必须是 1~{len(classifier.labels)} 的整数"}), 400

        threshold = payload.get("threshold")
        if threshold is not None and not isinstance(threshold, (int, float)):
            return jsonify({"error": "threshold 必须是数字"}), 400

        started = time.perf_counter()
        # 单卡推理串行执行，避免并发请求争抢显存
        with lock:
            results = classifier.predict(texts, top_k=top_k, threshold=threshold)
        elapsed_ms = (time.perf_counter() - started) * 1000

        return jsonify(
            {
                "model_dir": str(classifier.model_dir),
                "count": len(texts),
                "elapsed_ms": round(elapsed_ms, 2),
                "results": results,
            }
        )

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="电影类型分类 HTTP API")
    parser.add_argument("--config", default="config_zh_balance.yaml", help="配置文件路径")
    parser.add_argument("--task", choices=list(TASK_KEYS), default="single", help="使用哪个任务的模型")
    parser.add_argument("--model-dir", help="直接指定模型目录（优先于 --task）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，局域网访问用 0.0.0.0")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--max-batch", type=int, default=64, help="单次请求的最大文本条数")
    parser.add_argument("--device", help="推理设备，如 cuda:0 / cpu；默认自动选择")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GENRE_API_KEY") or None,
        help="访问 /predict 所需的 key（也可用环境变量 GENRE_API_KEY）；不填则不校验",
    )
    parser.add_argument("--log-file", help="日志文件路径")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(log_file=args.log_file)
    config = load_config(args.config)

    if args.model_dir:
        model_dir = Path(args.model_dir).expanduser().resolve()
    else:
        model_dir = config.best_model_dir(args.task)

    classifier = GenreClassifier(
        model_dir=model_dir,
        device=args.device,
        max_length=config.model.max_length,
    )
    app = build_app(classifier, max_batch=args.max_batch, api_key=args.api_key)
    logger.info(
        "API 已启动：http://%s:%d（模型 %s，鉴权 %s）",
        args.host,
        args.port,
        model_dir,
        "已启用" if args.api_key else "未启用",
    )
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
