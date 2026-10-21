"""把压缩实验结果渲染成 Markdown 报告。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

TASK_TITLES = {
    "single": "单标签分类模型",
    "multi": "多标签分类模型",
}


def _num(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _mb(size_bytes: float) -> str:
    return f"{size_bytes / 1024 ** 2:.1f}"


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _find(variants: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for variant in variants:
        if variant["name"] == name:
            return variant
    return None


def _primary_metric(task_key: str) -> str:
    return "accuracy" if task_key == "single" else "micro_f1"


def _single_table(variants: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "| 变体 | 压缩方式 | 参数量(M) | 大小(MB) | 大小占比 | CPU 延迟(ms) | CPU 吞吐(条/秒) | GPU 延迟(ms) | GPU 吞吐(条/秒) | Acc | Macro-F1 | Top-3 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in variants:
        metrics = item["metrics"]
        lines.append(
            "| `{name}` | {desc} | {params} | {size} | {ratio} | {cpu} | {cthr} | {gpu} | {thr} | {acc} | {mf1} | {top3} |".format(
                name=item["name"],
                desc=item["description"],
                params=_num(item["params"] / 1e6, 1),
                size=_mb(item["size_bytes"]),
                ratio=_pct(item["size_ratio"]),
                cpu=_num(item.get("cpu_latency_ms"), 2),
                cthr=_num(item.get("cpu_throughput"), 1),
                gpu=_num(item.get("gpu_latency_ms"), 2),
                thr=_num(item.get("gpu_throughput"), 1),
                acc=_num(metrics.get("accuracy")),
                mf1=_num(metrics.get("macro_f1")),
                top3=_num(metrics.get("top3_accuracy")),
            )
        )
    return lines


def _multi_table(variants: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "| 变体 | 压缩方式 | 参数量(M) | 大小(MB) | 大小占比 | CPU 延迟(ms) | CPU 吞吐(条/秒) | GPU 延迟(ms) | GPU 吞吐(条/秒) | Micro-F1 | Macro-F1 | 子集准确率 | 汉明损失 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in variants:
        metrics = item["metrics"]
        lines.append(
            "| `{name}` | {desc} | {params} | {size} | {ratio} | {cpu} | {cthr} | {gpu} | {thr} | {micro} | {macro} | {subset} | {ham} |".format(
                name=item["name"],
                desc=item["description"],
                params=_num(item["params"] / 1e6, 1),
                size=_mb(item["size_bytes"]),
                ratio=_pct(item["size_ratio"]),
                cpu=_num(item.get("cpu_latency_ms"), 2),
                cthr=_num(item.get("cpu_throughput"), 1),
                gpu=_num(item.get("gpu_latency_ms"), 2),
                thr=_num(item.get("gpu_throughput"), 1),
                micro=_num(metrics.get("micro_f1")),
                macro=_num(metrics.get("macro_f1")),
                subset=_num(metrics.get("subset_accuracy")),
                ham=_num(metrics.get("hamming_loss")),
            )
        )
    return lines


def _task_findings(task_key: str, variants: List[Dict[str, Any]]) -> List[str]:
    baseline = variants[0]
    metric = _primary_metric(task_key)
    base_acc = baseline["metrics"][metric]
    findings: List[str] = []

    int8 = _find(variants, "int8_dynamic")
    if int8:
        cpu_speedup = (
            int8["cpu_throughput"] / baseline["cpu_throughput"]
            if int8.get("cpu_throughput") and baseline.get("cpu_throughput")
            else None
        )
        findings.append(
            "**INT8 动态量化**：体积从 {base_size} MB 降到 {size} MB（{ratio}），CPU 单条延迟 "
            "{base_cpu} ms → {cpu} ms，CPU 吞吐 {base_thr} → {thr} 条/秒{extra}，{metric} "
            "{base_acc} → {acc}。".format(
                base_size=_mb(baseline["size_bytes"]),
                size=_mb(int8["size_bytes"]),
                ratio=_pct(int8["size_ratio"]),
                base_cpu=_num(baseline["cpu_latency_ms"], 2),
                cpu=_num(int8.get("cpu_latency_ms"), 2),
                base_thr=_num(baseline.get("cpu_throughput"), 1),
                thr=_num(int8.get("cpu_throughput"), 1),
                extra=f"（吞吐 {cpu_speedup:.2f}×）" if cpu_speedup else "",
                metric="准确率" if task_key == "single" else "Micro-F1",
                base_acc=_num(base_acc),
                acc=_num(int8["metrics"][metric]),
            )
        )

    for dtype_name in ("fp16", "bf16"):
        half = _find(variants, dtype_name)
        if not half:
            continue
        speedup = (
            baseline["gpu_latency_ms"] / half["gpu_latency_ms"]
            if half.get("gpu_latency_ms") and baseline.get("gpu_latency_ms")
            else None
        )
        findings.append(
            "**{name} 半精度**：体积 {size} MB（{ratio}），GPU 单条延迟 {base_gpu} ms → {gpu} ms{extra}，"
            "{metric} {acc}。".format(
                name=dtype_name.upper(),
                size=_mb(half["size_bytes"]),
                ratio=_pct(half["size_ratio"]),
                base_gpu=_num(baseline.get("gpu_latency_ms"), 2),
                gpu=_num(half.get("gpu_latency_ms"), 2),
                extra=f"（加速 {speedup:.2f}×）" if speedup else "",
                metric="准确率" if task_key == "single" else "Micro-F1",
                acc=_num(half["metrics"][metric]),
            )
        )

    unstructured = [item for item in variants if item["method"] == "unstructured"]
    for item in unstructured:
        findings.append(
            "**非结构化剪枝 {sparsity}**：实际稀疏率 {sparsity_real}，体积不变（{size} MB，仍为稠密存储），"
            "{metric} {acc}{delta}。".format(
                sparsity=item["description"].split()[-1],
                sparsity_real=_pct(item.get("sparsity", 0.0), 2),
                size=_mb(item["size_bytes"]),
                metric="准确率" if task_key == "single" else "Micro-F1",
                acc=_num(item["metrics"][metric]),
                delta=_delta_text(item["metrics"][metric] - base_acc),
            )
        )

    structured = [item for item in variants if item["method"] == "structured"]
    for item in structured:
        findings.append(
            "**结构化剪枝 {desc}**：参数量 {params} M、体积 {size} MB（{ratio}），GPU 单条延迟 {gpu} ms，"
            "{metric} {acc}。".format(
                desc=item["description"].split("：")[-1],
                params=_num(item["params"] / 1e6, 1),
                size=_mb(item["size_bytes"]),
                ratio=_pct(item["size_ratio"]),
                gpu=_num(item.get("gpu_latency_ms"), 2),
                metric="准确率" if task_key == "single" else "Micro-F1",
                acc=_num(item["metrics"][metric]),
            )
        )

    low_rank = [item for item in variants if item["method"] == "low_rank"]
    for item in low_rank:
        findings.append(
            "**低秩分解 {desc}**：参数量 {params} M、体积 {size} MB（{ratio}），GPU 单条延迟 {gpu} ms，"
            "{metric} {acc}。".format(
                desc=item["description"].split("：")[-1],
                params=_num(item["params"] / 1e6, 1),
                size=_mb(item["size_bytes"]),
                ratio=_pct(item["size_ratio"]),
                gpu=_num(item.get("gpu_latency_ms"), 2),
                metric="准确率" if task_key == "single" else "Micro-F1",
                acc=_num(item["metrics"][metric]),
            )
        )
    return findings


def _delta_text(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"（{sign}{delta * 100:.2f} 个百分点）"


def _delta_range(values: List[float]) -> str:
    """把若干任务的指标变化渲染成可读区间。"""
    if not values:
        return "—"
    low, high = min(values), max(values)
    if abs(high - low) < 1e-9:
        return f"{low * 100:+.2f} 个百分点"
    return f"{low * 100:+.2f} ~ {high * 100:+.2f} 个百分点"


def _collect(
    tasks: Dict[str, Dict[str, Any]],
    name: str,
) -> List[tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """收集各任务中指定变体及其基线，返回 (任务, 变体, 基线) 列表。"""
    collected = []
    for task_key, task_result in tasks.items():
        variants = task_result["variants"]
        item = _find(variants, name)
        baseline = _find(variants, "fp32")
        if item and baseline:
            collected.append((task_key, item, baseline))
    return collected


def _method_conclusions(tasks: Dict[str, Dict[str, Any]]) -> List[str]:
    """依据实测结果动态生成结论，避免把与数据无关的定性判断写死。"""
    lines: List[str] = []
    metric = _primary_metric

    for dtype_name in ("fp16", "bf16"):
        entries = _collect(tasks, dtype_name)
        if not entries:
            continue
        ratios = [item["size_ratio"] for _, item, _ in entries]
        deltas = [item["metrics"][metric(task)] - base["metrics"][metric(task)] for task, item, base in entries]
        speedups = [
            base["gpu_latency_ms"] / item["gpu_latency_ms"]
            for _, item, base in entries
            if item.get("gpu_latency_ms") and base.get("gpu_latency_ms")
        ]
        speed_text = f"，GPU 单条延迟加速 {min(speedups):.2f}× ~ {max(speedups):.2f}×" if speedups else ""
        lines.append(
            "**{name} 半精度**：体积为基线的 {ratio}，测试指标相对基线变化 {delta}{speed}。".format(
                name=dtype_name.upper(),
                ratio=f"{_pct(min(ratios))} ~ {_pct(max(ratios))}" if abs(max(ratios) - min(ratios)) > 1e-9 else _pct(ratios[0]),
                delta=_delta_range(deltas),
                speed=speed_text,
            )
        )

    entries = _collect(tasks, "int8_dynamic")
    if entries:
        ratios = [item["size_ratio"] for _, item, _ in entries]
        deltas = [item["metrics"][metric(task)] - base["metrics"][metric(task)] for task, item, base in entries]
        speedups = [
            item["cpu_throughput"] / base["cpu_throughput"]
            for _, item, base in entries
            if item.get("cpu_throughput") and base.get("cpu_throughput")
        ]
        speed_text = (
            f"，CPU 批量吞吐为基线的 {min(speedups):.2f}× ~ {max(speedups):.2f}×" if speedups else ""
        )
        lines.append(
            "**动态 INT8 量化**：体积为基线的 {ratio}，测试指标变化 {delta}{speed}，主要面向 CPU 部署。".format(
                ratio=f"{_pct(min(ratios))} ~ {_pct(max(ratios))}" if abs(max(ratios) - min(ratios)) > 1e-9 else _pct(ratios[0]),
                delta=_delta_range(deltas),
                speed=speed_text,
            )
        )

    for method, label in (("unstructured", "非结构化剪枝"), ("structured", "结构化剪枝"), ("low_rank", "低秩分解")):
        entries = []
        for task_key, task_result in tasks.items():
            baseline = _find(task_result["variants"], "fp32")
            for item in task_result["variants"]:
                if item["method"] == method and baseline:
                    entries.append((task_key, item, baseline))
        if not entries:
            continue
        ratios = [item["size_ratio"] for _, item, _ in entries]
        deltas = [item["metrics"][metric(task)] - base["metrics"][metric(task)] for task, item, base in entries]
        detail = ""
        if method == "unstructured":
            sparsities = [item.get("sparsity", 0.0) for _, item, _ in entries]
            detail = f"，实际稀疏率 {_pct(min(sparsities), 2)} ~ {_pct(max(sparsities), 2)}，稠密存储下体积不下降"
        lines.append(
            "**{label}**：体积为基线的 {ratio}{detail}，测试指标相对基线变化 {delta}。".format(
                label=label,
                ratio=f"{_pct(min(ratios))} ~ {_pct(max(ratios))}" if abs(max(ratios) - min(ratios)) > 1e-9 else _pct(ratios[0]),
                detail=detail,
                delta=_delta_range(deltas),
            )
        )

    if lines:
        lines = [f"- {line}" for line in lines]
        lines.append("")
        lines.append(
            "> 以上均为**训练后一次性压缩、未做任务微调**的结果；精度变化为相对 FP32 基线的实测差值。"
        )
    return lines


def _baseline_line(task_key: str, variants: List[Dict[str, Any]]) -> str:
    baseline = variants[0]
    metrics = baseline["metrics"]
    if task_key == "single":
        return (
            "基线（FP32）测试指标：准确率 {acc}、Macro-F1 {mf1}、Top-3 准确率 {top3}。"
        ).format(acc=_num(metrics.get("accuracy")), mf1=_num(metrics.get("macro_f1")), top3=_num(metrics.get("top3_accuracy")))
    return (
        "基线（FP32）测试指标：Micro-F1 {micro}、Macro-F1 {macro}、子集准确率 {subset}、汉明损失 {ham}。"
    ).format(
        micro=_num(metrics.get("micro_f1")),
        macro=_num(metrics.get("macro_f1")),
        subset=_num(metrics.get("subset_accuracy")),
        ham=_num(metrics.get("hamming_loss")),
    )


def build_markdown(results: Dict[str, Any]) -> str:
    """根据结果字典生成完整报告。"""
    environment = results["environment"]
    project = results.get("project", {})
    tasks = results["tasks"]

    model_name = project.get("model_name") or "BERT-base"
    dataset_name = project.get("dataset_name") or "IMDb"
    output_root_name = project.get("output_root_name") or "model"
    config_file = project.get("config_file") or "config.yaml"

    title = f"{dataset_name}类型分类模型压缩实验报告"

    lines: List[str] = [
        f"# {title}",
        "",
        f"- 生成时间：{results['generated_at']}",
        f"- 基础模型：{model_name}（BERT-base，12 层 / 768 隐藏维度 / 12 注意力头）",
        f"- 数据集：{dataset_name}（`{project.get('dataset_path', '')}`）",
        f"- 训练/压缩产物目录：`{project.get('output_root', '')}`",
        f"- 硬件：{environment['gpu']}；CPU {environment['cpu']}（{environment['cpu_threads']} 线程）",
        f"- 软件：Python {environment['python']}，PyTorch {environment['torch']}，Transformers {environment['transformers']}",
        "",
        "## 实验设置",
        "",
        "两个任务分别独立压缩，所有变体都在**与训练时一致的测试集**上评估，"
        "指标口径与训练阶段完全相同。",
        "",
    ]

    single = tasks.get("single")
    if single:
        lines.append(
            f"- 单标签：测试集 {single['test_size']:,} 条，指标为准确率、Macro-F1、Top-3 准确率；"
        )
    multi = tasks.get("multi")
    if multi:
        if multi.get("threshold_mode") == "per_class":
            threshold_text = "使用验证集逐类校准的阈值"
        else:
            threshold_text = f"固定阈值 {multi.get('threshold', 0.5)}"
        lines.append(
            f"- 多标签：测试集 {multi['test_size']:,} 条，{threshold_text}，"
            "指标为 Micro-F1、Macro-F1、子集准确率、汉明损失；"
        )

    lines.extend(
        [
            "- 延迟为纯前向耗时（不含分词），单条延迟取 batch=1；吞吐取 batch="
            f"{environment['latency_batch_size']}，CPU 与 GPU 分别测量；CPU 为 "
            f"{environment['cpu_threads']} 线程；",
            "- 体积为落盘目录总大小（含分词器与 labels.json），基线目录为训练产出的 `best/`。",
            "",
            "### 压缩方法",
            "",
            "| 方法 | 说明 |",
            "| --- | --- |",
            "| FP16 / BF16 | 权重与计算转为半精度，GPU Tensor Core 加速，体积约为 FP32 的一半 |",
            "| 动态 INT8 | 对全部 Linear 层权重做 int8 量化，激活在推理时动态量化，主要面向 CPU 部署 |",
            "| 非结构化剪枝 | 按全局权重绝对值分位数置零，精度损失较小；稠密存储下体积不变 |",
            "| 结构化剪枝 | 按 L2 范数裁剪 FFN 中间层神经元并同步裁剪输出投影，参数量与体积真实下降 |",
            "| 低秩分解 | 对 FFN 输入/输出投影做 SVD 截断，用两个小矩阵相乘近似原权重 |",
            "",
            "> 说明：剪枝与低秩分解均为**训练后一次性压缩，未做微调**，精度变化反映的是直接压缩的代价。",
            "",
        ]
    )

    for task_key, task_result in tasks.items():
        variants = task_result["variants"]
        lines.extend(
            [
                f"## {TASK_TITLES.get(task_key, task_key)}（`{task_key}`）",
                "",
                f"- 基线模型目录：`{task_result['baseline_dir']}`",
                f"- 类别数：{task_result['label_count']}，测试样本：{task_result['test_size']}",
                f"- {_baseline_line(task_key, variants)}",
                "",
            ]
        )
        lines.extend(_single_table(variants) if task_key == "single" else _multi_table(variants))
        lines.extend(["", "**结果解读**", ""])
        lines.extend(f"- {item}" for item in _task_findings(task_key, variants))
        lines.append("")

    lines.extend(
        [
            "## 结论",
            "",
            *_method_conclusions(tasks),
            "",
            "## 产物与复现",
            "",
            "所有压缩模型保存在各任务目录下：",
            "",
            "```",
            f"{output_root_name}/",
            "├── single_label/",
            "│   ├── best/                 # 原始模型",
            "│   ├── test_predictions.csv  # best 模型的逐条预测",
            "│   ├── test_probabilities.npy",
            "│   └── optimized/<variant>/  # 各压缩变体（同样含逐条预测）",
            "└── multi_label/",
            "    ├── best/",
            "    └── optimized/<variant>/",
            "```",
            "",
            "每个变体目录包含 `labels.json`、`optimization.json`（重建所需元信息）、模型权重，"
            "以及 `test_predictions.csv` / `test_probabilities.npy` 两份逐条预测结果；"
            "`fp16`/`bf16`/剪枝模型可直接用 transformers 加载，INT8 与低秩模型需按 "
            "`optimization.json` 重建结构后加载权重（见 `src/optimization.py` 的 `load_variant`）。",
            "",
            "```cmd",
            f"python run_optimization.py --config {config_file} --task all",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
