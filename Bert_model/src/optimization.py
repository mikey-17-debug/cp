"""模型压缩：多种精度量化、剪枝、低秩分解。

每个变体都遵循同一套约定：
    * `make_variants` 依据配置生成变体清单；
    * `apply_variant` 从基线模型构造变体模型（不修改传入的模型）；
    * `save_variant` 把变体落盘，并在 optimization.json 记录重建所需的元信息；
    * `load_variant` 读取落盘产物，便于验证与复用。

保存格式：
    * pretrained：可直接被 transformers 加载（fp32/fp16/bf16/剪枝模型）
    * state_dict：动态 INT8 与低秩模型需要先按元信息重建结构，再加载权重
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerBase

from .config import ProjectConfig

logger = logging.getLogger(__name__)

META_FILE = "optimization.json"

# torch.quantile 的输入元素上限较低，用随机子集估计分位数阈值
QUANTILE_SAMPLE_LIMIT = 4_000_000


@dataclass(frozen=True)
class VariantSpec:
    """一个压缩变体的定义。"""

    name: str
    method: str  # baseline / dtype / int8 / unstructured / structured / low_rank
    description: str
    device: str  # 指标测量所用设备：cuda / cpu
    param: Optional[float] = None  # dtype 变体不用，其余表示压缩比例
    save_format: str = "pretrained"  # pretrained / state_dict


class LowRankLinear(nn.Module):
    """把 Linear 权重做 SVD 截断后表示为两个小矩阵的乘积。"""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(inputs))

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int) -> "LowRankLinear":
        module = cls(linear.in_features, linear.out_features, rank, bias=linear.bias is not None)
        weight = linear.weight.detach().float().cpu()
        # W ≈ U_r S_r V_r^T：down 存 V_r^T，up 存 U_r S_r
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        module.down.weight.data = vh[:rank].clone()
        module.up.weight.data = (u[:, :rank] * s[:rank]).clone()
        if linear.bias is not None:
            module.up.bias.data = linear.bias.detach().clone()
        return module

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}"


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def make_variants(config: ProjectConfig) -> List[VariantSpec]:
    """按配置生成所有待评测的压缩变体。"""
    optimization = config.optimization
    variants: List[VariantSpec] = [
        VariantSpec("fp32", "baseline", "原始 FP32 模型", "cuda"),
    ]
    for dtype in optimization.dtypes:
        variants.append(
            VariantSpec(dtype, "dtype", f"半精度 {dtype.upper()} 推理", "cuda")
        )

    if optimization.int8_dynamic:
        variants.append(
            VariantSpec(
                "int8_dynamic",
                "int8",
                "动态 INT8 量化（Linear 层）",
                "cpu",
                save_format="state_dict",
            )
        )

    for ratio in optimization.unstructured_ratios:
        variants.append(
            VariantSpec(
                f"prune_unstructured_{int(round(ratio * 100))}",
                "unstructured",
                f"非结构化幅度剪枝 {int(round(ratio * 100))}%",
                "cuda",
                param=ratio,
            )
        )

    for keep in optimization.structured_keep_ratios:
        variants.append(
            VariantSpec(
                f"prune_structured_{int(round((1 - keep) * 100))}",
                "structured",
                f"结构化剪枝：FFN 中间层保留 {int(round(keep * 100))}%",
                "cuda",
                param=keep,
            )
        )

    for ratio in optimization.low_rank_ratios:
        variants.append(
            VariantSpec(
                f"low_rank_{int(round(ratio * 100))}",
                "low_rank",
                f"FFN 低秩分解：秩比例为 min(in, out) 的 {int(round(ratio * 100))}%",
                "cuda",
                param=ratio,
                save_format="state_dict",
            )
        )
    return variants


def apply_variant(model: nn.Module, spec: VariantSpec, config: ProjectConfig) -> nn.Module:
    """从基线模型构造变体模型（深拷贝，不修改传入模型）。"""
    if spec.method == "baseline":
        return copy.deepcopy(model)

    if spec.method == "dtype":
        target = copy.deepcopy(model)
        return target.to(dtype=dtype_from_name(spec.name))

    if spec.method == "int8":
        target = copy.deepcopy(model).cpu().eval()
        return torch.ao.quantization.quantize_dynamic(target, {nn.Linear}, dtype=torch.qint8)

    target = copy.deepcopy(model)
    if spec.method == "unstructured":
        prune_unstructured(target, float(spec.param))
    elif spec.method == "structured":
        prune_ffn_neurons(target, float(spec.param))
    elif spec.method == "low_rank":
        apply_low_rank_ffn(target, float(spec.param))
    else:
        raise ValueError(f"未知压缩方法：{spec.method}")
    return target


def prune_unstructured(model: nn.Module, ratio: float) -> float:
    """全局幅度剪枝：按所有权重绝对值分位数置零，返回实际稀疏率。

    torch.quantile 对超大张量有限制，这里用固定种子的随机子集估计分位数阈值，
    实际稀疏率仍按全部权重精确统计。
    """
    targets = [
        module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not name.startswith("classifier")
    ]
    if not targets:
        raise ValueError("未找到可剪枝的 Linear 层")

    magnitudes = torch.cat([module.weight.detach().abs().flatten() for module in targets])
    if magnitudes.numel() > QUANTILE_SAMPLE_LIMIT:
        generator = torch.Generator().manual_seed(0)
        sample_index = torch.randint(
            0, magnitudes.numel(), (QUANTILE_SAMPLE_LIMIT,), generator=generator
        )
        magnitudes = magnitudes[sample_index]
    threshold = torch.quantile(magnitudes, ratio)

    for module in targets:
        module.weight.data.mul_((module.weight.data.abs() > threshold).to(module.weight.dtype))

    zeros = sum(int((module.weight.data == 0).sum()) for module in targets)
    total = sum(module.weight.numel() for module in targets)
    sparsity = zeros / total
    logger.info("非结构化剪枝：目标 %.0f%%，实际稀疏率 %.2f%%", ratio * 100, sparsity * 100)
    return sparsity


def prune_ffn_neurons(model: nn.Module, keep_ratio: float) -> int:
    """结构化剪枝：按 L2 范数保留 FFN 中间层神经元，同时裁剪输出投影列。

    张量被真正裁掉，参数量与文件体积都会下降。
    """
    kept = 0
    for layer in model.bert.encoder.layer:
        dense = layer.intermediate.dense
        output = layer.output.dense
        total = dense.out_features
        kept = max(1, int(total * keep_ratio))
        norms = dense.weight.detach().norm(dim=1)
        index = torch.topk(norms, kept).indices.sort().values

        dense.weight = nn.Parameter(dense.weight.data[index].clone())
        if dense.bias is not None:
            dense.bias = nn.Parameter(dense.bias.data[index].clone())
        dense.out_features = kept

        output.weight = nn.Parameter(output.weight.data[:, index].clone())
        output.in_features = kept

    model.config.intermediate_size = kept
    logger.info("结构化剪枝：FFN 中间层 %d -> %d", total, kept)
    return kept


def apply_low_rank_ffn(model: nn.Module, rank_ratio: float, init_from_svd: bool = True) -> List[int]:
    """对 FFN 的输入投影与输出投影做同秩低秩分解。

    Args:
        init_from_svd: True 时用 SVD 初始化权重；加载已保存产物时置 False，避免重复分解。
    """
    ranks: List[int] = []
    for layer in model.bert.encoder.layer:
        dense = layer.intermediate.dense
        output = layer.output.dense
        rank = max(1, int(min(dense.in_features, dense.out_features) * rank_ratio))
        if init_from_svd:
            layer.intermediate.dense = LowRankLinear.from_linear(dense, rank)
            layer.output.dense = LowRankLinear.from_linear(output, rank)
        else:
            layer.intermediate.dense = LowRankLinear(dense.in_features, dense.out_features, rank)
            layer.output.dense = LowRankLinear(output.in_features, output.out_features, rank)
        ranks.append(rank)
    if init_from_svd:
        logger.info("低秩分解：FFN 秩比例 %.0f%%，秩 = %d", rank_ratio * 100, ranks[0])
    return ranks


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def linear_weight_sparsity(model: nn.Module) -> float:
    """Linear 权重中零元素占比（非结构化剪枝的有效稀疏率）。"""
    total = 0
    zeros = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not name.startswith("classifier"):
            total += module.weight.numel()
            zeros += int((module.weight.data == 0).sum())
    return zeros / total if total else 0.0


def save_variant(
    model: nn.Module,
    spec: VariantSpec,
    target_dir: Path,
    base_model_dir: Path,
    tokenizer: PreTrainedTokenizerBase,
    labels_payload: Dict[str, Any],
) -> Path:
    """保存变体模型、分词器与元信息。"""
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "variant": spec.name,
        "method": spec.method,
        "description": spec.description,
        "param": spec.param,
        "save_format": spec.save_format,
        "device": spec.device,
        "task": labels_payload.get("task"),
        "problem_type": labels_payload.get("problem_type"),
        "labels": labels_payload.get("labels"),
        "max_length": labels_payload.get("max_length"),
        "threshold": labels_payload.get("threshold"),
        "top_k": labels_payload.get("top_k"),
        "base_model_dir": str(base_model_dir),
        "config": asdict(spec),
    }

    tokenizer.save_pretrained(str(target_dir))
    (target_dir / "labels.json").write_text(
        json.dumps(labels_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if spec.save_format == "pretrained":
        model.save_pretrained(str(target_dir))
    else:
        torch.save(model.state_dict(), str(target_dir / "pytorch_model.pt"))

    (target_dir / META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target_dir


def load_variant(target_dir: Path) -> nn.Module:
    """按元信息重建并加载变体模型（用于验证产物可用）。"""
    meta = json.loads((target_dir / META_FILE).read_text(encoding="utf-8"))
    method = meta["method"]
    base_dir = meta["base_model_dir"]

    if method in {"baseline", "unstructured", "structured"}:
        return AutoModelForSequenceClassification.from_pretrained(str(target_dir))

    if method == "dtype":
        return AutoModelForSequenceClassification.from_pretrained(
            str(target_dir), dtype=dtype_from_name(meta["variant"])
        )

    if method == "int8":
        base = AutoModelForSequenceClassification.from_pretrained(base_dir).cpu().eval()
        model = torch.ao.quantization.quantize_dynamic(base, {nn.Linear}, dtype=torch.qint8)
        state = torch.load(target_dir / "pytorch_model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model

    if method == "low_rank":
        base = AutoModelForSequenceClassification.from_pretrained(base_dir)
        apply_low_rank_ffn(base, float(meta["param"]), init_from_svd=False)
        state = torch.load(target_dir / "pytorch_model.pt", map_location="cpu", weights_only=True)
        base.load_state_dict(state)
        return base

    raise ValueError(f"未知压缩方法：{method}")
