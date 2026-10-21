"""配置层：把 config.yaml 解析为强类型对象。

约定：
    * 代码中不出现任何硬编码的路径或超参数，一律从配置读取；
    * 相对路径统一以项目根目录（config.yaml 所在目录）为基准解析；
    * 命令行参数以「覆盖配置」的方式生效，而不是绕过配置。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

# 项目根目录 = 本文件所在包目录的上一级
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config.yaml"

# 两类任务的标识
SINGLE_LABEL = "single"
MULTI_LABEL = "multi"
TASK_KEYS = (SINGLE_LABEL, MULTI_LABEL)

# 任务标识 -> transformers 的 problem_type，决定损失函数（CE / BCE）
PROBLEM_TYPES = {
    SINGLE_LABEL: "single_label_classification",
    MULTI_LABEL: "multi_label_classification",
}


def resolve_path(value: str | Path, base: Path, field_name: str = "") -> Path:
    """相对路径按 base 解析为绝对路径，绝对路径原样返回。"""
    path = Path(str(value)).expanduser()
    resolved = path if path.is_absolute() else (base / path).resolve()
    return resolved


@dataclass
class DataConfig:
    """数据读取与标签过滤配置。"""

    # 数据源类型：text（分隔符文本文件）/ arrow（HuggingFace datasets 落盘目录）
    source: str = "text"
    path: str = "dataset/clean_data.txt"
    # 报告 / 文档中展示的数据集名称（为空时回退到数据目录名）
    display_name: str = ""
    raw_path: str = "../original_file/movie_info.txt"
    encoding: str = "utf-8"
    has_header: bool = True
    columns: List[str] = field(default_factory=lambda: ["id", "label", "name", "summary"])
    label_column: str = "label"
    text_column: str = "summary"
    # arrow 数据源：读取哪个 split（null 表示唯一 split 或优先 train）
    arrow_split: Optional[str] = None
    # arrow 数据源：把多个字段拼成模型输入的模板，例如 "{title}. {storyline}"
    text_template: str = ""
    # 模板渲染结果为空时，按顺序回退到这些字段（取第一个非空值）
    fallback_text_fields: List[str] = field(default_factory=list)
    # 需要剔除的标签（例如 IMDb 的 Short/Documentary 属于载体格式而非题材）
    exclude_labels: List[str] = field(default_factory=list)
    # 单标签构造方式：expand（一条多类型展开成多条）/ primary（只取主类型）
    single_label_strategy: str = "expand"
    min_label_count: int = 5
    drop_duplicated: bool = True
    missing_text: str = ""
    limit: Optional[int] = None
    # 是否把实际参与训练 / 验证 / 测试的样本写回磁盘
    export_splits: bool = True
    # 切分结果输出目录（相对项目根目录），每个任务一个子目录
    export_dir: str = "dataset/splits"
    # 导出文件的编码；utf-8-sig 便于 Excel 直接打开中文
    export_encoding: str = "utf-8-sig"


@dataclass
class SplitConfig:
    """训练 / 验证 / 测试集切分配置。"""

    test_size: float = 0.1
    val_size: float = 0.1
    seed: int = 42
    stratify: bool = True


@dataclass
class ModelConfig:
    """预训练模型配置。"""

    pretrained_path: str = "../../../NLP_learn10_12/PretrainedModel/chinese-bert-wwm"
    max_length: int = 256
    dropout: Optional[float] = None


@dataclass
class TrainingConfig:
    """训练超参数配置。"""

    output_root: str = "../../model"
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 2e-5
    num_train_epochs: float = 3
    weight_decay: float = 0.01
    optim: str = "adamw_torch_fused"
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    bf16: bool = True
    fp16: bool = False
    use_cpu: bool = False
    seed: int = 42
    logging_steps: int = 50
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 1
    load_best_model_at_end: bool = True
    dataloader_num_workers: int = 0
    gradient_accumulation_steps: int = 1
    report_to: List[str] = field(default_factory=list)


@dataclass
class TaskConfig:
    """单个分类任务的配置。"""

    output_dir: str = ""
    metric_for_best_model: str = "macro_f1"
    greater_is_better: bool = True
    top_k: int = 3
    threshold: float = 0.5


@dataclass
class OptimizationConfig:
    """压缩实验配置：量化 / 剪枝 / 低秩分解。"""

    # 产物子目录，相对每个任务的模型目录
    output_dir: str = "optimized"
    # 半精度量化种类，可选项：fp16 / bf16
    dtypes: List[str] = field(default_factory=lambda: ["fp16", "bf16"])
    int8_dynamic: bool = True
    # 非结构化剪枝比例（置零比例）
    unstructured_ratios: List[float] = field(default_factory=lambda: [0.3, 0.5])
    # 结构化剪枝：FFN 中间层保留比例
    structured_keep_ratios: List[float] = field(default_factory=lambda: [0.75, 0.5])
    # 低秩分解：秩占 min(in, out) 的比例
    low_rank_ratios: List[float] = field(default_factory=lambda: [0.75, 0.5, 0.25])
    eval_batch_size: int = 64
    # 吞吐测试的批大小与样本数
    latency_batch_size: int = 8
    latency_samples: int = 512
    # CPU 单条延迟测试的样本数（批大小固定为 1）
    latency_single_samples: int = 64
    latency_rounds: int = 3
    # 调试用：限制测试样本数（null 表示全量）
    max_test_samples: Optional[int] = None
    # 报告与结果文件，相对项目根目录
    report_path: str = "../model_optimization_report.md"
    results_path: str = "../../model/optimization_results.json"


@dataclass
class ProjectConfig:
    """完整项目配置（含解析后的绝对路径）。"""

    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    training: TrainingConfig
    tasks: Dict[str, TaskConfig]
    optimization: OptimizationConfig
    project_root: Path
    config_file: Path
    data_path: Path
    raw_data_path: Path
    pretrained_path: Path
    output_root: Path
    split_export_dir: Path
    report_path: Path
    results_path: Path

    @property
    def problem_type(self) -> Dict[str, str]:
        return dict(PROBLEM_TYPES)

    def task(self, task_key: str) -> TaskConfig:
        if task_key not in self.tasks:
            raise KeyError(f"未定义的任务：{task_key}，可选 {sorted(self.tasks)}")
        return self.tasks[task_key]

    def task_output_dir(self, task_key: str) -> Path:
        """任务输出目录：<output_root>/<task.output_dir>"""
        return self.output_root / self.task(task_key).output_dir

    def best_model_dir(self, task_key: str) -> Path:
        """任务最优模型目录：<任务输出目录>/best"""
        return self.task_output_dir(task_key) / "best"

    def optimized_dir(self, task_key: str) -> Path:
        """任务压缩模型目录：<任务输出目录>/<optimization.output_dir>"""
        return self.task_output_dir(task_key) / self.optimization.output_dir

    def snapshot(self) -> Dict[str, Any]:
        """导出可序列化的配置快照，用于随模型一起归档。"""
        payload = dataclasses.asdict(self)
        for key in (
            "project_root",
            "config_file",
            "data_path",
            "raw_data_path",
            "pretrained_path",
            "output_root",
            "split_export_dir",
            "report_path",
            "results_path",
        ):
            payload[key] = str(payload[key])
        return payload


def _flat(cls: type, raw: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """从 dict 中挑出目标 dataclass 声明的字段，多余字段忽略。"""
    if not raw:
        return {}
    allowed = set(cls.__dataclass_fields__)
    return {key: value for key, value in raw.items() if key in allowed}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """递归合并配置字典，override 优先。"""
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_file: str | Path | None = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> ProjectConfig:
    """加载配置文件并应用覆盖项，返回 ProjectConfig。

    Args:
        config_file: 配置文件路径，默认使用项目根目录下的 config.yaml。
        overrides: 形如 {"training": {"num_train_epochs": 1}} 的嵌套覆盖项。
    """
    config_path = Path(config_file) if config_file else DEFAULT_CONFIG_FILE
    config_path = config_path.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if overrides:
        raw = _deep_merge(raw, overrides)

    project_root = config_path.parent

    data = DataConfig(**_flat(DataConfig, raw.get("data")))
    split = SplitConfig(**_flat(SplitConfig, raw.get("split")))
    model = ModelConfig(**_flat(ModelConfig, raw.get("model")))
    training = TrainingConfig(**_flat(TrainingConfig, raw.get("training")))
    optimization = OptimizationConfig(**_flat(OptimizationConfig, raw.get("optimization")))

    raw_tasks = raw.get("tasks") or {}
    tasks: Dict[str, TaskConfig] = {}
    for task_key in TASK_KEYS:
        if task_key not in raw_tasks:
            raise KeyError(f"配置缺少任务定义：tasks.{task_key}")
        task_cfg = TaskConfig(**_flat(TaskConfig, raw_tasks[task_key]))
        if not task_cfg.output_dir:
            task_cfg.output_dir = f"{task_key}_label"
        tasks[task_key] = task_cfg

    config = ProjectConfig(
        data=data,
        split=split,
        model=model,
        training=training,
        tasks=tasks,
        optimization=optimization,
        project_root=project_root,
        config_file=config_path,
        data_path=resolve_path(data.path, project_root),
        raw_data_path=resolve_path(data.raw_path, project_root),
        pretrained_path=resolve_path(model.pretrained_path, project_root),
        output_root=resolve_path(training.output_root, project_root),
        split_export_dir=resolve_path(data.export_dir, project_root),
        report_path=resolve_path(optimization.report_path, project_root),
        results_path=resolve_path(optimization.results_path, project_root),
    )
    _validate(config)
    return config


def _validate(config: ProjectConfig) -> None:
    """配置合法性校验：尽早暴露路径或参数错误。"""
    if config.data.source not in {"text", "arrow"}:
        raise ValueError(f"data.source 只支持 text / arrow，当前 {config.data.source!r}")
    if not config.data_path.exists():
        raise FileNotFoundError(f"训练数据不存在：{config.data_path}")
    if not config.pretrained_path.exists():
        raise FileNotFoundError(f"预训练模型目录不存在：{config.pretrained_path}")
    if config.data.source == "arrow":
        if not config.data.text_template:
            raise ValueError("arrow 数据源必须配置 data.text_template")
        if not config.data.label_column:
            raise ValueError("arrow 数据源必须配置 data.label_column")
    else:
        if config.data.text_column not in config.data.columns:
            raise ValueError(
                f"data.text_column={config.data.text_column!r} 不在 data.columns={config.data.columns} 中"
            )
        if config.data.label_column not in config.data.columns:
            raise ValueError(
                f"data.label_column={config.data.label_column!r} 不在 data.columns={config.data.columns} 中"
            )
    if not 0 < config.split.test_size < 1:
        raise ValueError(f"split.test_size 必须位于 (0, 1)，当前 {config.split.test_size}")
    if not 0 < config.split.val_size < 1:
        raise ValueError(f"split.val_size 必须位于 (0, 1)，当前 {config.split.val_size}")
    if config.split.test_size + config.split.val_size >= 1:
        raise ValueError("split.test_size + split.val_size 必须小于 1")
    if config.data.min_label_count < 1:
        raise ValueError(f"data.min_label_count 必须 >= 1，当前 {config.data.min_label_count}")
    if config.data.single_label_strategy not in {"expand", "primary"}:
        raise ValueError(
            f"data.single_label_strategy 只支持 expand / primary，当前 {config.data.single_label_strategy!r}"
        )
    if config.model.max_length < 2:
        raise ValueError(f"model.max_length 至少为 2，当前 {config.model.max_length}")

    valid_dtypes = {"fp16", "bf16"}
    invalid = set(config.optimization.dtypes) - valid_dtypes
    if invalid:
        raise ValueError(f"optimization.dtypes 含不支持的精度 {sorted(invalid)}，可选 {sorted(valid_dtypes)}")
    for key in ("unstructured_ratios", "structured_keep_ratios", "low_rank_ratios"):
        for ratio in getattr(config.optimization, key):
            if not 0 < ratio < 1:
                raise ValueError(f"optimization.{key} 的比例必须位于 (0, 1)，当前 {ratio}")
    if config.optimization.latency_batch_size < 1:
        raise ValueError("optimization.latency_batch_size 必须 >= 1")
