"""数据层：读取原始文件、解析类型字段、构建标签空间、切分并封装 Dataset。

关键设计：
    * 切分在「影片级」完成（先切分、后展开），避免同一摘要同时出现在训练集与测试集造成泄漏；
    * 单标签样本由类型列表展开而来（同一摘要 + 多个不同标签）；
    * 多标签样本以 multi-hot 向量表示。
"""

from __future__ import annotations

import ast
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import SINGLE_LABEL, DataConfig, ProjectConfig, SplitConfig

logger = logging.getLogger(__name__)

# 解析后的类型列表存放列
GENRES_COLUMN = "_genres"


def parse_labels(raw: object) -> List[str]:
    """把 CSV 里的 "['剧情', '犯罪']" 解析为 Python 列表。

    对空值、异常格式做容错，避免个别脏数据中断整个流程。
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    text = str(raw).strip()
    if not text:
        return []
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [text]


def _load_text_frame(config: ProjectConfig) -> pd.DataFrame:
    """读取分隔符文本文件（原中文数据集格式）。"""
    data = config.data
    read_kwargs: Dict[str, object] = {
        "sep": data.separator,
        "encoding": data.encoding,
        "dtype": str,
        "keep_default_na": False,
        "header": None,
        "names": data.columns,
    }
    if data.has_header:
        read_kwargs["skiprows"] = 1

    frame = pd.read_csv(config.data_path, **read_kwargs)
    missing = [name for name in (data.label_column, data.text_column) if name not in frame.columns]
    if missing:
        raise ValueError(f"数据文件缺少列 {missing}，实际列：{list(frame.columns)}")
    return frame


def _template_fields(template: str) -> List[str]:
    """从文本模板中提取字段占位符，例如 "{title}. {storyline}" -> [title, storyline]。"""
    return sorted(set(re.findall(r"\{(\w+)\}", template)))


def dedupe_repeated_text(text: str) -> str:
    """去掉「整段重复两遍」的文本。

    IMDb-Media 的 storyline 绝大多数是 "A A" 形式的二倍重复，
    直接送入模型会让序列长度翻倍并造成不必要的截断。
    """
    stripped = str(text).strip()
    length = len(stripped)
    if length < 80:
        return stripped

    half = length // 2
    left, right = stripped[:half].strip(), stripped[half:].strip()
    if left and left == right:
        return left

    # 两份内容长度不一定完全相等：用开头片段定位第二份的起点
    probe = stripped[:40]
    position = stripped.find(probe, 40)
    if position > 0:
        first, second = stripped[:position].strip(), stripped[position:].strip()
        if first and second:
            if second == first:
                return first
            # 第二份是残段（略短）或尾部多了一点内容
            if first.startswith(second) and len(second) >= 0.6 * len(first):
                return first
            if second.startswith(first):
                return second
    return stripped


def _dedupe_fields(frame: pd.DataFrame, fields: Sequence[str]) -> int:
    """对指定字段做整段重复去除，返回发生变化的字段值个数。"""
    changed = 0
    before_chars = after_chars = 0
    for name in dict.fromkeys(fields):
        original = frame[name].fillna("").astype(str).str.strip()
        cleaned = original.map(dedupe_repeated_text)
        changed += int((cleaned != original).sum())
        before_chars += int(original.str.len().sum())
        after_chars += int(cleaned.str.len().sum())
        frame[name] = cleaned
    if changed:
        logger.info(
            "文本去重：修改 %d 个字段值，总字符数 %d -> %d（约 %.1f%%）",
            changed,
            before_chars,
            after_chars,
            100 * (1 - after_chars / max(before_chars, 1)),
        )
    return changed


def _render_template(frame: pd.DataFrame, template: str) -> pd.Series:
    """按模板逐行拼接字段，缺失值按空串处理。"""
    fields = _template_fields(template)
    columns = {}
    for name in fields:
        if name not in frame.columns:
            raise ValueError(f"文本模板引用的字段 {name!r} 不在数据列 {list(frame.columns)} 中")
        columns[name] = frame[name].fillna("").astype(str).str.strip()

    def render(row: pd.Series) -> str:
        return template.format(**{name: row[name] for name in fields})

    return pd.DataFrame(columns).apply(render, axis=1)


def _load_arrow_frame(config: ProjectConfig) -> pd.DataFrame:
    """读取 HuggingFace datasets 落盘的 Arrow 目录，按模板拼出文本列。"""
    from datasets import load_from_disk

    data = config.data
    loaded = load_from_disk(str(config.data_path))
    if hasattr(loaded, "keys"):
        candidates = list(loaded.keys())
        split_name = data.arrow_split or ("train" if "train" in candidates else candidates[0])
        if split_name not in candidates:
            raise ValueError(f"arrow 数据源没有 split {split_name!r}，可选 {candidates}")
        dataset = loaded[split_name]
    else:
        dataset = loaded

    needed = _template_fields(data.text_template) + list(data.fallback_text_fields) + [data.label_column]
    unique = list(dict.fromkeys(name for name in needed if name))
    absent = [name for name in unique if name not in dataset.column_names]
    if absent:
        raise ValueError(f"arrow 数据源缺少列 {absent}，实际列：{dataset.column_names}")

    frame = pd.DataFrame({name: dataset[name] for name in unique})
    if data.dedupe_repeated_text:
        _dedupe_fields(
            frame,
            _template_fields(data.text_template) + list(data.fallback_text_fields),
        )
    frame[data.text_column] = _render_template(frame, data.text_template)

    # 模板渲染为空（例如剧情简介缺失）时，按配置顺序回退到其他字段
    for fallback in data.fallback_text_fields:
        if fallback == data.text_column:
            continue
        fill = frame[fallback].fillna("").astype(str).str.strip()
        need = (frame[data.text_column].str.strip() == "") & (fill != "")
        frame.loc[need, data.text_column] = fill[need]

    logger.info("载入 arrow 数据：%s | split=%s | 样本 %d 条", config.data_path, split_name, len(frame))
    return frame


def load_dataframe(config: ProjectConfig) -> pd.DataFrame:
    """按配置读取原始数据，返回带解析后类型列表的 DataFrame。"""
    data = config.data
    if data.source == "arrow":
        frame = _load_arrow_frame(config)
    else:
        frame = _load_text_frame(config)

    if data.limit:
        frame = frame.head(data.limit)

    frame = frame.copy()
    frame[data.text_column] = frame[data.text_column].fillna(data.missing_text).astype(str).str.strip()
    if data.dedupe_repeated_text:
        frame[data.text_column] = frame[data.text_column].map(dedupe_repeated_text)
    frame = frame[frame[data.text_column] != ""].reset_index(drop=True)

    frame[GENRES_COLUMN] = frame[data.label_column].map(parse_labels)
    frame = frame[frame[GENRES_COLUMN].map(len) > 0].reset_index(drop=True)

    if data.drop_duplicated:
        before = len(frame)
        frame = frame.drop_duplicates(subset=[data.text_column, data.label_column]).reset_index(drop=True)
        if before != len(frame):
            logger.info("按 (文本, 标签) 去重：%d -> %d", before, len(frame))

    if data.drop_conflicting_labels:
        label_counts = frame.groupby(data.text_column)[data.label_column].nunique()
        conflict_texts = label_counts[label_counts > 1].index
        if len(conflict_texts):
            before = len(frame)
            frame = frame[~frame[data.text_column].isin(conflict_texts)].reset_index(drop=True)
            logger.info(
                "剔除跨类别冲突文本：%d 条文本 / %d 行 -> %d 行",
                len(conflict_texts),
                before,
                len(frame),
            )

    logger.info("载入影片样本 %d 条（文本列=%s，标签列=%s）", len(frame), data.text_column, data.label_column)
    return frame


def build_label_space(
    frame: pd.DataFrame,
    min_label_count: int,
    exclude_labels: Sequence[str] = (),
) -> List[str]:
    """统计类型频次，剔除配置中排除的标签与低频类型，返回排序后的标签空间。"""
    counter: Counter = Counter()
    for genres in frame[GENRES_COLUMN]:
        counter.update(genres)

    excluded = set(exclude_labels)
    classes = sorted(
        genre for genre, count in counter.items() if count >= min_label_count and genre not in excluded
    )
    if not classes:
        raise ValueError(f"标签空间为空：所有类型出现次数都低于 min_label_count={min_label_count}")

    dropped = sorted((genre, count) for genre, count in counter.items() if count < min_label_count)
    logger.info(
        "标签空间 %d 个（阈值 %d，排除 %d 个格式类）；剔除低频类型 %d 个：%s",
        len(classes),
        min_label_count,
        len(excluded),
        len(dropped),
        ", ".join(f"{genre}({count})" for genre, count in dropped) or "无",
    )
    kept_excluded = sorted(genre for genre in excluded if genre in counter)
    if kept_excluded:
        logger.info("按配置排除的标签：%s", ", ".join(kept_excluded))
    return classes


def filter_label_space(frame: pd.DataFrame, classes: Sequence[str]) -> pd.DataFrame:
    """把样本类型列表裁剪到标签空间内，并丢弃裁剪后没有类型的样本。"""
    keep = set(classes)
    frame = frame.copy()
    frame[GENRES_COLUMN] = frame[GENRES_COLUMN].map(lambda genres: [g for g in genres if g in keep])
    return frame[frame[GENRES_COLUMN].map(len) > 0].reset_index(drop=True)


# 分层抽样键的兜底桶：主类型样本过少时并入该桶，避免 sklearn 直接拒绝切分
RARE_STRATIFY_BUCKET = "__其他__"
# 主类型至少要有这么多条样本才单独成为一个分层组
MIN_STRATIFY_CLASS_SIZE = 10


def _stratify_key(frame: pd.DataFrame) -> pd.Series:
    """分层抽样的键：取样本的主类型（类型列表首项）。

    主类型分布长尾明显（例如「传记」很少排在第一位），样本数不足的类别
    会让两级切分失败，这里统一并入兜底桶，保证每组的样本量都足够。
    """
    keys = frame[GENRES_COLUMN].map(lambda genres: genres[0])
    counts = keys.value_counts()
    rare = counts[counts < MIN_STRATIFY_CLASS_SIZE].index
    return keys.where(~keys.isin(rare), RARE_STRATIFY_BUCKET)


def split_dataframe(frame: pd.DataFrame, split: SplitConfig) -> Dict[str, pd.DataFrame]:
    """按 训练 / 验证 / 测试 三段切分，可选按主类型分层抽样。"""
    stratify = _stratify_key(frame) if split.stratify else None
    try:
        train_val, test = train_test_split(
            frame,
            test_size=split.test_size,
            random_state=split.seed,
            stratify=stratify,
        )
        train_stratify = _stratify_key(train_val) if split.stratify else None
        val_ratio = split.val_size / (1.0 - split.test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=split.seed,
            stratify=train_stratify,
        )
    except ValueError as error:
        logger.warning("分层抽样失败（%s），退回随机切分", error)
        train_val, test = train_test_split(frame, test_size=split.test_size, random_state=split.seed)
        val_ratio = split.val_size / (1.0 - split.test_size)
        train, val = train_test_split(train_val, test_size=val_ratio, random_state=split.seed)

    parts = {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    logger.info(
        "影片级切分：train=%d, val=%d, test=%d",
        len(parts["train"]),
        len(parts["val"]),
        len(parts["test"]),
    )
    return parts


def build_single_label_samples(
    frame: pd.DataFrame,
    classes: Sequence[str],
    data: DataConfig,
) -> Tuple[List[str], np.ndarray]:
    """构造单标签样本，返回 (文本列表, 标签 id 数组)。

    两种策略：
        * expand：一条多类型影片展开成多条「相同文本 + 不同标签」样本；
        * primary：只保留类型列表的首项（主类型），一条影片一个样本，
          避免同一文本对应互相矛盾的监督信号。
    """
    if data.single_label_strategy == "primary":
        primary = frame[[data.text_column, GENRES_COLUMN]].copy()
        primary["_label"] = primary[GENRES_COLUMN].map(lambda genres: genres[0])
        if data.drop_duplicated:
            before = len(primary)
            primary = primary.drop_duplicates(subset=[data.text_column, "_label"]).reset_index(drop=True)
            if before != len(primary):
                logger.info("按 (文本, 主类型) 去重：%d -> %d", before, len(primary))
        exploded = primary
    else:
        exploded = frame[[data.text_column, GENRES_COLUMN]].explode(GENRES_COLUMN)
        exploded = exploded.rename(columns={GENRES_COLUMN: "_label"})
        if data.drop_duplicated:
            before = len(exploded)
            exploded = exploded.drop_duplicates().reset_index(drop=True)
            if before != len(exploded):
                logger.info("按 (文本, 单标签) 去重：%d -> %d", before, len(exploded))

    label_to_id = {label: index for index, label in enumerate(classes)}
    labels = exploded["_label"].map(label_to_id).to_numpy(dtype=np.int64)
    texts = exploded[data.text_column].tolist()
    return texts, labels


def build_multi_label_samples(
    frame: pd.DataFrame,
    classes: Sequence[str],
    data: DataConfig,
) -> Tuple[List[str], np.ndarray]:
    """把类型列表转换为 multi-hot 向量，返回 (文本列表, 标签矩阵)。"""
    label_to_id = {label: index for index, label in enumerate(classes)}
    matrix = np.zeros((len(frame), len(classes)), dtype=np.float32)
    for row, genres in enumerate(frame[GENRES_COLUMN]):
        for genre in genres:
            matrix[row, label_to_id[genre]] = 1.0
    texts = frame[data.text_column].tolist()
    return texts, matrix


class GenreDataset(Dataset):
    """预分词数据集。

    在构造阶段一次性批量分词全部文本（Rust 分词器一次调用完成），
    训练循环里只做切片，避免每个 batch 都在 Python 端逐条分词造成 GPU 空等。
    批内仍由 collator 动态填充。
    """

    # 分块编码，控制一次性分词的内存峰值
    ENCODE_CHUNK_SIZE = 20000

    def __init__(
        self,
        texts: Sequence[str],
        labels: np.ndarray,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        problem_type: str,
    ) -> None:
        self.labels = labels
        self.problem_type = problem_type
        self.texts = list(texts)

        encoded_chunks: List[List[int]] = []
        for start in range(0, len(self.texts), self.ENCODE_CHUNK_SIZE):
            chunk = self.texts[start : start + self.ENCODE_CHUNK_SIZE]
            encoded = tokenizer(
                chunk,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            encoded_chunks.extend(encoded["input_ids"])

        if not encoded_chunks:
            self.input_ids = np.zeros((0, 1), dtype=np.int32)
            self.lengths = np.zeros((0,), dtype=np.int32)
            return

        lengths = np.fromiter((len(ids) for ids in encoded_chunks), dtype=np.int32, count=len(encoded_chunks))
        width = int(lengths.max())
        # 统一放进 [N, 实际最大长度] 的 int32 矩阵，取用时按真实长度切片
        self.input_ids = np.full((len(encoded_chunks), width), tokenizer.pad_token_id, dtype=np.int32)
        for row, ids in enumerate(encoded_chunks):
            self.input_ids[row, : len(ids)] = ids
        self.lengths = lengths

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        length = int(self.lengths[index])
        input_ids = torch.from_numpy(self.input_ids[index, :length].astype(np.int64))
        item = {
            "input_ids": input_ids,
            "attention_mask": torch.ones(length, dtype=torch.long),
        }
        label = self.labels[index]
        if self.problem_type == "multi_label_classification":
            item["labels"] = torch.tensor(np.asarray(label, dtype=np.float32), dtype=torch.float)
        else:
            item["labels"] = torch.tensor(int(label), dtype=torch.long)
        return item


class DynamicPadCollator:
    """批内动态填充：按当前 batch 最长序列补齐，缩短序列不做无谓填充。"""

    _PAD_KEYS = ("input_ids", "attention_mask", "token_type_ids")

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, torch.Tensor] = {}
        for key in self._PAD_KEYS:
            if key not in features[0]:
                continue
            fill = self.pad_token_id if key == "input_ids" else 0
            length = max(feature[key].shape[0] for feature in features)
            padded = torch.full(
                (len(features), length), fill, dtype=features[0][key].dtype
            )
            for row, feature in enumerate(features):
                size = feature[key].shape[0]
                padded[row, :size] = feature[key][:size]
            batch[key] = padded
        batch["labels"] = torch.stack([feature["labels"] for feature in features])
        return batch


def build_tokenizer(config: ProjectConfig) -> PreTrainedTokenizerBase:
    """加载分词器，并确保具备可用的 padding token。

    transformers 5.x 默认 use_fast=False，这里显式启用快速分词器（Rust 实现），
    否则十万级样本逐条分词的耗时会长到不可接受。
    """
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        fallback = tokenizer.unk_token or tokenizer.eos_token
        if fallback is None:
            raise ValueError("分词器没有 pad/unk token，无法动态填充")
        tokenizer.pad_token = fallback
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(fallback)
        logger.warning("分词器缺少 pad token，已回退为 %s", tokenizer.pad_token)
    logger.info("分词器：%s（词表 %d）", type(tokenizer).__name__, tokenizer.vocab_size)
    return tokenizer


def export_splits(
    config: ProjectConfig,
    task_key: str,
    splits: Dict[str, Tuple[List[str], np.ndarray]],
    classes: Sequence[str],
) -> Path | None:
    """把实际参与训练 / 验证 / 测试的样本写回磁盘。

    输出到 <data.export_dir>/<tasks.*.output_dir>/，每个切分一个 CSV：
        * 单标签：summary, label
        * 多标签：summary, labels（多个类型用 | 分隔）
    """
    data = config.data
    if not data.export_splits:
        return None

    target = config.split_export_dir / config.task(task_key).output_dir
    target.mkdir(parents=True, exist_ok=True)

    is_single = task_key == SINGLE_LABEL
    for split_name in ("train", "val", "test"):
        texts, labels = splits[split_name]
        if is_single:
            frame = pd.DataFrame(
                {
                    data.text_column: texts,
                    "label": [classes[int(label)] for label in labels],
                }
            )
        else:
            frame = pd.DataFrame(
                {
                    data.text_column: texts,
                    "labels": [
                        "|".join(classes[index] for index, flag in enumerate(row) if flag > 0.5)
                        for row in labels
                    ],
                }
            )
        frame.to_csv(target / f"{split_name}.csv", index=False, encoding=data.export_encoding)

    logger.info("[%s] 训练/验证/测试集已写回 %s", task_key, target)
    return target


def prepare_data(config: ProjectConfig, task_key: str) -> Dict[str, object]:
    """完整数据准备流程：读取 -> 标签空间 -> 切分 -> 构造样本 -> 写回切分文件。

    Returns:
        {"classes": [...], "splits": {"train": (texts, labels), ...}, "frame_sizes": {...}}
    """
    frame = load_dataframe(config)
    classes = build_label_space(frame, config.data.min_label_count, config.data.exclude_labels)
    frame = filter_label_space(frame, classes)
    parts = split_dataframe(frame, config.split)

    is_single = task_key == SINGLE_LABEL
    builder = build_single_label_samples if is_single else build_multi_label_samples
    splits = {name: builder(part, classes, config.data) for name, part in parts.items()}

    logger.info(
        "[%s] 样本量：train=%d, val=%d, test=%d, 标签数=%d",
        task_key,
        len(splits["train"][0]),
        len(splits["val"][0]),
        len(splits["test"][0]),
        len(classes),
    )
    export_splits(config, task_key, splits, classes)
    return {
        "classes": classes,
        "splits": splits,
        "frame_sizes": {name: len(part) for name, part in parts.items()},
    }
