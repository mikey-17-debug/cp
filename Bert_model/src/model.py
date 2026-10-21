"""模型层：基于预训练权重构建序列分类模型。

两个任务共用同一个模型类，差异只体现在 config.problem_type：
    * single_label_classification -> 交叉熵（每个样本一个类别）
    * multi_label_classification  -> 二分类交叉熵（每个类别独立判定）
"""

from __future__ import annotations

import logging
from typing import Dict

from transformers import AutoConfig, AutoModelForSequenceClassification

from .config import PROBLEM_TYPES, ProjectConfig

logger = logging.getLogger(__name__)


def build_model(
    config: ProjectConfig,
    task_key: str,
    classes: list,
) -> AutoModelForSequenceClassification:
    """加载预训练权重并改造成序列分类模型。"""
    model_config = AutoConfig.from_pretrained(config.pretrained_path)
    model_config.num_labels = len(classes)
    model_config.problem_type = PROBLEM_TYPES[task_key]

    # 写入标签映射，使保存后的模型可直接被 pipeline 使用
    id2label: Dict[int, str] = {index: label for index, label in enumerate(classes)}
    model_config.id2label = id2label
    model_config.label2id = {label: index for index, label in id2label.items()}

    if config.model.dropout is not None:
        model_config.hidden_dropout_prob = config.model.dropout
        model_config.attention_probs_dropout_prob = config.model.dropout

    model = AutoModelForSequenceClassification.from_pretrained(
        config.pretrained_path, config=model_config
    )
    logger.info(
        "加载预训练模型 %s：num_labels=%d, problem_type=%s, 新增分类头将随机初始化",
        config.pretrained_path.name,
        model_config.num_labels,
        model_config.problem_type,
    )
    return model
