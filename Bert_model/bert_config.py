"""数据清洗配置。

路径统一从项目根目录的 config.yaml 读取，避免在代码里硬编码。
"""

from __future__ import annotations

from src.config import load_config


class BertConfig(object):
    def __init__(self, config_file: str | None = None) -> None:
        project_config = load_config(config_file)
        # 原始数据（movie_info.txt）作为输入
        self.original_data_path = str(project_config.raw_data_path)
        # 清洗结果写回训练数据文件（dataset/clean_data.txt）
        self.clean_data_path = str(project_config.data_path)
