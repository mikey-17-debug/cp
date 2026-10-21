"""
原数据处理
"""
import json
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import BaseConfig
# 固定比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
def split_summary_label(
    df: pd.DataFrame,
    *,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    输入 DataFrame（必须含 summary、label 两列），
    按 label 分层切分为 训练 / 验证 / 测试 三份，比例固定 8:1:1。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含 'summary' 和 'label' 两列。
    random_state : int
        随机种子，保证可复现。

    Returns
    -------
    (train_df, val_df, test_df)
        三个 DataFrame，均只含 summary、label 两列，且各列顺序一致。
    """
    # ---------- 1. 校验输入 ----------
    required = {"summary", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入 DataFrame 缺少列: {missing}，现有列: {list(df.columns)}")
    if df.empty:
        raise ValueError("输入 DataFrame 为空")
    if df["label"].isna().any():
        raise ValueError("label 列存在缺失值，请先处理")
    if (df["label"].value_counts() < 2).any():
        bad = df["label"].value_counts()
        bad = bad[bad < 2].index.tolist()
        raise ValueError(f"以下类别样本数 < 2，无法分层切分: {bad}")

    # 只保留 summary、label 两列，列顺序固定
    df = df[["summary", "label"]]

    # ---------- 2. 第一次切分：train(80%) vs temp(20%) ----------
    train_df, temp_df = train_test_split(
        df,
        test_size=VAL_RATIO + TEST_RATIO,   # 0.2
        stratify=df["label"],
        random_state=random_state,
        shuffle=True,
    )

    # ---------- 3. 第二次切分：temp(20%) → val(10%) / test(10%) ----------
    # 注意：sklearn 的 test_size 是占 temp 的比例，这里 = 0.1 / 0.2 = 0.5
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),  # 0.5
        stratify=temp_df["label"],
        random_state=random_state,
        shuffle=True,
    )

    # ---------- 4. 重置索引、统一列顺序 ----------
    train_df = train_df.reset_index(drop=True)[["summary", "label"]]
    val_df   = val_df.reset_index(drop=True)[["summary", "label"]]
    test_df  = test_df.reset_index(drop=True)[["summary", "label"]]

    return train_df, val_df, test_df
def main():

    summary_list = []
    genres_list = []
    config_obj = BaseConfig()
    genre_top10 = ['喜剧', '动作', '恐怖', '爱情', '悬疑', '动画', '科幻', '音乐', '犯罪', '纪录片']
    with open(config_obj.original_data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data_dict = json.loads(line)
            genres:list = data_dict['genres']
            summary:str = data_dict.get('summary',None)
            if summary is None :
                continue
            if '剧情' in genres:
                genres.remove('剧情')
            if len(genres) == 0:
                continue
            genre = genres[0] if genres[0] != '惊悚' else '恐怖'
            if genre not in genre_top10:
                continue
            genres_list.append(genre)
            summary_list.append(data_dict['summary'])
    df = pd.DataFrame()
    df['summary'] = summary_list
    df['label'] = genres_list
    df = df.dropna(subset=["summary", "label"])
    print(df.head(50))
    df.to_csv(config_obj.handle_save_path, index=False, encoding='utf-8', sep='\t')
if __name__ == '__main__':
    config_obj = BaseConfig()
    df = pd.read_csv(config_obj.handle_save_path, sep='\t')
    df['label'] = df['genre']
    train_df,test_df,dev_df = split_summary_label(df)
    train_df.to_csv(config_obj.train_path, index=False, encoding='utf-8', sep='\t')
    test_df.to_csv(config_obj.test_path, index=False, encoding='utf-8', sep='\t')
    dev_df.to_csv(config_obj.dev_path, index=False, encoding='utf-8', sep='\t')