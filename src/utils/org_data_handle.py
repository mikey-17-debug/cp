"""
原数据处理
"""
import json


import pandas as pd

from src.config import BaseConfig


def main():

    summary_list = []
    genres_list = []
    config_obj = BaseConfig()
    genre_top10 = ['喜剧', '动作', '恐怖', '爱情', '悬疑', '动画', '科幻', '音乐', '犯罪', '纪录片']
    with open(config_obj.original_data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data_dict = json.loads(line)
            genres:list = data_dict['genres']
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
    df['genre'] = genres_list
    df.to_csv(config_obj.handle_save_path, index=False, encoding='utf-8', sep='\t')
if __name__ == '__main__':
    main()