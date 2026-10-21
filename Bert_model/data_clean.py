import json
import pandas as pd
from bert_config import BertConfig


def read_file(file_path: str) -> list[str]|None:
    try:
        with open(file_path, 'r', encoding='utf-8') as file_read:
            file = file_read.readlines()
    except Exception as e:
        print(f'{e}')
    else:
        return file


def clean_file(file, path):
    id_list = []
    # image_url_list = []
    name_list = []
    label_list = []
    summary_list = []

    for item in file:
        i = json.loads(item)

        id_list.append(i['id'])
        # image_url_list.append(i['image_url'])
        name_list.append(i['name'])
        summary_list.append(i['summary'])
        label_list.append(i['genres'])

    df = pd.DataFrame({
        'label': label_list,
        'name': name_list,
        'summary': summary_list,
    })
    df.to_csv(path, index_label=False)



if __name__ == '__main__':
    bert_config = BertConfig()
    temp_file = read_file(bert_config.original_data_path)
    clean_file(temp_file, bert_config.clean_data_path)

