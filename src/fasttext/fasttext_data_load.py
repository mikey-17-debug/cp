import jieba
import pandas as pd

from src.config import FasttextConfig


def data_load_path(is_train=False, is_test=False, is_dev=False, split_mode='char'):
    config_obj = FasttextConfig()
    template_path: str = config_obj.template_path
    if is_train:
        template_path = template_path.replace('{{type}}', 'train')
    elif is_test:
        template_path = template_path.replace('{{type}}', 'test')
    elif is_dev:
        template_path = template_path.replace('{{type}}', 'dev')
    return template_path.replace('{{split_mode}}', split_mode)


def get_save_model_path(autotune=False, split_mode='char'):
    config_obj = FasttextConfig()
    template_save_model_path = config_obj.template_save_model_path
    return template_save_model_path.replace('{{autotune}}', 'autotune' if autotune else 'no_autotune').replace(
        '{{split_mode}}', split_mode)


def processing_data(is_train=False, is_test=False, is_dev=False ,split_mode='char'):
    config_obj = FasttextConfig()
    if is_train:
        df = pd.read_csv(config_obj.train_path,sep='\t')
    elif is_test:
        df = pd.read_csv(config_obj.test_path,sep='\t')
    elif is_dev:
        df = pd.read_csv(config_obj.dev_path,sep='\t')
    if split_mode == 'char':
        df['summary'] = df['summary'].apply(lambda x: " ".join(list(x)[:128]))
    elif split_mode == 'jieba':
        df['summary'] = df['summary'].apply(lambda x: " ".join(jieba.lcut(x)[:128]))


    summary_list = df['summary'].tolist()
    label_list = df['label'].tolist()
    lines = []
    for index in range(len(summary_list)):
        lines.append(f'__label__{label_list[index]} {summary_list[index]}\n')
    with open(data_load_path(is_train, is_test, is_dev, split_mode), 'w', encoding='utf-8') as f:
        f.writelines(lines)

if __name__ == '__main__':
    split_mode_list = ['char','jieba']
    for split_mode in split_mode_list:
        processing_data(is_train=True,split_mode=split_mode)
        processing_data(is_test=True,split_mode=split_mode)
        processing_data(is_dev=True,split_mode=split_mode)
