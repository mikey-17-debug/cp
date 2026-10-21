import os


class BaseConfig(object):
    def __init__(self):
        root_path = os.path.abspath(os.path.dirname(__file__))
        self.root_path = root_path
        self.original_data_path = root_path + '/file/original_file/movie_info_final.txt'
        self.handle_save_path = root_path + '/file/original_file/data_balance.tsv'
        self.train_path = root_path + '/file/data/train.csv'
        self.test_path = root_path + '/file/data/test.csv'
        self.dev_path = root_path + '/file/data/val.csv'


class RFConfig(BaseConfig):
    def __init__(self):
        super().__init__()


class FasttextConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.template_path = self.root_path + '/file/data/{{type}}_{{split_mode}}_dataset.txt'
        self.template_save_model_path = self.root_path + '/file/model/fasttext_{{autotune}}_{{split_mode}}_model.bin'


class BertConfig(BaseConfig):
    def __init__(self):
        super().__init__()


class LLMConfig(BaseConfig):
    def __init__(self):
        super().__init__()
