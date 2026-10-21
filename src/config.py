import os


class BaseConfig(object):
    def __init__(self):
        root_path = os.path.abspath(os.path.dirname(__file__))
        self.root_path = root_path
        self.original_data_path = root_path + '/file/original_file/movie_info.txt'
        self.handle_save_path = root_path + '/file/original_file/movie_info.tsv'
        self.train_path = root_path + '/file/data/train.txt'
        self.test_path = root_path + '/file/data/test.txt'
        self.dev_path = root_path + '/file/data/dev.txt'


class RFConfig(BaseConfig):
    def __init__(self):
        super().__init__()


class FasttextConfig(BaseConfig):
    def __init__(self):
        super().__init__()


class BertConfig(BaseConfig):
    def __init__(self):
        super().__init__()


class LLMConfig(BaseConfig):
    def __init__(self):
        super().__init__()
