import os


class BaseConfig(object):
    def __init__(self):
        root_path = os.path.abspath(os.path.dirname(__file__))
        self.root_path = root_path
        self.original_data_path = root_path + '/file/original_file/movie_info_final.txt'
        self.handle_save_path = root_path + '/file/original_file/data_balance.tsv'
        self.train_path = root_path + '/file/data/train.tsv'
        self.test_path = root_path + '/file/data/test.tsv'
        self.dev_path = root_path + '/file/data/val.tsv'


class RFConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.embedding_path_rf =  self.root_path + '/file/model/rf_embedding2.pkl'
        self.model_path_rf = self.root_path + '/file/model/rf_model2.pkl'


class FasttextConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.template_path = self.root_path + '/file/data/{{type}}_{{split_mode}}_dataset.txt'
        self.template_save_model_path = self.root_path + '/file/model/fasttext_{{autotune}}_{{split_mode}}_model.bin'


class BertConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_dir = self.root_path + '/file/model/bert_model/best'


class LLMConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # 大模型相关配置（DeepSeek）
        self.api_key_env = "DEEPSEEK_API_KEY"          # 从环境变量读取 API Key
        self.base_url = "https://api.deepseek.com"
        self.model = "deepseek-chat"
        self.class_path = self.root_path + "/file/data/class.txt"
        # 预测服务端口（
        self.service_port = 8002

    def get_genres(self):
        """读取 class.txt，返回 ['剧情','喜剧',...]（下标即标签 id）"""
        with open(self.class_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]