class Config(object):
    def __init__(self):
        # 数据集路径
        self.train_path = "../01-data/train.txt"
        self.dev_path = "../01-data/dev.txt"
        self.test_path = "../01-data/test.txt"
        self.class_path = "../01-data/class.txt"

        # 大模型相关配置（DeepSeek）
        self.api_key_env = "DEEPSEEK_API_KEY"          # 从环境变量读取 API Key
        self.base_url = "https://api.deepseek.com"
        self.model = "deepseek-chat"

        # 预测服务端口（
        self.service_port = 8002

    def get_genres(self):
        """读取 class.txt，返回 ['剧情','喜剧',...]（下标即标签 id）"""
        with open(self.class_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]