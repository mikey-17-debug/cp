"""电影类型分类项目（chinese-bert-wwm）。

模块划分：
    config   —— 配置解析（唯一配置来源 config.yaml）
    data     —— 数据读取、标签空间、切分、Dataset
    model    —— 序列分类模型构建
    metrics  —— 单标签 / 多标签评估指标
    trainer  —— 训练与评估编排
    predictor—— 已保存模型的推理
    utils    —— 日志、随机种子、设备等通用工具
"""
