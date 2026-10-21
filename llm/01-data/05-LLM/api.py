"""
目标：对 LLM（DeepSeek）电影类型预测进行 api 封装

步骤：
1. 实例化flask对象
2. 绑定路由与视图函数
"""
from flask import Flask, request, render_template
import json
from predict import predict
from config import Config


# 步骤：
# 1. 实例化flask对象
config = Config()
app = Flask(__name__)
# 2. 绑定路由与视图函数


@app.route("/", methods=["GET"])
def page():
    return render_template("predict.html")


@app.route('/predict', methods=['POST', 'GET'])
def api_predict():
    """
    根据输入的电影简介，预测电影类型

    1. 获取输入
    2. 格式校验
    3. 调用预测 返回结果
    """
    input_data = request.json or {}
    if "text" not in input_data or not str(input_data["text"]).strip():
        return json.dumps({"code": "FAIL", "result": "未找到有效的text字段"}, ensure_ascii=False)

    result = predict(input_data)
    return json.dumps(result, ensure_ascii=False)


if __name__ == '__main__':
    app.run(port=config.service_port, debug=False)