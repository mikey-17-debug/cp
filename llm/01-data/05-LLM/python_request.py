"""
目标：调用flask后端服务，拿到预测结果

步骤：
1. 定义url
2. 定义所需数据
3. 调用
4. 看结果
"""
import requests
import time

# 步骤：
# 1. 定义url - 地址（与 api.py 的 service_port 一致）
url = "http://localhost:8002/predict"
# 2. 定义所需数据
title = "你的名字。巫女世家出身的高中女孩宫水三叶在梦中变成东京的高中男孩立花泷，两人交换身份后发生了一系列奇幻的故事。"
# 3. 调用
start_time = time.time()
result = requests.post(url, json={"text": title},
                       headers={"Content-Type": "application/json"})

elapsed_time = (time.time() - start_time)
print(f"请求耗时: {elapsed_time:.2f} ms")
# print(result.json())
# 4. 看结果
print(result.json())
# print(type(result.json()))
#
# print(result.json()['pred_class'])