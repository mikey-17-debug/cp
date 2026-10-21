"""
目标：
探索性数据分析，分析训练数据集，检查
1. 数据量
2. 检查各个标签的数据分布
3. 句子长度分布

步骤：
1. 实例化配置类
2. 数据加载
3. 探索性数据分析
    3.1 查看数据量
    3.2 检查标签分布
    3.3 计算句子长度分布
"""
from config import Config
import pandas as pd
from collections import Counter
import numpy as np

# 步骤：
# 1. 实例化配置类
config_obj = Config()
# 2. 数据加载
## csv - comma seperated values
## 参数1：文件路径，参数sep：分隔符
data = pd.read_csv(config_obj.training_data_path,sep='\t',names=['title', 'label'])

# 3. 探索性数据分析
#     3.1 查看数据量
# print(f"数据量：{len(data)}")
print(data.head(5))
print(f"数据量：{data.shape[0]}")

#     3.2 检查标签分布
## 思路1：使用dict
# label_dist = {}
# for label in data['label'].tolist():
#     label_dist[label] = label_dist.get(label, 0) + 1
# print(label_dist)
## 思路2：Counter
with open(config_obj.class_path, 'r', encoding='utf-8') as f:
    content = f.read()
    label_list = content.split('\n')

# print(label_list)
for label_ind,counts in Counter(data['label']).items():
    print(f"标签{label_list[label_ind]}出现了{counts}次")
#     3.3 计算句子长度分布

## 思路1: 循环计算
# len_list = []
# for title in data['title'].tolist():
#     len_list.append(len(title))
# len_list = np.array(len_list)
## 思路2：map内置函数
## 参数1：处理函数；参数2：可迭代的需要被处理的列表，返回map对象，元素是处理函数处理的结果
# len_list = np.array( list(map(lambda x:len(x), data['title'])))
## 思路3：使用dataframe apply
# len_list =  data['title'].apply(lambda x:len(x))
len_list = data['title'].str.len()
# print(type(lens))
# print(lens)
print(f"句子长度均值{len_list.mean()}")
print(f"句子长度最大值：{len_list.max()}")
print(f"句子长度最小值：{len_list.min()}")
print(f"句子长度标准差：{len_list.std()}")
