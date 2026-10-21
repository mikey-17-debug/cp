# cp
基于 chinese-bert-wwm 的中文电影类型分类（单标签 / 多标签）

## 环境初始化
- python 版本 3.11+（本项目实测于 `D:\conda3\envs\nlp_learn`：Python 3.12 + torch 2.6 + transformers 5.19）
- 在项目根目录执行以下命令：
```cmd
 pip install -r install/requirements.txt
```

## 项目结构
```
Bert_model/
├── config.yaml            # 唯一配置入口：路径、切分、模型、训练超参、任务参数
├── main.py                # 命令行入口（train / predict）
├── src/
│   ├── config.py          # YAML -> 强类型配置，相对路径统一按项目根目录解析
│   ├── data.py            # 读取、类型解析、标签空间、影片级切分、预分词 Dataset
│   ├── model.py           # 预训练权重 -> 序列分类模型（problem_type 决定损失）
│   ├── metrics.py         # 单标签 / 多标签两套指标
│   ├── trainer.py         # 训练编排、最优模型保存、指标落盘
│   ├── predictor.py       # 模型推理
│   └── utils.py           # 日志、随机种子、设备信息
├── data_clean.py          # 原始数据清洗脚本（生成 dataset/clean_data.txt）
└── dataset/clean_data.txt # 训练数据（唯一数据源）
```

## 两个任务
| 任务 | 标签构造 | 损失 | 关键指标 |
| --- | --- | --- | --- |
| `single` 单标签 | 一条影片的多个类型展开为多条「相同摘要 + 不同标签」样本 | 交叉熵 | accuracy / macro-F1 / top-k |
| `multi` 多标签 | 36 个类型上的 multi-hot 向量 | BCEWithLogits | micro/macro-F1 / 子集准确率 / 汉明损失 |

- 切分先在**影片级**完成（9:1:1），再展开单标签样本，避免同一摘要跨集合泄漏。
- 输入文本默认只用 `summary`（剧情简介），可在 `config.yaml` 的 `data.text_column` 修改。
- 出现次数低于 `data.min_label_count` 的类型自动剔除，避免低频噪声标签。

## 运行
在 `Bert_model` 目录下执行（`--config`、`--log-file` 等全局参数需写在子命令之前）：

```cmd
:: 训练单标签 + 多标签
python main.py train --task all

:: 只训练多标签
python main.py train --task multi

:: 临时覆盖超参（不改配置文件）
python main.py train --task single --epochs 2 --batch-size 16

:: 另开窗口训练并写日志
python main.py --log-file train.log train --task all

:: 推理
python main.py predict --task single --text "一名银行家蒙冤入狱，用二十年时间挖通墙壁越狱。"
python main.py predict --task multi  --text "一名银行家蒙冤入狱，用二十年时间挖通墙壁越狱。"
```

## 输出
模型与指标保存到 `config.yaml` 中 `training.output_root`（默认 `../../model`）：

```
model/
├── single_label/
│   ├── best/              # 最优模型 + 分词器 + labels.json
│   ├── labels.json        # 标签空间、阈值、最大长度
│   └── metrics.json       # 验证/测试指标 + 完整配置快照
└── multi_label/           # 同上
```

每次训练还会把实际使用的切分数据写回 `data.export_dir`（默认 `dataset/splits`），
与模型使用完全一致的样本和标签，便于复现与检查：

```
Bert_model/dataset/splits/
├── single_label/{train,val,test}.csv   # 列：summary, label
└── multi_label/{train,val,test}.csv    # 列：summary, labels（多个类型用 | 分隔）
```

关闭该行为：在 `config.yaml` 中设置 `data.export_splits: false`。

## 逐条预测结果
训练结束与压缩实验都会把**测试集每一条样本**的预测结果落盘，便于逐例核对与误差分析：

```cmd
:: 为最优模型与全部压缩变体导出（默认 test 切分）
python main.py predict-set --task all

:: 只导出验证集
python main.py predict-set --task single --split val
```

每个模型目录下生成：

```
model/<task>/best/test_predictions.csv
model/<task>/best/test_probabilities.npy
model/<task>/optimized/<variant>/test_predictions.csv
model/<task>/optimized/<variant>/test_probabilities.npy
```

- `test_predictions.csv`
  - 单标签列：`row, summary, true_label, pred_label, confidence, top_k, correct`
  - 多标签列：`row, summary, true_labels, pred_labels, max_probability, num_pred, subset_correct`
- `test_probabilities.npy`：形状 `(样本数, 类别数)`，单标签为 softmax、多标签为 sigmoid，可直接用于调阈值。
- `row` 与 `dataset/splits/<task>/test.csv` 的行号一一对应。

已为 2 个最优模型与 20 个压缩变体生成预测文件，并用保存的概率矩阵复算指标，
与 `metrics.json` / `optimization_results.json` 中的记录一致（浮点显示位数差异除外）。

> 注意：动态 INT8 在 CPU 上的数值与批大小相关（不同批形状会走不同底层算子）。
> 若需复现实验中的指标，请用实验时的批大小导出：`--batch-size 64`。
> 其余变体与基线对批大小不敏感。

首次补齐已有模型：已用上述命令为 2 个最优模型与 20 个压缩变体生成预测文件。

## 模型压缩实验
对已训练好的两个模型分别做多种精度量化、剪枝与低秩分解，并输出对比报告：

```cmd
:: 两个任务全部执行（量化 + 剪枝 + 低秩）
python run_optimization.py --task all

:: 只做单标签模型
python run_optimization.py --task single
```

- 变体与比例全部在 `config.yaml` 的 `optimization` 段配置；
- 压缩模型保存到 `model/<task>/optimized/<variant>/`；
- 结果写入 `model/optimization_results.json`，报告写入项目根目录 `model_optimization_report.md`；
- 报告中记录参数量、文件体积、CPU/GPU 单条延迟与吞吐、以及对应任务的精度指标。

## 性能说明
- 全部文本在训练前一次性批量分词；训练循环内只做张量切片，避免 GPU 空等 CPU。
- bf16 + 融合 AdamW（`training.optim`）+ batch 32，RTX 4070 Laptop 实测约 105 samples/s，GPU 利用率约 98%。
- Windows 下 `dataloader_num_workers>0` 反而更慢（进程启动开销），默认保持 0。
