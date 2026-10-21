# cp

中英文电影类型分类（单标签 / 多标签）+ 训练后压缩实验

- 中文：`chinese-bert-wwm` + `config.yaml`
- 英文：`bert-base-uncased` + `config_imdb.yaml`（IMDb-Media）

两套数据共用同一套训练/推理/压缩代码，通过配置文件切换，互不覆盖产物。

## 环境初始化

- python 版本 3.11+（实测于 `D:\conda3\envs\nlp_learn`：Python 3.12 + torch 2.6 + transformers 5.19）
- 在项目根目录执行：

```cmd
pip install -r install/requirements.txt
```

## 项目结构

```
Bert_model/
├── config.yaml            # 中文配置：路径、切分、模型、训练超参、任务参数
├── config_imdb.yaml       # 英文 IMDb 配置（独立数据源与输出目录）
├── main.py                # 命令行入口（train / predict / predict-set）
├── run_optimization.py    # 压缩实验入口（量化 / 剪枝 / 低秩分解）
├── src/
│   ├── config.py          # YAML -> 强类型配置，相对路径统一按项目根目录解析
│   ├── data.py            # 文本 / Arrow 数据读取、标签解析、影片级切分、预分词 Dataset
│   ├── model.py           # 预训练权重 -> 序列分类模型（problem_type 决定损失）
│   ├── metrics.py         # 单标签 / 多标签两套指标
│   ├── trainer.py         # 训练编排、最优模型保存、指标落盘
│   ├── predictor.py       # 模型推理
│   ├── predictions.py     # 逐条预测与概率矩阵导出
│   ├── optimization.py    # 半精度 / INT8 / 剪枝 / 低秩分解及其保存与加载
│   ├── benchmark.py       # 精度评估、延迟与吞吐测量、目录体积统计
│   ├── report.py          # 压缩实验 Markdown 报告渲染
│   └── utils.py           # 日志、随机种子、设备信息
├── data_clean.py          # 中文原始数据清洗脚本（生成 dataset/clean_data.txt）
└── dataset/
    ├── clean_data.txt     # 中文训练数据
    └── imdb_splits/       # 英文切分导出（运行后生成，默认不入库）
```

## 数据与任务

| 项目 | 中文（`config.yaml`） | 英文 IMDb（`config_imdb.yaml`） |
| --- | --- | --- |
| 数据源 | `dataset/clean_data.txt`（分隔符文本） | `../../T1/imdb_media_dataset`（HF Arrow） |
| 输入文本 | `summary` | `{title}. {storyline}`，缺失时回退 `presentation` / `title` |
| 标签字段 | `label` | `genres`（多值） |
| 标签空间 | 36 类 | 21 个题材类（剔除 7 个格式类，丢弃低频 <10 的异体标签） |
| 单标签口径 | 多类型展开为多条「同摘要 + 不同标签」样本 | `primary`（genres 首项） |
| 多标签阈值 | 固定 0.5 | 验证集逐类校准（写入 `labels.json`） |
| 切分 | 影片级 9:1:1（seed=42） | 影片级 9:1:1（seed=42） |

- `single`：交叉熵，指标为 accuracy / macro-F1 / top-k；
- `multi`：BCEWithLogits，指标为 micro/macro-F1 / 子集准确率 / 汉明损失；
- 切分先在**影片级**完成，再展开单标签样本，避免同一摘要跨集合泄漏；
- 出现次数低于 `data.min_label_count` 的类型自动剔除，避免低频噪声标签。

## 运行

在 `Bert_model` 目录下执行（`--config`、`--log-file` 等全局参数需写在子命令之前）。

### 中文

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

### 英文 IMDb

```cmd
:: 训练单标签 + 多标签（bert-base-uncased，2 epoch）
python main.py --config config_imdb.yaml train --task all

:: 只训练多标签（验证集逐类阈值校准）
python main.py --config config_imdb.yaml train --task multi

:: 推理（英文剧情简介）
python main.py --config config_imdb.yaml predict --task single --text "A banker wrongly imprisoned escapes through a tunnel he dug over twenty years."
python main.py --config config_imdb.yaml predict --task multi  --text "A banker wrongly imprisoned escapes through a tunnel he dug over twenty years."

:: 后台运行并写日志
python main.py --config config_imdb.yaml --log-file imdb_train.log train --task all
```

## 输出

中文模型保存到 `config.yaml` 的 `training.output_root`（默认 `../../model`），英文保存到 `config_imdb.yaml` 的 `../../model_imdb`：

```
model/ 或 model_imdb/
├── single_label/
│   ├── best/              # 最优模型 + 分词器 + labels.json
│   ├── labels.json        # 标签空间、阈值、最大长度
│   ├── metrics.json       # 验证/测试指标 + 完整配置快照
│   ├── test_predictions.csv
│   ├── test_probabilities.npy
│   └── optimized/<variant>/   # 压缩变体及其逐条预测
└── multi_label/           # 同上
```

训练结束后会把实际使用的切分写回 `data.export_dir`：

```
Bert_model/dataset/splits/       # 中文
Bert_model/dataset/imdb_splits/  # 英文
├── single_label/{train,val,test}.csv   # 列：summary, label
└── multi_label/{train,val,test}.csv    # 列：summary, labels（多个类型用 | 分隔）
```

关闭该行为：在对应配置中设置 `data.export_splits: false`。两个目录体积较大（英文 `train.csv` 超过 100 MB），默认不纳入 Git。

## 逐条预测结果

训练结束与压缩实验都会把**测试集每一条样本**的预测结果落盘，便于逐例核对与误差分析：

```cmd
:: 为最优模型与全部压缩变体导出（默认 test 切分）
python main.py predict-set --task all
python main.py --config config_imdb.yaml predict-set --task all

:: 只导出验证集
python main.py predict-set --task single --split val
```

每个模型目录下生成：

```
<output_root>/<task>/best/test_predictions.csv
<output_root>/<task>/best/test_probabilities.npy
<output_root>/<task>/optimized/<variant>/test_predictions.csv
<output_root>/<task>/optimized/<variant>/test_probabilities.npy
```

- `test_predictions.csv`
  - 单标签列：`row, summary, true_label, pred_label, confidence, top_k, correct`
  - 多标签列：`row, summary, true_labels, pred_labels, max_probability, num_pred, subset_correct`
- `test_probabilities.npy`：形状 `(样本数, 类别数)`，单标签为 softmax、多标签为 sigmoid，可直接用于调阈值；
- `row` 与 `dataset/{splits,imdb_splits}/<task>/test.csv` 的行号一一对应。

> 注意：动态 INT8 在 CPU 上的数值与批大小相关（不同批形状会走不同底层算子）。
> 若需复现实验中的指标，请用实验时的批大小导出：`--batch-size 64`。
> 其余变体与基线对批大小不敏感。

## 模型压缩实验

对已训练好的两个 best 模型分别做 FP16/BF16、动态 INT8、非结构化剪枝（30%/50%）、结构化剪枝（保留 75%/50%）与低秩分解（75%/50%/25%），共 10 个变体/任务：

```cmd
:: 中文
python run_optimization.py --task all

:: 英文 IMDb
python run_optimization.py --config config_imdb.yaml --task all

:: 只做单标签模型
python run_optimization.py --config config_imdb.yaml --task single
```

- 变体与比例在配置文件的 `optimization` 段调整；
- 压缩模型保存到 `<output_root>/<task>/optimized/<variant>/`；
- 结果写入 `<output_root>/optimization_results.json`，报告写入 `../model_optimization_report.md` / `../model_imdb_optimization_report.md`；
- 报告记录参数量、体积、CPU/GPU 单条延迟与吞吐、精度指标，以及各方法的结论。

## 实验结果

### 英文 IMDb（bert-base-uncased，2 epoch，逐类阈值）

| 任务 | 测试集指标 |
| --- | --- |
| 单标签 | Acc 0.5413、Macro-F1 0.3590、Top-3 0.8289（多数类基线 0.2190） |
| 多标签 | Micro-F1 0.6180、Macro-F1 0.5878、子集准确率 0.2118、汉明损失 0.0795 |

压缩要点：FP16/BF16 体积 209.6 MB（50.1%）、GPU 延迟 4.98~5.51 ms，精度无损；INT8 体积 173.8 MB（41.5%）、CPU 吞吐提升 1.24~1.31×，主指标下降 3.7~4.3 个百分点；结构化剪枝与低秩分解未微调时精度损失较大，需配合微调后再部署。完整数据见 [model_imdb_optimization_report.md](model_imdb_optimization_report.md)。

### 中文（chinese-bert-wwm，3 epoch）

| 任务 | 测试集指标 |
| --- | --- |
| 单标签 | Acc 0.3248、Macro-F1 0.1931、Top-3 0.6990（多数类基线 0.1770） |
| 多标签 | Micro-F1 0.5777、Macro-F1 0.2887、子集准确率 0.1828、汉明损失 0.0474（阈值 0.5） |

压缩要点：FP16/BF16 体积约 195.6 MB，GPU 延迟下降且精度无损；INT8 体积 146.0 MB（37.4%）；剪枝与低秩分解在不微调时精度下降明显。完整数据见 [model_optimization_report.md](model_optimization_report.md)。

## 交付产物

英文二期的报告、两期最优模型的逐条预测、项目总结与压缩后的数据集归档在：

```
Bert_model/imdb_deliverables/
├── report/model_imdb_optimization_report.md
├── predictions/{single_label,multi_label}/           # 英文
├── predictions/chinese/{single_label,multi_label}/   # 中文
├── large_files/*.zip                                 # 超 100MB 训练集的压缩包
└── IMDb项目总结与可行性分析.md
```

其中 `IMDb项目总结与可行性分析.md` 对两期数据、指标、压缩结果、可行性与项目不足做了完整对比。

## 性能说明

- 全部文本在训练前一次性批量分词；训练循环内只做张量切片，避免 GPU 空等 CPU。
- bf16 + 融合 AdamW（`training.optim`）+ batch 32，RTX 4070 Laptop 实测约 105 samples/s，GPU 利用率约 98%。
- Windows 下 `dataloader_num_workers>0` 反而更慢（进程启动开销），默认保持 0。
- 英文数据量大（约 24 万部影片），2 epoch 训练约 3 小时；压缩实验含 CPU 延迟测量约 3 小时 20 分钟。
