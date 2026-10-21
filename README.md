# cp

中英文电影类型分类（单标签 / 多标签）+ 训练后压缩实验

- 中文一期：`chinese-bert-wwm` + `config.yaml`（36 类）
- 中文二期：`chinese-bert-wwm` + `config_zh_balance.yaml`（10 类均衡，单标签）
- 英文：`bert-base-uncased` + `config_imdb.yaml`（IMDb-Media）

三套数据共用同一套训练/推理/压缩代码，通过配置文件切换，互不覆盖产物。

## 环境初始化

- python 版本 3.11+（实测于 `D:\conda3\envs\nlp_learn`：Python 3.12 + torch 2.6 + transformers 5.19）
- 在项目根目录执行：

```cmd
pip install -r install/requirements.txt
```

## 项目结构

```
Bert_model/
├── config.yaml            # 中文一期（36 类）
├── config_zh_balance.yaml # 中文二期（10 类均衡，单标签）
├── config_imdb.yaml       # 英文 IMDb 配置（独立数据源与输出目录）
├── main.py                # 命令行入口（train / predict / predict-set）
├── predict_genre.py       # 独立预测脚本（只改模型路径即可用）
├── api_server.py          # HTTP API（Flask，支持 API Key 与 CORS）
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
    ├── clean_data.txt       # 中文一期训练数据
    ├── zh_balance_splits/   # 中文二期切分导出（运行后生成）
    └── imdb_splits/         # 英文切分导出（运行后生成，默认不入库）
```

## 数据与任务

| 项目 | 中文一期（`config.yaml`） | 中文二期（`config_zh_balance.yaml`） | 英文 IMDb（`config_imdb.yaml`） |
| --- | --- | --- | --- |
| 数据源 | `dataset/clean_data.txt` | `../original_file/data_balance.tsv` | `../../T1/imdb_media_dataset`（HF Arrow） |
| 输入文本 | `summary` | `summary` | `{title}. {storyline}`，回退 `presentation` / `title` |
| 标签字段 | `label` | `genre` | `genres`（多值） |
| 标签空间 | 36 类 | 10 类（每类 10,000 条） | 21 个题材类（剔除 7 个格式类） |
| 单标签口径 | 多类型展开为多条样本 | `primary`（每行一个类型） | `primary`（genres 首项） |
| 多标签阈值 | 固定 0.5 | 不训练多标签 | 验证集逐类校准 |
| 切分 | 影片级 9:1:1 | 分层 8:1:1 | 影片级 9:1:1 |

- `single`：交叉熵，指标为 accuracy / macro-F1 / top-k；
- `multi`：BCEWithLogits，指标为 micro/macro-F1 / 子集准确率 / 汉明损失；
- 中文二期会剔除同一摘要对应不同类别的冲突样本，并按 `data.separator: "\t"` 读取 TSV；
- 切分先在**影片级**完成，避免同一摘要跨集合泄漏；低频标签按 `data.min_label_count` 剔除。

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

### 中文二期（10 类均衡）

```cmd
:: 只训练单标签（12 epoch 上限 + patience=3 早停）
python main.py --config config_zh_balance.yaml train --task single

:: 从最新 checkpoint 续训
python main.py --config config_zh_balance.yaml train --task single --resume

:: 指定检查点续训
python main.py --config config_zh_balance.yaml train --task single --resume "..\model_zh_balance\single_label\checkpoint-39864"

:: 推理
python main.py --config config_zh_balance.yaml predict --task single --text "一名银行家蒙冤入狱，用二十年时间挖通墙壁越狱。"
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

中文一期保存到 `../../model`，中文二期保存到 `../../model_zh_balance`，英文保存到 `../../model_imdb`：

```
model/ 或 model_zh_balance/ 或 model_imdb/
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
Bert_model/dataset/splits/             # 中文一期
Bert_model/dataset/zh_balance_splits/  # 中文二期
Bert_model/dataset/imdb_splits/        # 英文
├── single_label/{train,val,test}.csv   # 列：summary, label
└── multi_label/{train,val,test}.csv    # 列：summary, labels（多个类型用 | 分隔）
```

关闭该行为：在对应配置中设置 `data.export_splits: false`。英文两个 `train.csv` 超过 GitHub 单文件 100 MB 上限，仓库中以同目录的 `train.csv.gz` 归档（解压即得原文件）；压缩包与校验值同时保留在 `imdb_deliverables/large_files/`。

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

## 独立预测与 HTTP API

### 独立预测脚本

`predict_genre.py` 只依赖 `torch` 与 `transformers`，导入时直接加载模型，改顶部 `MODEL_DIR` 即可切换模型：

```python
from predict_genre import predict

result = predict({"text": "一名银行家蒙冤入狱，用二十年时间挖通墙壁越狱。"})
# {"text": ..., "pred_class": "犯罪", "score": 0.9087, "top_k": [...], "duration": 45}
```

命令行与 CPU 模式：

```cmd
python predict_genre.py --text "剧情简介"

:: 强制 CPU
set GENRE_DEVICE=cpu
python predict_genre.py --text "剧情简介"
```

### HTTP API

```cmd
python api_server.py --config config_zh_balance.yaml --task single --host 0.0.0.0 --port 8000 --api-key <key>
```

- `GET /health`：模型、标签、设备状态
- `POST /predict`：`{"texts": ["..."], "top_k": 3}`
- 请求头 `X-API-Key` 或 `Authorization: Bearer <key>`；已内置 CORS
- 需要公网访问时可用 `cloudflared tunnel --url http://127.0.0.1:8000` 生成临时 HTTPS 地址

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

### 中文二期（10 类均衡，chinese-bert-wwm，12 epoch）

| 任务 | 测试集指标 |
| --- | --- |
| 单标签 | Acc 0.8887、Macro-F1 0.8875、Top-3 0.9742（验证集 Acc 0.8947 / Macro-F1 0.8943） |

- 训练数据：99,658 条（10 类均衡，剔除重复与跨类冲突），train 79,726 / val 9,966 / test 9,966
- 配置：`max_length=384`、batch 16、lr 2e-5、bf16、12 epoch 上限 + patience=3；12 轮均有小幅提升，未触发早停
- 从 checkpoint-39864（epoch 8）续训到 epoch 12，测试 Acc 0.8819 → 0.8887、Macro-F1 0.8810 → 0.8875

### 英文 IMDb（bert-base-uncased，2 epoch，逐类阈值）

| 任务 | 测试集指标 |
| --- | --- |
| 单标签 | Acc 0.5413、Macro-F1 0.3590、Top-3 0.8289（多数类基线 0.2190） |
| 多标签 | Micro-F1 0.6180、Macro-F1 0.5878、子集准确率 0.2118、汉明损失 0.0795 |

压缩要点：FP16/BF16 体积 209.6 MB（50.1%）、GPU 延迟 4.98~5.51 ms，精度无损；INT8 体积 173.8 MB（41.5%）、CPU 吞吐提升 1.24~1.31×，主指标下降 3.7~4.3 个百分点；结构化剪枝与低秩分解未微调时精度损失较大，需配合微调后再部署。完整数据见 [model_imdb_optimization_report.md](model_imdb_optimization_report.md)。

### 中文一期（36 类，chinese-bert-wwm，3 epoch）

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
- 中文二期 10 类均衡数据、seq 384 / batch 16，12 epoch 实测约 5.5 小时；可用 `--resume` 从检查点续训。
