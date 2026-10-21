import os
import time
import random
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from openai import OpenAI
import json

from src.config import LLMConfig

config = LLMConfig()
GENRES = config.get_genres()
GENRE_SET = set(GENRES)

# 开启 JSON 模式（DeepSeek 支持）；若接口/代理不支持会自动降级重试
USE_JSON_MODE = True

# 模型偶尔会输出英文类别名，做一次归一化兜底
EN2ZH = {
    'comedy': '喜剧', 'action': '动作', 'horror': '恐怖', 'thriller': '恐怖',
    'romance': '爱情', 'mystery': '悬疑', 'animation': '动画', 'anime': '动画',
    'sci-fi': '科幻', 'science fiction': '科幻', 'music': '音乐', 'musical': '音乐',
    'crime': '犯罪', 'documentary': '纪录片',
}


def _chat(client, messages, use_json):
    """统一封装一次请求；use_json 控制是否启用 response_format"""
    kwargs = dict(
        model=config.model,
        messages=messages,
        stream=False,
        temperature=0,          # 分类任务用 0，输出稳定
    )
    if use_json:
        kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**kwargs)


def predict(data):
    """
    预测函数封装，给定输入字典（包含text字段），返回
    {
    "text",
    "pred_class", # 预测结果（电影类型中文名）
    "duration"   # 耗时
    }
    """
    t1 = time.time()

    # 假设传入的 text 已经包含了片名和简介
    text = data["text"]
    # 超长简介截断，聚焦主线剧情，同时节省 token
    if len(text) > 500:
        text = text[:500]

    # 把 10个类型拼进系统提示词
    genre_list = "、".join(GENRES)
    messages = [
        {"role": "system", "content": f"""你是电影类型分类专家。输入是一段中文剧情简介，可能夹杂宣传语、残缺句子或外语片段，请从下面 {len(GENRES)} 个类型中选出最匹配的一个。

可选类型：{genre_list}

类型定义：
- 喜剧：以搞笑、幽默、误会、荒诞桥段为主要看点，目的就是逗乐观众
- 动作：以打斗、枪战、追逐、战争、功夫、特工行动等动作场面为核心
- 恐怖：以鬼怪、僵尸、灵异、诅咒、凶宅、血腥杀戮等制造惊吓
- 爱情：以男女主人公的相识、相爱、感情纠葛为主线
- 悬疑：以案件、谜团、推理、追查真相为主线，靠悬念推进
- 动画：动画、动漫、卡通作品（含剧场版、配音动画），不论题材
- 科幻：以外星、太空、未来世界、机器人、人工智能、克隆、时空穿越等设定为核心
- 音乐：以音乐、歌手、乐队、演唱、选秀、音乐梦想为核心内容
- 犯罪：以黑帮、贩毒、抢劫、绑架、诈骗、越狱、警匪对抗等犯罪行为及侦破为主线
- 纪录片：非虚构，真实记录现实人物、事件、历史、自然或演出实况

判定步骤（按顺序执行）：
1. 先看形式：是动画作品 → 动画；是非虚构的真实记录（含跟拍演唱会、巡演、人物纪实） → 纪录片。
2. 再看内容主线：以歌手/乐队/音乐梦想为核心 → 音乐；以恋爱关系为核心 → 爱情；以搞笑为核心 → 喜剧。
3. 最后裁决题材：有鬼神灵异 → 恐怖；以案件推理谜团为主 → 悬疑；以犯罪集团或警匪对抗为主 → 犯罪；以动作/战争场面为主 → 动作；以科幻设定为主 → 科幻。

冲突裁决规则：
- "虚构的歌手/乐队追梦故事" → 音乐；"跟拍真实歌手乐队的巡演、演出实况" → 纪录片
- 动画片即使是科幻或恐怖题材，也选动画
- 警匪题材即使有感情线也选犯罪，不选爱情
- 主线是犯罪侦破选犯罪；主线是打斗动作场面选动作
- 有鬼怪灵异选恐怖；纯逻辑推理、无鬼神选悬疑
- 简介里大量与剧情无关的宣传文字时，只依据剧情部分判断

输出要求：
1. 只输出一个 JSON 对象，形如 {{"genre": "类型名"}}
2. value 必须是上述 10 个中文类型名之一，不能输出英文、不能输出多个、不要解释
3. 不要输出 Markdown 标记（如 ```json）

示例：
简介：社区里爱管闲事的邻家大叔把街坊们的琐事全搅成了连串乌龙，最后反倒促成几段姻缘。 → {{"genre": "喜剧"}}
简介：退役特种兵为营救被绑架的女儿独自闯入黑帮据点，一路枪战肉搏、以一敌百。 → {{"genre": "动作"}}
简介：搬进老宅的一家人每到午夜都会听见女人的哭声，随后接连遭遇无法解释的怪异事件。 → {{"genre": "恐怖"}}
简介：两个性格迥异的年轻人从互相看不顺眼到彼此依靠，经历误会与分离后终于走到一起。 → {{"genre": "爱情"}}
简介：密闭书房里出现一具尸体，每位嫌疑人都有不在场证明，侦探在细节中揪出真凶。 → {{"genre": "悬疑"}}
简介：小男孩与森林中的精灵结为伙伴，一起守护即将被砍伐的家园。 → {{"genre": "动画"}}
简介：人类被人工智能接管后，一群幸存者潜入主控核心，试图关闭掌控全球的超级系统。 → {{"genre": "科幻"}}
简介：小镇女孩凭独特嗓音闯入纽约爵士乐坛，从驻唱歌手一路成为当红女星。 → {{"genre": "音乐"}}
简介：两名偷车贼被黑帮胁迫参与运毒交易，警方卧底逐步收网，各方在雨夜展开生死博弈。 → {{"genre": "犯罪"}}
简介：镜头跟随一支乐团完成世界巡演，记录他们在各地的演出实况与后台生活片段。 → {{"genre": "纪录片"}}
"""},
        {
            "role": "user",
            "content": text,
        }
    ]

    client = OpenAI(api_key=os.getenv(config.api_key_env), base_url=config.base_url)

    # 增加重试机制
    max_retries = 3
    use_json = USE_JSON_MODE
    pred_genre = "未知类别"
    for attempt in range(max_retries):
        try:
            try:
                response = _chat(client, messages, use_json)
            except Exception as e:
                # 接口/代理不支持 JSON 模式时自动降级，避免整批预测失败
                if use_json:
                    print(f"\n[提示] JSON 模式不可用，已降级为普通模式：{e}")
                    use_json = False
                    response = _chat(client, messages, False)
                else:
                    raise

            result_text = response.choices[0].message.content.strip()

            # 清理可能存在的 markdown 标记
            result_text = result_text.replace("```json", "").replace("```", "").strip()

            # 解析 JSON
            result_json = json.loads(result_text)
            pred_genre = str(result_json.get("genre") or "").strip()

            # 输出归一化：英文类别名 -> 中文
            if pred_genre not in GENRE_SET and pred_genre.lower() in EN2ZH:
                pred_genre = EN2ZH[pred_genre.lower()]
            # 仍不合法时做一次包含匹配（如 "科幻片" -> "科幻"）
            if pred_genre not in GENRE_SET:
                for g in GENRES:
                    if g in pred_genre:
                        pred_genre = g
                        break

            # 只有拿到合法类别才结束，否则继续重试
            if pred_genre in GENRE_SET:
                break

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n[警告] 重试 {max_retries} 次后仍失败，错误信息: {e}")
                pred_genre = "未知类别"
            time.sleep(1)  # 失败后等待1秒重试

    return {
        "text": text,
        "pred_class": pred_genre if pred_genre in GENRE_SET else "未知类别",
        'duration': int((time.time() - t1) * 1000)
    }


def evaluate(data_path, predict_fun):
    """
    评估大模型的 1. 准确率；2. 速度
    """
    # 数据集实际为两列：title \t label（title 中已含片名+简介全文）
    # quoting=3 (QUOTE_NONE)：关闭引号解析，避免标题中的引号吞并多行导致 label 串列
    data = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=["title", "label"],
        quoting=3,
        engine="python",
    )
    # 兼容可能存在的三列格式 title \t 简介 \t label
    first_line = ""
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                first_line = line.rstrip("\r\n")
                break
    if first_line.count("\t") >= 2:
        data = pd.read_csv(
            data_path, sep="\t", header=None,
            names=["title", "text_content", "label"], quoting=3, engine="python",
        )
        data["input_text"] = data["title"].astype(str) + "。" + data["text_content"].astype(str)
    else:
        data["input_text"] = data["title"].astype(str)

    labels = data["label"].tolist()
    input_texts = data["input_text"].tolist()

    # 中文类型名 -> 数字标签 id
    zh_ind_map = {name: i for i, name in enumerate(GENRES)}
    zh_ind_map["未知类别"] = -1

    # 打印类别映射表，方便核对标签是否正确
    print("\n===== 类别映射核对 =====")
    for name, idx in zh_ind_map.items():
        print(f"{idx} -> {name}")
    print("========================\n")

    total_time = 0
    pred_list = []
    label_list = []

    num_test_data = 100
    # 如果数据量大于 200，使用 random.sample 随机抽样，避免分布不均
    if len(input_texts) > num_test_data:
        sample_indices = random.sample(range(len(input_texts)), num_test_data)
    else:
        sample_indices = list(range(len(input_texts)))

    for i in tqdm(sample_indices, desc="LLM模型测试"):
        text = input_texts[i]
        label = labels[i]

        try:
            result = predict_fun({"text": text})
            total_time += result["duration"]

            # 将预测的中文类别转回数字 id
            pred_id = zh_ind_map.get(result["pred_class"], -1)
            pred_list.append(pred_id)
            label_list.append(label)
        except Exception as e:
            print(f"\n[警告] 预测第 {i} 条数据失败，已跳过。错误信息: {e}")

    # 3. 计算准确率，计算速度
    if len(label_list) > 0:
        acc = accuracy_score(label_list, pred_list)
        print(f"\n✅ 总测试数据量: {len(label_list)} 条")
        print(f"✅ 准确率: {acc:.4f}")
        print(f"✅ 平均速度: {total_time / len(label_list):.2f} 秒/条")




if __name__ == "__main__":
    TEST_DATA_PATH = "../01-data/test.txt"
    if os.path.exists(TEST_DATA_PATH):
        evaluate(TEST_DATA_PATH, predict)
    else:
        print(f"❌ 错误：找不到数据文件 '{TEST_DATA_PATH}'，请修改代码最下方的 TEST_DATA_PATH 路径！")
        print(f"当前运行目录下只有: {os.listdir('.')}")