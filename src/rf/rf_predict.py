"""
目标：导入保存的模型，做预测

步骤：
1. 导入模型
2. 预测
"""
import pickle
import jieba
import time

from src.config import RFConfig

# 步骤：
# 1. 导入模型
config = RFConfig()
with open(config.embedding_path_rf, "rb") as f:
    tfidf_vectorizer = pickle.load(f)
with open(config.model_path_rf, "rb") as f:
    model = pickle.load(f)
    model.verbose = 0
def predict(data):
    """
    预测函数封装，给定输入字典（包含text字段），返回
    {
    "text",
    "pred_class", # 预测结果
    "duration # 耗时
    }
    """
    t1 = time.time()

    # 2. 预测
    title = data["text"]
    # 向量化
    vector = tfidf_vectorizer.transform([' '.join(jieba.lcut(title)[:300])])
    # 预测
    result = model.predict(vector)[0]


    # 导入标签定义
    # with open(config.class_data_path,'r') as f:
    #     ind2label = f.read().split('\n')
    # category_map = {
    #     "finance": "财经",
    #     "realty": "房产",
    #     "stocks": "股票",
    #     "education": "教育",
    #     "science": "科技",
    #     "society": "社会",
    #     "politics": "政治",
    #     "sports": "体育",
    #     "game": "游戏",
    #     "entertainment": "娱乐"
    # }

    return {
        # "text": data['text'],
        # "pred_class": category_map[ind2label[result]],
        # "duration": time.time() - t1
        "text": data['text'],
        "pred_class":result,
        'duration': int((time.time() - t1)*1000)
    }

if __name__ == '__main__':
    text_list = ["身陷囹圄的李问（郭富城饰）被引渡回港，他原本隶属于一个的跨国假钞制贩组织。该组织曾犯下过多宗恶性案件，而首脑“画家”不仅始终逍遥法外，连真面目都没人知道。为了逼迫李问吐露“画家”真实身份，警方不惜用足以判死刑的假罪证使其就范。就在此时，富有的遗孀阮文（张静初饰）申请保释李问，而警方提出的条件依然是“画家”的真面目。原来早在十数年前，李问和阮文是一对画家情侣，无奈女友的作品受人青睐，李问的画作却被贬得一文不值。就在此困顿期间，他制作假画的才能被“画家”（周润发饰）发掘，进而成为对方美元假币团伙中的一员……",
                 "普通中年男子程勇（徐峥饰）经营着一家保健品店，失意又失婚。不速之客吕受益（王传君饰）的到来，让他开辟了一条去印度买药做“代购”的新事业，虽然困难重重，但他在这条“买药之路”上发现了商机，一发不可收拾地做起了治疗慢粒白血病的印度仿制药独家代理商。赚钱的同时，他也认识了几个病患及家属，为救女儿被迫做舞女的思慧（谭卓饰）、说一口流利“神父腔”英语的刘牧师（杨新鸣饰），以及脾气暴烈的“黄毛”（章宇饰），几个人合伙做起了生意，利润倍增的同时也危机四伏。程勇昔日的小舅子曹警官（周一围饰）奉命调查仿制药的源头，假药贩子张长林（王砚辉饰）和瑞士正牌医药代表（李乃文饰）也对其虎视眈眈，生意逐渐变成了一场关于救赎的拉锯战。本片改编自慢粒白血病患者陆勇代购抗癌药的真实事迹。",
                "20世纪40年代末，小有成就的青年银行家安迪（蒂姆·罗宾斯TimRobbins饰）因涉嫌杀害妻子及她的情人而锒铛入狱。在这座名为肖申克的监狱内，希望似乎虚无缥缈，终身监禁的惩罚无疑注定了安迪接下来灰暗绝望的人生。未过多久，安迪尝试接近囚犯中颇有声望的瑞德（摩根·弗里曼MorganFreeman饰），请求对方帮自己搞来小锤子。以此为契机，二人逐渐熟稔，安迪也仿佛在鱼龙混杂、罪恶横生、黑白混淆的牢狱中找到属于自己的求生之道。他利用自身的专业知识，帮助监狱管理层逃税、洗黑钱，同时凭借与瑞德的交往在犯人中间也渐渐受到礼遇。表面看来，他已如瑞德那样对那堵高墙从憎恨转变为处之泰然，但是对自由的渴望仍促使他朝着心中的希望和目标前进。而关于其罪行的真相，似乎更使这一切朝前推进了一步……本片根据著名作家斯蒂芬·金（StephenEdwinKing）的原著改编。"]
    for text in text_list:
        result = predict({"text":text})
        print(f"{text}的结果是{result}")