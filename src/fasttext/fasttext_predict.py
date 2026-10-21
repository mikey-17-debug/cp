import time

import fasttext
import jieba

from src.config import FasttextConfig
from src.fasttext.fasttext_data_load import get_save_model_path

config_obj = FasttextConfig()
split_mode = 'jieba'
autotune = True
model_dir = get_save_model_path(autotune, split_mode)
fasttext_model = fasttext.load_model(model_dir)

def ft_predict(data:dict):
    start_time = time.time()
    text = data["text"]
    if split_mode == 'char':
        result = fasttext_model.predict(" ".join(list(text)))
    else:
        result = fasttext_model.predict(" ".join(jieba.lcut(text)[:128]))
    return {
        'text': text,
        'pred_class': result[0][0][9:],
        'duration': int((time.time() - start_time)*1000)
    }

if __name__ == '__main__':
    data = {'text':'普通中年男子程勇（徐峥饰）经营着一家保健品店，失意又失婚。不速之客吕受益（王传君饰）的到来，让他开辟了一条去印度买药做“代购”的新事业，虽然困难重重，但他在这条“买药之路”上发现了商机，一发不可收拾地做起了治疗慢粒白血病的印度仿制药独家代理商。赚钱的同时，他也认识了几个病患及家属，为救女儿被迫做舞女的思慧（谭卓饰）、说一口流利“神父腔”英语的刘牧师（杨新鸣饰），以及脾气暴烈的“黄毛”（章宇饰），几个人合伙做起了生意，利润倍增的同时也危机四伏。程勇昔日的小舅子曹警官（周一围饰）奉命调查仿制药的源头，假药贩子张长林（王砚辉饰）和瑞士正牌医药代表（李乃文饰）也对其虎视眈眈，生意逐渐变成了一场关于救赎的拉锯战。本片改编自慢粒白血病患者陆勇代购抗癌药的真实事迹'}
    print(ft_predict(data))
    print(ft_predict(data))
    print(ft_predict(data))
    print(ft_predict(data))
