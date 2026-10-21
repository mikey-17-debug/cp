import json

from flask import request, Blueprint

from src.rf.rf_predict import predict as rf_p
from src.fasttext.fasttext_predict import ft_predict as fasttext_p
from src.bert.predict_genre import predict as bert_p
from src.llm.predict import predict as llm_p

predict_bp = Blueprint('predict', __name__, url_prefix='/api')
@predict_bp .route(rule='/rf/predict', methods=['GET', 'POST'])
def rf_predict_api():
    if request.method == 'GET':
        input_data = request.args.to_dict()
    else:
        input_data = request.get_json(silent=True)
    if not input_data or 'text' not in input_data:
        return json.dumps({"code": "FAIL", "result": "缺少text参数"}, ensure_ascii=False)
    return json.dumps({"code": "SUCCESS", "result": "SUCCESS", "data": rf_p(input_data)}, ensure_ascii=False)

@predict_bp .route(rule='/fasttext/predict', methods=['GET', 'POST'])
def fasttext_predict_api():
    if request.method == 'GET':
        input_data = request.args.to_dict()
    else:
        input_data = request.get_json(silent=True)
    if not input_data or 'text' not in input_data:
        return json.dumps({"code": "FAIL", "result": "缺少text参数"}, ensure_ascii=False)
    return json.dumps({"code": "SUCCESS", "result": "SUCCESS", "data": fasttext_p(input_data)}, ensure_ascii=False)

@predict_bp .route(rule='/bert/predict', methods=['GET', 'POST'])
def bert_predict_api():
    if request.method == 'GET':
        input_data = request.args.to_dict()
    else:
        input_data = request.get_json(silent=True)
    if not input_data or 'text' not in input_data:
        return json.dumps({"code": "FAIL", "result": "缺少text参数"}, ensure_ascii=False)
    return json.dumps({"code": "SUCCESS", "result": "SUCCESS", "data": bert_p(input_data)}, ensure_ascii=False)

@predict_bp .route(rule='/llm/predict', methods=['GET', 'POST'])
def llm_predict_api():
    if request.method == 'GET':
        input_data = request.args.to_dict()
    else:
        input_data = request.get_json(silent=True)
    if not input_data or 'text' not in input_data:
        return json.dumps({"code": "FAIL", "result": "缺少text参数"}, ensure_ascii=False)
    return json.dumps({"code": "SUCCESS", "result": "SUCCESS", "data": llm_p(input_data)}, ensure_ascii=False)