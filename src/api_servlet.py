
from flask import  send_file

from app_factory import app
from api_predict import predict_bp

# 注册其他模块文件里的接口
app.register_blueprint(predict_bp)
@app.route(rule='/', methods=['GET'])
def index():
    return send_file('../index.html')

@app.route(rule='/predict', methods=['GET'])
def predict_web():
    return send_file('./predict.html')

if __name__ == '__main__':
    print("running app routes:", [str(r) for r in app.url_map.iter_rules()])
    app.run(debug=True, host='0.0.0.0', port=80)
