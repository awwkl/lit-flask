from flask import Flask, request, jsonify
from predict import model_predict

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, edited this is lit-flask@@@!'

@app.route('/predict_text', methods=['POST'])
def predict_text():
    text = request.args.get('text')
    category_list = model_predict(text)
    return str(category_list)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, threaded=True, debug=True)
