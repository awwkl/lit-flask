from flask import Flask, request, jsonify
from predict import model_predict

app = Flask(__name__)
app.run()

@app.route('/')
def hello_world():
    return 'Hello, this is lit-flask@@@!'

@app.route('/predict_text', methods=['POST'])
def predict_text():
    text = request.args.get('text')

    category_list = model_predict(text)

    return str(category_list)
