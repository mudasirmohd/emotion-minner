from flask import Flask, request
from rnn.infer import Infer

infer = Infer(dimension=32, vocab_size=32889, save_dir='../data/checkpoints/')

app = Flask(__name__)


@app.route('/get_prediction')
def get_data():
    text = request.args['post']
    predicted_class = infer.classify(text)[0]
    return predicted_class


app.run(host='localhost', debug=False, port=5522)
