from flask import Flask
from controllers.predict_controller import home, predict

app = Flask(__name__)

@app.route('/')
def index():
    return home()

@app.route('/predict', methods=['POST'])
def make_prediction():
    return predict()

if __name__ == '__main__':
    app.run(debug=True)
