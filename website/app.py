import pickle
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd


flask_app = Flask(__name__)
model = pickle.load(open("factfinder1.pkl", "rb"))
tokenizer = Tokenizer()


# route to index.html by default and get the post request and make prediction on the data
@flask_app.route("/")
def home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    maxlen = 1000
    data = request.form["text"]
    data = [data]
    print(data)
    data = tokenizer.texts_to_sequences(data)
    print(data)
    data = pad_sequences(data, maxlen=maxlen)
    print(data)
    prediction = (model.predict(data)>=0.5).astype(int)
    print(model.predict(data))
    print(prediction)
    #if prediction array is 1 print it is a real news else fake news
    if prediction[0][0] == 1:
        return render_template("index.html", prediction_text="It is a real news")
    else:
        return render_template("index.html", prediction_text="It is a fake news")


if __name__ == "__main__":
    flask_app.run(debug=True)

