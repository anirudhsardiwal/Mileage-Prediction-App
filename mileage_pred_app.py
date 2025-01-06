import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_file, redirect, url_for
import pickle
import os

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("mileage.html")


def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 37)
    loaded_model = pickle.load(open("mileage_prediction_model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route("/result", methods=["POST"])
def result():
    to_predict_list = request.form.to_dict()
    to_predict_list = list(to_predict_list.values())

    to_predict_list[:-2] = list(map(int, to_predict_list[:-2]))

    origin = to_predict_list[-2]
    make = to_predict_list[-1]

    origin_mapping = {"america": 0, "asia": 1, "europe": 2}

    make_mapping = {
        "amc": 0,
        "audi": 1,
        "bmw": 2,
        "buick": 3,
        "cadillac": 4,
        "capri": 5,
        "chevrolet": 6,
        "chrysler": 7,
        "datsun": 8,
        "dodge": 9,
        "fiat": 10,
        "ford": 11,
        "hi": 12,
        "honda": 13,
        "mazda": 14,
        "mercedes": 15,
        "mercury": 16,
        "nissan": 17,
        "oldsmobile": 18,
        "opel": 19,
        "peugeot": 20,
        "plymouth": 21,
        "pontiac": 22,
        "renault": 23,
        "saab": 24,
        "subaru": 25,
        "toyota": 26,
        "triumph": 27,
        "volkswagen": 28,
        "volvo": 29,
    }

    origin_encoded = [0] * (len(origin_mapping) - 1)
    make_encoded = [0] * (len(make_mapping) - 1)

    if origin in origin_mapping:
        origin_index = origin_mapping[origin] - 1
        if origin_index >= 0:
            origin_encoded[origin_index] = 1

    if make in make_mapping:
        make_index = make_mapping[make] - 1
        if make_index >= 0:
            make_encoded[make_index] = 1

    to_predict_list = to_predict_list[:-2] + origin_encoded + make_encoded

    result = ValuePredictor(to_predict_list)
    return render_template("result.html", prediction=result)


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)
            data = pd.read_csv(file_path)

            data["horsepower"] = pd.to_numeric(data["horsepower"], errors="coerce")
            dataEncoded = pd.get_dummies(data, drop_first=True)
            dataEncoded = dataEncoded.replace({False: 0, True: 1})

            for col in dataEncoded.columns:
                dataEncoded[col] = pd.to_numeric(dataEncoded[col], errors="coerce")

            loaded_model = pickle.load(open("mileage_prediction_model.pkl", "rb"))
            predictions = loaded_model.predict(dataEncoded).round(1)

            original_data = pd.read_csv(file_path)
            original_data["predicted_mileage"] = predictions

            original_file_name = os.path.splitext(file.filename)[0]
            result_filename = f"{original_file_name}_predictions.csv"
            result_file_path = os.path.join("uploads", result_filename)
            original_data.to_csv(result_file_path, index=False)
            return send_file(result_file_path, as_attachment=True)
    return redirect(url_for("home"))


# def bulk_predict(data):
#     loaded_model = pickle.load(open("mileage_prediction_model.pkl", "rb"))
#     predictions = loaded_model.predict(data)
#     data["predicted_mileage"] = predictions
#     return data


if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
