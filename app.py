from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

data = joblib.load("California_housing.joblib")

model = data["Model"]
columns = data["Columns"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    MedInc = float(request.form["MedInc"])
    HouseAge = float(request.form["HouseAge"])
    AveRooms = float(request.form["AveRooms"])
    Population = float(request.form["Population"])
    AveOccup = float(request.form["AveOccup"])
    Latitude = float(request.form["Latitude"])

    input_data = np.array([[
        MedInc,
        HouseAge,
        AveRooms,
        Population,
        AveOccup,
        Latitude
    ]])

    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction=float(prediction)
    )


if __name__ == "__main__":
    app.run(debug=True)