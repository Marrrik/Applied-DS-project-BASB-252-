from flask import Flask, jsonify, render_template
import pandas as pd
import joblib
import os


app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(
    BASE_DIR, "data", "raw", "Bank", "Bank Customer Churn Prediction.csv"
)
MODEL_FILE = os.path.join(BASE_DIR, "churn_classifier.pkl")


clients = pd.read_csv(CSV_FILE)
model = joblib.load(MODEL_FILE)
cat_features = ['country', 'gender', 'active_member']


@app.route("/predict")
def predict():
    X = clients.drop(columns=["client_id"])
    probs = model.predict_proba(X)[:, 1]
    clients_with_probs = clients.copy()
    clients_with_probs["churn_probability"] = probs
    return clients_with_probs.to_json(orient="records")


@app.route("/")
def index():
    return render_template("undex.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
