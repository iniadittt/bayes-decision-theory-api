import __main__
import os
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
from flask import Flask, request, jsonify
from bdt_model import BayesianDecisionModel

__main__.BayesianDecisionModel = BayesianDecisionModel

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR,"model","bdt_skin_model.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR,"model","encoders.pkl"))
X_cols = joblib.load(os.path.join(BASE_DIR,"model","feature_columns.pkl"))

BOOL_SET = {True, False, "True", "False", "true", "false"}
AGE_SET = {"18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"}
RACE_SET = {"American Indian/Alaskan Native", "Asian", "Black", "Hispanic", "Other", "White"}
DIABETIC_SET = {"No", "No (borderline diabetes)", "Yes", "Yes (during pregnancy)"}
GENHEALTH_SET = {"Excellent", "Fair", "Good", "Poor", "Very good"}

def normalize_bool(v):
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, str):
        if v.lower() in ["true", "yes", "1", "false", "no", "0"]:
            v = v.lower()
            if v in ["true", "yes", "1"]:
                return "Yes"
            if v in ["false", "no", "0"]:
                return "No"
        else:
            return v
    return v

def validate(data):
    try:
        for f in ["HeartDisease", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", "PhysicalActivity", "Asthma", "KidneyDisease"]:
            if data[f] not in BOOL_SET:
                return f"{f} harus True/False"
        if not isinstance(data["BMI"], (int,float)):
            return "BMI harus angka"
        if not (0 <= float(data["PhysicalHealth"]) <= 30):
            return "PhysicalHealth harus 0-30"
        if not (0 <= float(data["MentalHealth"]) <= 30):
            return "MentalHealth harus 0-30"
        if data["Sex"] not in ["Male","Female"]:
            return "Sex hanya Male/Female"
        if data["AgeCategory"] not in AGE_SET:
            return "AgeCategory tidak valid"
        if data["Race"] not in RACE_SET:
            return "Race tidak valid"
        if data["Diabetic"] not in DIABETIC_SET:
            return "Diabetic tidak valid"
        if data["GenHealth"] not in GENHEALTH_SET:
            return "GenHealth tidak valid"
        if not (0 <= float(data["SleepTime"]) <= 24):
            return "SleepTime harus 0–24"
    except:
        return "Request body rusak / tidak lengkap"
    return None

@app.get("/")
def read_root():
    return jsonify({
        "success": True,
        "data": {
            "anggota": [
                {"nama": "Aditya Bayu Aji", "nim": "200511140"},
                {"nama": "Ifhals Bobby Duhita", "nim": "220511186"},
                {"nama": "Muhammad Daffa Raditha Pratama", "nim": "220511144"},
            ]
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    REQUIRED_FIELDS = ["HeartDisease", "BMI", "Smoking", "AlcoholDrinking", "Stroke", "PhysicalHealth", "MentalHealth", "DiffWalking", "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity", "GenHealth", "SleepTime", "Asthma", "KidneyDisease"]
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({
            "success": False,
            "message": "Data salah",
            "data": {
                "error": missing
            }
        }), 400
    err = validate(data)
    if err:
        return jsonify({
            "success": False,
            "message": "Data salah",
            "data": {
                "error": err
            }
        }), 400
    x = []
    for col in X_cols:
        val = data[col]
        val = normalize_bool(val)
        if col in encoders:
            val = encoders[col].transform([val])[0]
        x.append(val)
    x = np.array(x)
    posterior = model.posterior(x)
    decision = min(model.loss,  key=lambda d: sum(model.loss[d][c]*posterior[c] for c in model.classes))
    return jsonify({
        "success": True,
        "message": "Berhasil prediksi",
        "data": {
            "decision": decision,
            "posterior": {k.lower(): float(v) for k,v in posterior.items()}
        }
        
    })

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 9000)))