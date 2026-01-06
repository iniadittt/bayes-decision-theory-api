from flask import Blueprint, jsonify, request
from flask_cors import CORS
import numpy as np
import joblib, tensorflow as tf
from scipy.stats import multivariate_normal
from tensorflow.keras.preprocessing.image import load_img, img_to_array

api_bp = Blueprint("api", __name__)

bayes = joblib.load("/bayes_skin_model.pkl")
class_names = joblib.load("models/class_names.pkl")
cnn = tf.keras.models.load_model("models/cnn_feature_model.h5")

idx_to_class = {v:k for k,v in class_names.items()}

def preprocess(img):
    img = img.resize((320,320))
    img = np.array(img)/255.0
    return np.expand_dims(img,0)

@api_bp.get("/predict", methods=["POST"])
def predict():
    file = request.files['image']
    img = load_img(file)
    x = preprocess(img)
    feat = cnn.predict(x)[0]
    posteriors = {}
    for c in bayes.mean:
        likelihood = multivariate_normal.pdf(feat, bayes.mean[c], bayes.cov[c])
        posteriors[c] = likelihood * bayes.prior[c]
    total = sum(posteriors.values())
    posteriors = {k:v/total for k,v in posteriors.items()}
    result = sorted(posteriors.items(), key=lambda x:x[1], reverse=True)[:3]
    response = []
    for c,p in result:
        response.append({
            "disease": idx_to_class[c],
            "probability": round(p*100,2)
        })

    return jsonify({
        "top_diagnosis": response[0]["disease"],
        "confidence": response[0]["probability"],
        "top3": response
    })