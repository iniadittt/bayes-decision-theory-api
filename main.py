import os
import joblib
import numpy as np
import tensorflow as tf
from io import BytesIO
from flask_cors import CORS
from flask import Flask, jsonify, request
from scipy.stats import multivariate_normal
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
CORS(app)

class GaussianBayes:
    def __init__(self):
        self.mean = {}
        self.var  = {}
        self.prior = {}

    def fit(self, X, y):
        for c in np.unique(y):
            Xc = X[y == c]
            self.mean[c] = Xc.mean(axis=0)
            self.var[c]  = Xc.var(axis=0) + 1e-6
            self.prior[c] = len(Xc) / len(X)

    def predict(self, X):
        preds = []
        for x in X:
            scores = {}
            for c in self.mean:
                loglik = -0.5 * np.sum(
                    np.log(2*np.pi*self.var[c]) +
                    ((x - self.mean[c])**2) / self.var[c]
                )
                scores[c] = loglik + np.log(self.prior[c])
            preds.append(max(scores, key=scores.get))
        return np.array(preds)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
bayes = joblib.load(os.path.join(BASE_DIR, "models/bayes_skin_model.pkl"))
class_names = joblib.load(os.path.join(BASE_DIR, "models/class_names.pkl"))
cnn = tf.keras.models.load_model(os.path.join(BASE_DIR, "models/cnn_feature_model.h5"))

idx_to_class = {v:k for k,v in class_names.items()}

def preprocess(img, target_size=(320,320)):
    img = img.resize(target_size)
    img = np.array(img)/255.0
    if len(img.shape)==2:
        img = np.stack([img]*3, axis=-1)
    return np.expand_dims(img,0)

@app.get("/")
def read_root():
    return jsonify({
        "success": True,
        "data": {
            "anggota": [
                {"nama": "Aditya Bayu", "nim": "200511140"},
                {"nama": "Bobi", "nim": ""},
                {"nama": "Daffa", "nim": ""},
            ]
        }
    })

@app.post("/predict")
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['image']
    img = load_img(BytesIO(file.read()))
    x = preprocess(img)
    feat = cnn.predict(x)[0]
    posteriors = {}
    for c in bayes.mean:
        likelihood = multivariate_normal.pdf(feat, mean=bayes.mean[c], cov=np.diag(bayes.var[c]))
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
        "success": True,
        "data": {
            "top_diagnosis": response[0]["disease"],
            "confidence": response[0]["probability"],
            "top3": response
        }
    })

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 9000)))