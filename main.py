from flask import Flask
from flask_cors import CORS
from endpoints import api_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(api_bp)

@app.get("/")
def read_root():
    return jsonify(
        {
            "success": True,
            "data": {
                "anggota": [
                    {"nama": "Aditya Bayu", "nim": "200511140"},
                    {"nama": "Bobi", "nim": ""},
                    {"nama": "Daffa", "nim": ""},
                ]
            }
        }
    )
