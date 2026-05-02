import os
from flask import Flask
from flask_cors import CORS
from app.api.predict import predict_bp
from app.models.train_knn import train_knn_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(train_knn_bp, url_prefix='/api')
app.register_blueprint(predict_bp, url_prefix='/api')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)