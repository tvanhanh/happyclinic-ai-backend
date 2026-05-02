from flask import Blueprint
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from joblib import dump
from app.config.dataset import patient_collection

train_knn_bp = Blueprint('train_knn', __name__)

@train_knn_bp.route('/train', methods=['POST'])
def train_knn_model():
    try:
        data = list(patient_collection.find())
        if not data:
            return {"error": "Không tìm thấy dữ liệu"}, 400

        df = pd.DataFrame(data)

        # ===== CLEAN =====
        df.drop(columns=["ID", "No_Pation", "_id"], inplace=True, errors="ignore")

        df["Gender"] = df["Gender"].astype(str).str.lower().str.strip()
        df["Gender"] = df["Gender"].replace({
            "male": "M", "m": "M", "nam": "M",
            "female": "F", "f": "F", "nữ": "F"
        }).map({"M": 0, "F": 1})

        df["CLASS"] = df["CLASS"].astype(str).str.upper().str.strip()
        df["CLASS"] = df["CLASS"].replace({
            "NO": "N", "NEGATIVE": "N",
            "YES": "Y", "POSITIVE": "Y"
        }).map({"N": 0, "Y": 1, "P": 2})

        df.fillna(df.median(numeric_only=True), inplace=True)

        # ===== FEATURES =====
        features = ["Gender", "AGE", "Urea", "Cr", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI"]
        X = df[features]
        y = df["CLASS"]

        # ===== SMOTE =====
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)

        # ===== SPLIT =====
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ===== SCALE =====
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        dump(scaler, "app/models/scaler.pkl")

        # =========================
        # 🔥 1. KNN THƯỜNG
        # =========================
        knn_basic = KNeighborsClassifier(n_neighbors=5)
        knn_basic.fit(X_train, y_train)
        pred_basic = knn_basic.predict(X_test)
        acc_basic = accuracy_score(y_test, pred_basic)

        # =========================
        # 🔥 2. WEIGHTED KNN
        # =========================
        knn_weighted = KNeighborsClassifier(n_neighbors=5, weights='distance')
        knn_weighted.fit(X_train, y_train)
        pred_weighted = knn_weighted.predict(X_test)
        acc_weighted = accuracy_score(y_test, pred_weighted)

        # =========================
        # 🔥 3. GRID SEARCH (BEST MODEL)
        # =========================
        param_grid = {
            'n_neighbors': [3,5,7,9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }

        grid = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=5,
            scoring='accuracy'
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        pred_best = best_model.predict(X_test)
        acc_best = accuracy_score(y_test, pred_best)

        # ===== SAVE BEST =====
        dump(best_model, "app/models/knn_best.pkl")

        # ===== REPORT =====
        report = classification_report(y_test, pred_best, output_dict=True)

        return {
            "message": "🔥 Train thành công (Advanced KNN)",
            "accuracy": {
                "basic": acc_basic,
                "weighted": acc_weighted,
                "best": acc_best
            },
            "best_params": grid.best_params_,
            "report": report
        }, 200

    except Exception as e:
        return {"error": str(e)}, 500