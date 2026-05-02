import pandas as pd
from joblib import load
import os

def predict_knn(input_data):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # load model + scaler
        model_path = os.path.join(BASE_DIR, "knn_best.pkl")
        scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

        model = load(model_path)
        scaler = load(scaler_path)

        selected_features = [
            "Gender", "AGE", "Urea", "Cr", "HbA1c",
            "Chol", "TG", "HDL", "LDL", "VLDL", "BMI"
        ]

        # validate input
        for f in selected_features:
            if f not in input_data:
                return {"error": f"Thiếu field: {f}"}

        # tạo dataframe
        input_df = pd.DataFrame(
            [[input_data[f] for f in selected_features]],
            columns=selected_features
        )

        print("INPUT:", input_df)

        # scale
        input_scaled = scaler.transform(input_df)

        # predict
        prediction = model.predict(input_scaled)[0]

        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba(input_scaled)[0]
        else:
            prediction_proba = []

        class_mapping = {
            0: "Không mắc bệnh (N)",
            1: "Mắc bệnh (Y)",
            2: "Tiền tiểu đường (P)"
        }

        predicted_class = class_mapping.get(prediction, "Unknown")

        # ✅ RETURN PHẢI NẰM TRONG TRY
        return {
            "Dự đoán": predicted_class,
            "Xác suất": {} if len(prediction_proba) == 0 else {
    class_mapping.get(i, str(i)): round(float(prob), 2)
    for i, prob in enumerate(prediction_proba)
}
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}