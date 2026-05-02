#Custom_KNN và Fuzzy KNN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, log_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from collections import Counter
from sklearn.metrics import mean_squared_error
import time

# Định nghĩa CustomKNN
class CustomKNN:
    def __init__(self, k=5):
        self.k = k
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
    def predict(self, X):
        X = np.array(X)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    def _predict_single(self, x):
        distances = [self.manhattan_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        k_distances = [distances[i] for i in k_indices]
        weights = [1 / (d + 1e-10) for d in k_distances]
        weighted_votes = {}
        for label, weight in zip(k_labels, weights):
            weighted_votes[label] = weighted_votes.get(label, 0) + weight
        return max(weighted_votes, key=weighted_votes.get)
    def predict_proba(self, X):
        X = np.array(X)
        probas = []
        for x in X:
            distances = [self.manhattan_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices]
            k_distances = [distances[i] for i in k_indices]
            weights = [1 / (d + 1e-10) for d in k_distances]
            total_weight = sum(weights)
            class_weights = {}
            for label, weight in zip(k_labels, weights):
                class_weights[label] = class_weights.get(label, 0) + weight
            proba = [class_weights.get(i, 0) / total_weight for i in range(3)]
            probas.append(proba)
        return np.array(probas)

# Định nghĩa Fuzzy KNN
def fuzzy_knn_predict(X_train, y_train, X_test, k=5):
    n_classes = len(np.unique(y_train))
    y_pred = []
    for test_sample in X_test:
        distances = np.linalg.norm(X_train - test_sample, axis=1)
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_indices]
        membership = np.zeros(n_classes)
        for i, label in enumerate(nearest_labels):
            weight = 1 / (distances[nearest_indices[i]] + 1e-5)
            membership[label] += weight
        predicted_label = np.argmax(membership)
        y_pred.append(predicted_label)
    return np.array(y_pred)

def fuzzy_knn_predict_proba(X_train, y_train, X_test, k=5):
    n_classes = len(np.unique(y_train))
    probas = []
    for test_sample in X_test:
        distances = np.linalg.norm(X_train - test_sample, axis=1)
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_indices]
        membership = np.zeros(n_classes)
        for i, label in enumerate(nearest_labels):
            weight = 1 / (distances[nearest_indices[i]] + 1e-5)
            membership[label] += weight
        total_weight = np.sum(membership)
        proba = membership / total_weight if total_weight > 0 else np.ones(n_classes) / n_classes
        probas.append(proba)
    return np.array(probas)

# Đọc dữ liệu
file_path = "/content/drive/MyDrive/colab/Datamain.csv"
try:
    df = pd.read_csv(file_path, encoding="utf-8")
    print("✅ Đọc dữ liệu thành công!")
except Exception as e:
    print(f"❌ Lỗi khi đọc dữ liệu: {e}")
    exit()

# Loại bỏ cột không cần thiết
columns_to_drop = ["ID", "No_Pation"]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors="ignore")

# Chuyển đổi Gender
df["Gender"] = df["Gender"].map({"M": 0, "F": 1})

# Lọc CLASS hợp lệ
df = df[df["CLASS"].isin(["P", "Y", "N"])]

# Mã hóa CLASS
class_mapping = {"N": 0, "Y": 1, "P": 2}
df["CLASS"] = df["CLASS"].map(class_mapping)

# Xử lý NaN
df.fillna(df.median(), inplace=True)

# Kiểm tra số lượng mẫu
print("\n📊 Số lượng mẫu trong từng lớp CLASS:")
print(df["CLASS"].value_counts())

# Tách dữ liệu
selected_features = ["Gender", "AGE", "Urea", "Cr", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI"]
X = df[selected_features]
y = df["CLASS"]

# Cân bằng bằng SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Chia train-test
X_train, X_test, y_train, y_test = train_test_split(
    X_smote, y_smote, test_size=0.2, random_state=42, stratify=y_smote
)

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "./scaler.pkl")

# Chuyển thành numpy array
X_train_np = np.array(X_train_scaled)
X_test_np = np.array(X_test_scaled)
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

# Kiểm tra số lượng mẫu sau SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train_np, palette="Set2", edgecolor="black")
plt.title("📊 Số lượng mẫu sau khi cân bằng bằng SMOTE")
plt.xlabel("CLASS")
plt.ylabel("Số lượng")
plt.xticks([0, 1, 2], ["Không mắc", "Mắc", "Tiền tiểu đường"])
plt.show()

# === So sánh CustomKNN và Fuzzy KNN ===
k_values = list(range(1, 21, 2))
metrics = {
    'k': k_values,
    'accuracy_custom': [], 'precision_custom': [], 'recall_custom': [], 'f1_custom': [],
    'log_loss_custom': [], 'mae_custom': [], 'rmse_custom': [],
    'accuracy_fuzzy': [], 'precision_fuzzy': [], 'recall_fuzzy': [], 'f1_fuzzy': [],
    'log_loss_fuzzy': [], 'mae_fuzzy': [], 'rmse_fuzzy': []
}

# Vòng lặp so sánh
for k in k_values:
    # CustomKNN
    knn_custom = CustomKNN(k=k)
    knn_custom.fit(X_train_np, y_train_np)
    y_pred_custom = knn_custom.predict(X_test_np)
    y_proba_custom = knn_custom.predict_proba(X_test_np)

    metrics['accuracy_custom'].append(accuracy_score(y_test_np, y_pred_custom))
    metrics['precision_custom'].append(precision_score(y_test_np, y_pred_custom, average='weighted'))
    metrics['recall_custom'].append(recall_score(y_test_np, y_pred_custom, average='weighted'))
    metrics['f1_custom'].append(f1_score(y_test_np, y_pred_custom, average='weighted'))
    metrics['log_loss_custom'].append(log_loss(y_test_np, y_proba_custom))

    # MAE và RMSE trên xác suất
    # MAE và RMSE trên xác suất
    y_test_one_hot = label_binarize(y_test_np, classes=[0, 1, 2])
    metrics['mae_custom'].append(mean_absolute_error(y_test_one_hot, y_proba_custom))

# Tính MSE rồi tự lấy căn bậc hai để được RMSE
    mse = mean_squared_error(y_test_one_hot, y_proba_custom)
    rmse = np.sqrt(mse)
    metrics['rmse_custom'].append(rmse)

    # Fuzzy KNN
    y_pred_fuzzy = fuzzy_knn_predict(X_train_np, y_train_np, X_test_np, k=k)
    y_proba_fuzzy = fuzzy_knn_predict_proba(X_train_np, y_train_np, X_test_np, k=k)

    metrics['accuracy_fuzzy'].append(accuracy_score(y_test_np, y_pred_fuzzy))
    metrics['precision_fuzzy'].append(precision_score(y_test_np, y_pred_fuzzy, average='weighted'))
    metrics['recall_fuzzy'].append(recall_score(y_test_np, y_pred_fuzzy, average='weighted'))
    metrics['f1_fuzzy'].append(f1_score(y_test_np, y_pred_fuzzy, average='weighted'))
    metrics['log_loss_fuzzy'].append(log_loss(y_test_np, y_proba_fuzzy))
    metrics['mae_fuzzy'].append(mean_absolute_error(y_test_one_hot, y_proba_fuzzy))
    # RMSE cho Fuzzy KNN
    mse_fuzzy = mean_squared_error(y_test_one_hot, y_proba_fuzzy)
    rmse_fuzzy = np.sqrt(mse_fuzzy)
    metrics['rmse_fuzzy'].append(rmse_fuzzy)

# Chi tiết cho k=5
print("\n=== So sánh CustomKNN và Fuzzy KNN (k=5) ===")

# CustomKNN
start_train = time.time()
knn_custom = CustomKNN(k=5)
knn_custom.fit(X_train_np, y_train_np)
joblib.dump(knn_custom, "./knn_custom_model.pkl")
y_pred_custom = knn_custom.predict(X_test_np)
end_train = time.time()
print(f"⏱️ Thời gian huấn luyện (CustomKNN): {end_train - start_train:.4f} giây")
y_proba_custom = knn_custom.predict_proba(X_test_np)

y_train_pred= knn_custom.predict(X_train_np)
y_train_proba = knn_custom.predict_proba(X_train_np)

print("\nCustomKNN (k=5):")

print("Train Performance:")
print(f"Accuracy: {accuracy_score(y_train_np, y_train_pred):.4f}")
print(f"Precision (weighted): {precision_score(y_train_np, y_train_pred, average='weighted'):.4f}")
print(f"Recall (weighted): {recall_score(y_train_np, y_train_pred, average='weighted'):.4f}")
print(f"F1 Score (weighted): {f1_score(y_train_np, y_train_pred, average='weighted'):.4f}")
print(f"Log Loss: {log_loss(y_train_np, y_train_proba):.4f}")

print("\nTest Performance:")
print(f"Accuracy: {accuracy_score(y_test_np, y_pred_custom):.4f}")
print(f"Precision (weighted): {precision_score(y_test_np, y_pred_custom, average='weighted'):.4f}")
print(f"Recall (weighted): {recall_score(y_test_np, y_pred_custom, average='weighted'):.4f}")
print(f"F1 Score (weighted): {f1_score(y_test_np, y_pred_custom, average='weighted'):.4f}")
print(f"Log Loss: {log_loss(y_test_np, y_proba_custom):.4f}")
print(f"MAE (proba): {mean_absolute_error(label_binarize(y_test_np, classes=[0, 1, 2]), y_proba_custom):.4f}")
y_test_one_hot = label_binarize(y_test_np, classes=[0, 1, 2])
mse = mean_squared_error(y_test_one_hot, y_proba_custom)
rmse = np.sqrt(mse)
print(f"RMSE (proba): {rmse:.4f}")

print("Classification Report:")
print(classification_report(y_test_np, y_pred_custom, target_names=["Không mắc", "Mắc", "Tiền tiểu đường"]))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test_np, y_pred_custom), annot=True, fmt='d', cmap='Blues',
            xticklabels=["N", "Y", "P"], yticklabels=["N", "Y", "P"])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - CustomKNN (k=5)')
plt.show()

# Fuzzy KNN
start_train_fuzzy = time.time()
y_pred_fuzzy = fuzzy_knn_predict(X_train_np, y_train_np, X_test_np, k=5)
end_train_fuzzy = time.time()
print(f"⏱️ Thời gian huấn luyện (FuzzyKNN): {end_train_fuzzy - start_train_fuzzy:.4f} giây")

model_data = {
    'X_train': X_train_scaled,
    'y_train': y_train,
    'best_k': k,
    'class_mapping': class_mapping,
    'scaler': scaler,
    'selected_features': selected_features
}
joblib.dump(model_data, "./knn_model.pkl")
y_proba_fuzzy = fuzzy_knn_predict_proba(X_train_np, y_train_np, X_test_np, k=5)
print("\nFuzzy KNN (k=5):")
print(f"Accuracy: {accuracy_score(y_test_np, y_pred_fuzzy):.4f}")
print(f"Precision (weighted): {precision_score(y_test_np, y_pred_fuzzy, average='weighted'):.4f}")
print(f"Recall (weighted): {recall_score(y_test_np, y_pred_fuzzy, average='weighted'):.4f}")
print(f"F1 Score (weighted): {f1_score(y_test_np, y_pred_fuzzy, average='weighted'):.4f}")
print(f"Log Loss: {log_loss(y_test_np, y_proba_fuzzy):.4f}")
print(f"MAE (proba): {mean_absolute_error(label_binarize(y_test_np, classes=[0, 1, 2]), y_proba_fuzzy):.4f}")
y_test_one_hot = label_binarize(y_test_np, classes=[0, 1, 2])
mse_fuzzy = mean_squared_error(y_test_one_hot, y_proba_fuzzy)
rmse_fuzzy = np.sqrt(mse_fuzzy)
print(f"RMSE (proba fuzzy): {rmse_fuzzy:.4f}")
print("Classification Report:")
print(classification_report(y_test_np, y_pred_fuzzy, target_names=["Không mắc", "Mắc", "Tiền tiểu đường"]))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test_np, y_pred_fuzzy), annot=True, fmt='d', cmap='Blues',
            xticklabels=["N", "Y", "P"], yticklabels=["N", "Y", "P"])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Fuzzy KNN (k=5)')
plt.show()

# ROC-AUC
y_test_bin = label_binarize(y_test_np, classes=[0, 1, 2])
plt.figure(figsize=(8, 6))
for i, label in enumerate(["Không mắc", "Mắc", "Tiền tiểu đường"]):
    fpr_c, tpr_c, _ = roc_curve(y_test_bin[:, i], y_proba_custom[:, i])
    auc_c = roc_auc_score(y_test_bin[:, i], y_proba_custom[:, i])
    plt.plot(fpr_c, tpr_c, label=f'CustomKNN - {label} (AUC = {auc_c:.2f})')

    fpr_f, tpr_f, _ = roc_curve(y_test_bin[:, i], y_proba_fuzzy[:, i])
    auc_f = roc_auc_score(y_test_bin[:, i], y_proba_fuzzy[:, i])
    plt.plot(fpr_f, tpr_f, linestyle='--', label=f'Fuzzy KNN - {label} (AUC = {auc_f:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - CustomKNN vs Fuzzy KNN (k=5)')
plt.legend()
plt.grid(True)
plt.show()

# Biểu đồ so sánh các chỉ số
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('So sánh CustomKNN và Fuzzy KNN theo k')

# Accuracy
axes[0, 0].plot(k_values, metrics['accuracy_custom'], marker='o', linestyle='-', label='CustomKNN')
axes[0, 0].plot(k_values, metrics['accuracy_fuzzy'], marker='s', linestyle='--', label='Fuzzy KNN')
axes[0, 0].set_title('Accuracy')
axes[0, 0].set_xlabel('k')
axes[0, 0].set_ylabel('Score')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Precision
axes[0, 1].plot(k_values, metrics['precision_custom'], marker='o', linestyle='-', label='CustomKNN')
axes[0, 1].plot(k_values, metrics['precision_fuzzy'], marker='s', linestyle='--', label='Fuzzy KNN')
axes[0, 1].set_title('Precision (weighted)')
axes[0, 1].set_xlabel('k')
axes[0, 1].set_ylabel('Score')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Recall
axes[1, 0].plot(k_values, metrics['recall_custom'], marker='o', linestyle='-', label='CustomKNN')
axes[1, 0].plot(k_values, metrics['recall_fuzzy'], marker='s', linestyle='--', label='Fuzzy KNN')
axes[1, 0].set_title('Recall (weighted)')
axes[1, 0].set_xlabel('k')
axes[1, 0].set_ylabel('Score')
axes[1, 0].legend()
axes[1, 0].grid(True)

# F1-score
axes[1, 1].plot(k_values, metrics['f1_custom'], marker='o', linestyle='-', label='CustomKNN')
axes[1, 1].plot(k_values, metrics['f1_fuzzy'], marker='s', linestyle='--', label='Fuzzy KNN')
axes[1, 1].set_title('F1 Score (weighted)')
axes[1, 1].set_xlabel('k')
axes[1, 1].set_ylabel('Score')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Biểu đồ Log Loss, MAE, RMSE
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('So sánh Log Loss, MAE, RMSE theo k')

# Log Loss
axes[0].plot(k_values, metrics['log_loss_custom'], marker='o', linestyle='-', label='CustomKNN')
axes[0].plot(k_values, metrics['log_loss_fuzzy'], marker='s', linestyle='--', label='Fuzzy KNN')
axes[0].set_title('Log Loss')
axes[0].set_xlabel('k')
axes[0].set_ylabel('Score')
axes[0].legend()
axes[0].grid(True)

# MAE
axes[1].plot(k_values, metrics['mae_custom'], marker='o', linestyle='-', label='CustomKNN')
axes[1].plot(k_values, metrics['mae_fuzzy'], marker='s', linestyle='--', label='Fuzzy KNN')
axes[1].set_title('MAE (proba)')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Score')
axes[1].legend()
axes[1].grid(True)

# RMSE
axes[2].plot(k_values, metrics['rmse_custom'], marker='o', linestyle='-', label='CustomKNN')
axes[2].plot(k_values, metrics['rmse_fuzzy'], marker='s', linestyle='--', label='Fuzzy KNN')
axes[2].set_title('RMSE (proba)')
axes[2].set_xlabel('k')
axes[2].set_ylabel('Score')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_train_np)
unique_classes = np.unique(y_train_np)
palette = sns.color_palette("tab10", len(unique_classes))
plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_train_np, palette=palette, alpha=0.8)
handles, labels = scatter.get_legend_handles_labels()
class_names = {0: "Không mắc", 1: "Mắc", 2: "Tiền tiểu đường"}
labels = [class_names[int(l)] for l in labels]
plt.legend(handles, labels, title="CLASS")
plt.title("📊 Phân bố dữ liệu bằng t-SNE")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()

# Ma trận tương quan
correlation_matrix = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Biểu đồ tương quan giữa các đặc điểm')
plt.show()