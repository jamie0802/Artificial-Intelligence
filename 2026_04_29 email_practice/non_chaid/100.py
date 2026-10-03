import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import chi2

df = pd.read_csv("spam_dataset_100.csv")
X = df.drop(columns=["is_spam"])
y = df["is_spam"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average="macro"))
    print("Recall   :", recall_score(y_true, y_pred, average="macro"))
    print("F1-score :", f1_score(y_true, y_pred, average="macro"))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

def print_f1(name, y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name} F1-score: {f1:.4f}")

scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

print("\n開始驗證模型表現並找出最好超參數")

# ===================
# Logistic Regression
# ===================
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_std, y_train)
y_log = log_model.predict(X_test_std)
print_f1("Logistic Regression", y_test, y_log)

# ==================
# Random Forest（交叉驗證找最佳超參數）
# ==================
n_estimators_range = [50, 100, 200, 300, 500]
max_depth_range = [5, 10, 20, 30]

rf_best_score = -1
rf_best_params = {}

for n_est in n_estimators_range:
    for max_d in max_depth_range:
        fold_scores = []

        for train_idx, val_idx in kf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, bootstrap=True,random_state=42)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            score = f1_score(y_val, y_pred, average="macro")
            fold_scores.append(score)

        mean_score = np.mean(fold_scores)

        if mean_score > rf_best_score:
            rf_best_score = mean_score
            rf_best_params = {"n_estimators": n_est, "max_depth": max_d}

print("RF Best Params:", rf_best_params)
print("RF Best CV F1:", rf_best_score)

rf_model = RandomForestClassifier(
    n_estimators=rf_best_params["n_estimators"],
    max_depth=rf_best_params["max_depth"],
    random_state=42
)
rf_model.fit(X_train, y_train)
y_rf = rf_model.predict(X_test)
print_f1("Random Forest", y_test, y_rf)

# ==================
# SVM（用獨立變數名避免被 KNN 覆蓋）
# ==================
C_range = [0.01, 0.1, 1, 10, 100]
kernel_options = ["linear", "rbf", "sigmoid", "poly"]

svm_best_score = -1          
svm_best_params = {}         

for kernel in kernel_options:
    for c in C_range:
        fold_scores = []

        for train_idx, val_idx in kf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            inner_scaler = StandardScaler()
            X_tr_std = inner_scaler.fit_transform(X_tr)
            X_val_std = inner_scaler.transform(X_val)

            model = SVC(kernel=kernel, C=c)
            model.fit(X_tr_std, y_tr)
            y_pred = model.predict(X_val_std)

            score = f1_score(y_val, y_pred, average="macro")
            fold_scores.append(score)

        mean_score = np.mean(fold_scores)

        if mean_score > svm_best_score:
            svm_best_score = mean_score
            svm_best_params = {"kernel": kernel, "C": c}

print("SVM Best Params:", svm_best_params)
print("SVM Best CV F1:", svm_best_score)

svm_model = SVC(kernel=svm_best_params["kernel"], C=svm_best_params["C"])
svm_model.fit(X_train_std, y_train)
y_svm = svm_model.predict(X_test_std)
print_f1("SVM", y_test, y_svm)

# ==================
# KNN（用獨立變數名）
# ==================
k_range = range(1, 6)
metrics = ["manhattan", "euclidean" , "chebyshev"]
weights_options = ["uniform", "distance"]

knn_best_score = -1
knn_best_params = {}         

for m in metrics:
    for w in weights_options:
        for k in k_range:
            fold_scores = []

            for train_idx, val_idx in kf.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                inner_scaler = StandardScaler()
                X_tr_std = inner_scaler.fit_transform(X_tr)
                X_val_std = inner_scaler.transform(X_val)

                model = KNeighborsClassifier(n_neighbors=k, metric=m, weights=w)
                model.fit(X_tr_std, y_tr)
                y_pred = model.predict(X_val_std)

                score = f1_score(y_val, y_pred, average="macro")
                fold_scores.append(score)

            mean_score = np.mean(fold_scores)

            if mean_score > knn_best_score:
                knn_best_score = mean_score
                knn_best_params = {"k": k, "metric": m, "weight": w}

print("KNN Best Params:", knn_best_params)
print("KNN Best CV F1:", knn_best_score)

knn_model = KNeighborsClassifier(
    n_neighbors=knn_best_params["k"],
    metric=knn_best_params["metric"],
    weights=knn_best_params["weight"]
)
knn_model.fit(X_train_std, y_train)
y_knn = knn_model.predict(X_test_std)
print_f1("KNN", y_test, y_knn)

# =====================
# 用 spam_test.csv 測試
# =====================
test_df = pd.read_csv("spam_test.csv")

X_test_final = test_df.drop(columns=['is_spam'])
y_test_final = test_df["is_spam"]

scaler_final = StandardScaler()
X_new_train_std = scaler_final.fit_transform(X)
X_test_final_std = scaler_final.transform(X_test_final)

log_model_final = LogisticRegression(max_iter=1000, random_state=42)
log_model_final.fit(X_new_train_std, y)
y_log_final = log_model_final.predict(X_test_final_std)
evaluate_model("Logistic Regression", y_test_final, y_log_final)

rf_model_final = RandomForestClassifier(
    n_estimators=rf_best_params["n_estimators"],
    max_depth=rf_best_params["max_depth"],
    random_state=42
)
rf_model_final.fit(X, y)
y_rf_final = rf_model_final.predict(X_test_final)
evaluate_model("Random Forest", y_test_final, y_rf_final)

svm_model_final = SVC(kernel=svm_best_params["kernel"], C=svm_best_params["C"])
svm_model_final.fit(X_new_train_std, y)
y_svm_final = svm_model_final.predict(X_test_final_std)
evaluate_model("SVM", y_test_final, y_svm_final)

knn_model_final = KNeighborsClassifier(
    n_neighbors=knn_best_params["k"],       
    metric=knn_best_params["metric"],
    weights=knn_best_params["weight"]
)
knn_model_final.fit(X_new_train_std, y)
y_knn_final = knn_model_final.predict(X_test_final_std)
evaluate_model("KNN", y_test_final, y_knn_final)