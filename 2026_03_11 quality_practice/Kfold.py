import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("manufacturing_overfit.csv")

X = df[["Temperature (°C)",
        "Pressure (kPa)",
        "Temperature x Pressure",
        "Material Fusion Metric",
        "Material Transformation Metric"
]]

y = df["Quality Rating"]

degrees = range(1,11)

mse_list = []

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

degree_mse = []

for d in degrees:
    fold_mse = []

    for train_index, test_index in kf.split(X):
        # 切分資料
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 建立模型
        poly = PolynomialFeatures(degree=d)

        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # 訓練模型
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # 預測 test data
        y_pred = model.predict(X_test_poly)

        # 評估
        mse = mean_squared_error(y_test, y_pred)

        # 紀錄結果
        fold_mse.append(mse)
    
    avg_mse = np.mean(fold_mse)

    degree_mse.append(avg_mse)

    print(f"degree = {d}, MSE = {avg_mse:.4f}")

plt.plot(degrees, degree_mse, marker='o')
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.title("Model Selection")
plt.show()

best_degree = degrees[np.argmin(degree_mse)]

print("Best degree:", best_degree)


