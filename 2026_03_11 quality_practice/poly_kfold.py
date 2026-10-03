import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

df = pd.read_csv("manufacturing_overfit.csv")
X = df.drop(columns=["Quality Rating"])
y = df["Quality Rating"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

degrees = range(1,11)

degree_mse = []

for d in degrees:
    fold_mse = []

    for train_index, test_index in kf.split(X):
        # 切分資料
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        poly = PolynomialFeatures(degree=d)

        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_poly,y_train)
        y_pred = model.predict(X_test_poly)

        mse = mean_squared_error(y_test, y_pred)
        fold_mse.append(mse)

    degree_mse.append(np.mean(fold_mse))
    print(f"degree = {d}, MSE = {np.mean(fold_mse):.4f}")

plt.plot(degrees, degree_mse, marker='o')
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.title("Model Selection")
plt.show()

best_degree = degrees[np.argmin(degree_mse)]
print("Best degree:", best_degree)