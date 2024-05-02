import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self):
        self.X = None
        self.y = None
        self.weights = None
        self.b = None

    def fit(self, X, y, batch_size=1, lr=0.01, epochs=1000):
        self.X = X
        self.y = y
        n_samples, n_features = self.X.shape
        self.weights = np.zeros(n_features)
        self.b = 0
        for i in range(epochs):
            y_pred = self.weights @ self.X.T + self.b
            cost = y_pred - self.y
            self.weights -= (lr / n_samples) * X.T @ cost
            self.b -= (lr / n_samples) * np.sum(cost)

    def predict(self, X):
        return np.dot(X, self.weights) + self.b


df = pd.read_csv('boston.csv')
np_df = np.array(df)
np_value = np_df[:, 13]
np_data = np_df[:, 0:13]
model = LinearRegression()
model.fit(np_data, np_value)
y_pred = model.predict(np_data)

# 计算均方误差
mse = np.mean((np_value - y_pred)**2)
print("MSE:", mse)
