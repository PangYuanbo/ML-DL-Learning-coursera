import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

from Regression import LinearRegression


def z_score_normalize(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_standardized = (X - X_mean) / X_std
    return X_standardized


df = pd.read_csv('boston.csv')
np_df = np.array(df)
np_value = np_df[:, 13]
np_data = np_df[:, 0:13]
np_data = z_score_normalize(np_data)
print(np_data.shape, np_value.shape)
t = time.time()
model = LinearRegression()
model.fit(np_data, np_value, epochs=1)
y_pred = model.predict(np_data)
print("Time:", time.time() - t)
# 计算均方误差
mse = np.mean((np_value - y_pred) ** 2)
print("MSE:", mse)
