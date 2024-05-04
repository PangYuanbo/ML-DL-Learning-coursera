import numpy as np
class LinearRegression:
    def __init__(self):
        self.X = None
        self.y = None
        self.weights = None
        self.b = None
        self.batch_size = 32

    def fit(self, X, y, lr=1, epochs=10000):
        self.X = X
        self.y = y
        n_samples, n_features = self.X.shape
        self.weights = np.zeros(n_features)
        self.b = 0
        for i in range(epochs):
            # np.random.shuffle(self.X)
            for j in self.get_batches(self.X, self.batch_size):
                y_pred = self.weights @ self.X[j].T + self.b
                print(X[j])
                cost = y_pred - self.y[j]
                self.weights -= (lr / n_samples) * self.X[j].T @ cost
                self.b -= (lr / n_samples) * np.sum(cost)

    def get_batches(self, X, batch_size=32):
        self.batch_size = batch_size
        self.X = X
        n_samples = len(self.X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch_idx = indices[start:end]
            yield batch_idx

    def predict(self, X):
        return np.dot(X, self.weights) + self.b
