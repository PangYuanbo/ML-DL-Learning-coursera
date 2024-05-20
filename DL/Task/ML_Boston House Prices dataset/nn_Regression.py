import numpy as np


class nn_Regressor:
    def __init__(self, X, Y, n_h, learning_rate=0.01, epochs=1000):
        self.X = X
        self.Y = Y
        self.n_h = n_h
        self.n_x = self.X.shape[1]
        self.n_y =1
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m = self.X.shape[0]

    def initialize_weights(self):
        W1 = np.random.randn(self.n_x, self.n_h)
        b1 = np.zeros((1, self.n_h))
        W2 = np.random.randn(self.n_h, 1)
        b2 = np.zeros((1, 1))
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        return parameters

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward_propagation(self, X, parameters):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        Z1 = np.dot(X, W1) + b1
        A1 =np.tanh(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = Z2
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def compute_cost(self, A2):
        cost = - np.sum(np.power((A2-self.Y),2) )/ self.m
        return cost

    def backward_propagation(self, parameters, cache):
        W1 = parameters['W1']
        W2 = parameters['W2']
        A1 = cache['A1']
        A2 = cache['A2']
        dZ2 =2*( A2 - self.Y)
        dW2 = np.dot(A1.T, dZ2) / self.m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / self.m
        dZ1 = np.dot(dZ2, W2.T) * (1 - np.power(A1, 2))
        dW1 = np.dot(self.X.T, dZ1) / self.m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / self.m
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return grads

    def update_parameters(self, parameters, grads):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']
        W1 = W1 - self.learning_rate * dW1
        b1 = b1 - self.learning_rate * db1
        W2 = W2 - self.learning_rate * dW2
        b2 = b2 - self.learning_rate * db2
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        return parameters

    def fit(self):
        parameters = self.initialize_weights()
        for i in range(0, self.epochs):
            A2, cache = self.forward_propagation(self.X, parameters)
            cost = self.compute_cost(A2)
            grads = self.backward_propagation(parameters, cache)
            parameters = self.update_parameters(parameters, grads)
            if i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
        return parameters

    def predict(self, X, parameters):
        A2, cache = self.forward_propagation(X, parameters)
        predictions = np.round(A2)
        return predictions
