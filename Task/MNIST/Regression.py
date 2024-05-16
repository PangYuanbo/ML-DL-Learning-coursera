import platform
if platform.system() == 'Windows':
    import cupy as np
elif platform.system() == 'Linux':
    import numpy as np
elif platform.system() == 'Darwin':
    import numpy as np

import pickle

class Model(object):
    def __init__(self, layers_dims):
        self.parameters = {}
        self.costs = []
        self.layers_dims = layers_dims
        self.L = len(layers_dims)

    def initialize_parameters(self):
        np.random.seed(3)
        for l in range(1, self.L):
            self.parameters['W' + str(l)] = np.random.randn(self.layers_dims[l], self.layers_dims[l - 1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((self.layers_dims[l], 1))

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(AL))
        cost = np.squeeze(cost)
        return cost

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    def tanh(self, Z):
        return np.tanh(Z)

    def sigmoid_backward(self, dA, Z):
        s = self.sigmoid(Z)
        return dA * s * (1 - s)

    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def softmax_backward(self, dA, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        softmax = expZ / expZ.sum(axis=0, keepdims=True)
        dZ = softmax * (dA - np.sum(dA * softmax, axis=0, keepdims=True))
        return dZ

    def tanh_backward(self, dA, Z):
        return dA * (1 - np.power(np.tanh(Z), 2))

    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        Z, linear_cache = self.linear_forward(A_prev, W, b)
        if activation == "sigmoid":
            A = self.sigmoid(Z)
        elif activation == "relu":
            A = self.relu(Z)
        elif activation == "tanh":
            A = self.tanh(Z)
        elif activation == "softmax":
            A = self.softmax(Z)
        else:
            A = Z
        cache = (linear_cache, Z)
        return A, cache

    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, Z = cache
        if activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, Z)
        elif activation == "relu":
            dZ = self.relu_backward(dA, Z)
        elif activation == "tanh":
            dZ = self.tanh_backward(dA, Z)
        elif activation == "softmax":
            dZ = self.softmax_backward(dA, Z)
        else:
            dZ = Z
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def L_model_forward(self, X):
        caches = []
        A = X
        for l in range(1, self.L - 1):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)], "relu")
            caches.append(cache)
        A_prev = A
        AL, cache = self.linear_activation_forward(A_prev, self.parameters['W' + str(self.L - 1)], self.parameters['b' + str(self.L - 1)], "softmax")
        caches.append(cache)
        return AL, caches

    def L_model_backward(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = caches[L - 1]
        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dAL, current_cache, "softmax")
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp
        dZ = dA_prev_temp
        for i in reversed(range(L - 1)):
            current_cache = caches[i]
            dZ, dW_temp, db_temp = self.linear_activation_backward(dZ, current_cache, "relu")
            grads["dW" + str(i + 1)] = dW_temp
            grads["db" + str(i + 1)] = db_temp
        return grads

    def update_parameters(self, grads, learning_rate):
        for l in range(1, self.L):
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    def fit(self, X, Y, learning_rate=0.01, num_iterations=3000, print_cost=False, Ifload=False):
        if Ifload:
            self.load_parameters('model_lab/model.pkl')
        self.initialize_parameters()
        for i in range(0, num_iterations):
            AL, caches = self.L_model_forward(X)
            cost = self.compute_cost(AL, Y)
            self.costs.append(cost)
            grads = self.L_model_backward(AL, Y, caches)
            self.update_parameters(grads, learning_rate)
            if print_cost and i % 100 == 0:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        return self.costs

    def predict_cost(self, X, Y):
        AL, caches = self.L_model_forward(X)
        cost = self.compute_cost(AL, Y)
        print("Cost after iteration {}: {}".format(10000, np.squeeze(cost)))

    def predict(self, X):
        AL, _ = self.L_model_forward(X)
        predictions = np.expand_dims(np.argmax(AL, axis=0), axis=0)
        return predictions

    def accuracy(self, X, Y):
        predictions = self.predict(X)
        labels = np.expand_dims(np.argmax(Y, axis=0), axis=0)
        accuracy = np.mean(predictions == labels)
        return accuracy

    def save_parameters(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.parameters, file)

    def load_parameters(self, filename):
        with open(filename, 'rb') as file:
            self.parameters = pickle.load(file)
