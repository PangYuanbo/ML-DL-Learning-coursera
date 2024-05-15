# the model for MNIST regression task
import numpy as np


def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -1 / m * np.sum(Y * np.log(AL))
    cost = np.squeeze(cost)
    return cost


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(0, Z)


def tanh(Z):
    return np.tanh(Z)


def sigmoid_backward(dA, Z):
    s = sigmoid(Z)
    return dA * s * (1 - s)


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)


def tanh_backward(dA, Z):
    return dA * (1 - np.power(np.tanh(Z), 2))


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(W, b, activation):
    Z, linear_cache = linear_forward(W, b)
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)
    elif activation == "tanh":
        A = tanh(Z)
    elif activation == "softmax":
        A= softmax(Z)
    cache = (linear_cache, Z)
    return A, cache

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, Z = cache
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, Z)
    elif activation == "relu":
        dZ = relu_backward(dA, Z)
    elif activation == "tanh":
        dZ = tanh_backward(dA, Z)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

class Model(object):
    def __init__(self, layers_dims):
        self.parameters = {}
        self.costs = None
        self.layers_dims = layers_dims
        self.L = len(layers_dims)

    def initialize_parameters(self):
        np.random.seed(3)
        for l in range(1, self.L):
            self.parameters['W' + str(l)] = np.random.randn(self.layers_dims[l], self.layers_dims[l - 1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((self.layers_dims[l], 1))

    def L_model_forward(self, X):
        caches=[]
        A = X
        for l in range(1, self.L):
            A_prev = A
            A, cache = linear_activation_forward(self.parameters['W' + str(l)], self.parameters['b' + str(l)], "relu")
            caches.append(cache)
        AL, cache = linear_activation_forward(self.parameters['W' + str(self.L)], self.parameters['b' + str(self.L)], "softmax")
        caches.append(cache)
        return AL, caches

    def L_model_backward(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = caches[L-1]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "softmax")
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp
        for i in reversed(range(L-1)):
            current_cache = caches[i]
            dAL, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "relu")
            grads["dW" + str(i + 1)] = dW_temp
            grads["db" + str(i + 1)] = db_temp
        return grads

    def update_parameters(self, grads, learning_rate):
        for l in range(1, self.L):
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    def fit(self, X, Y, learning_rate=0.01, num_iterations=3000, print_cost=False):
        self.initialize_parameters()
        for i in range(0, num_iterations):
            AL, caches = self.L_model_forward(X)
            cost = compute_cost(AL, Y)
            grads = self.L_model_backward(AL, Y, caches)
            self.update_parameters(grads, learning_rate)
            if print_cost and i % 100 == 0:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0:
                self.costs.append(cost)



