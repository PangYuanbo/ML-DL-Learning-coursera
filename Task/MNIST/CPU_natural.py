# the model for MNIST regression task
import numpy as np
import pickle
import struct
import os
from array import array


class ReadMNIST(object):
    def __init__(self, train_images_filepath, train_labels_filepath, test_images_filepath, test_labels_filepath):
        self.train_images_filepath = train_images_filepath
        self.train_labels_filepath = train_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049 but got {}'.format(magic))
            labels = array('B', file.read())
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051 but got {}'.format(magic))
            image_data = array('B', file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            images[i][:] = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(rows, cols)
        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.train_images_filepath, self.train_labels_filepath)
        x_train=self.np_change(x_train)
        y_train=self.OneHot(y_train)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        x_test=self.np_change(x_test)
        y_test=self.OneHot(y_test)
        return (x_train, y_train), (x_test, y_test)

    def np_change(self, x):
        np_x = np.array(x)
        np_x= np_x.reshape(np_x.shape[0],-1).T
        np_x = np_x/255
        return np_x

    def OneHot(self, y):
        y=np.array(y)
        onehot = np.zeros((y.max() + 1, y.size))
        onehot[y, np.arange(y.size)] = 1
        return onehot


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
def softmax_backward(dA, Z):
    """
    Backward propagation for a single softmax layer.

    Arguments:
    dA -- Gradient of the loss with respect to the output of the softmax layer
    Z -- The input to the softmax layer (pre-activation values)

    Returns:
    dZ -- Gradient of the loss with respect to the input of the softmax layer
    """
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    softmax = expZ / expZ.sum(axis=0, keepdims=True)

    # Compute the gradient dZ
    dZ = softmax * (dA - np.sum(dA * softmax, axis=0, keepdims=True))

    return dZ


def tanh_backward(dA, Z):
    return dA * (1 - np.power(np.tanh(Z), 2))


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev,W, b, activation):
    Z, linear_cache = linear_forward(A_prev,W, b)
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)
    elif activation == "tanh":
        A = tanh(Z)
    elif activation == "softmax":
        A= softmax(Z)
    else:
        A = Z
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
    elif activation=="softmax":
        dZ=softmax_backward(dA,Z)
    else:
        dZ=Z
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

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

    def L_model_forward(self, X):
        caches=[]
        A = X
        for l in range(1, self.L-1):
            A_prev = A
            A, cache = linear_activation_forward(A_prev,self.parameters['W' + str(l)], self.parameters['b' + str(l)], "relu")
            caches.append(cache)
        A_prev = A
        AL, cache = linear_activation_forward(A_prev,self.parameters['W' + str(self.L-1)], self.parameters['b' + str(self.L-1)], "softmax")
        caches.append(cache)
        return AL, caches

    def L_model_backward(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = caches[L-1]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache,"softmax" )
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp
        dZ=dA_prev_temp
        for i in reversed(range(L-1)):
            current_cache = caches[i]
            dZ, dW_temp, db_temp = linear_activation_backward(dZ, current_cache, "relu")
            grads["dW" + str(i + 1)] = dW_temp
            grads["db" + str(i + 1)] = db_temp
        return grads

    def update_parameters(self, grads, learning_rate):
        for l in range(1, self.L):
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    def fit(self, X, Y, learning_rate=0.01, num_iterations=3000, print_cost=False,Ifload=False):
        if Ifload:
            self.load_parameters('model_lab/model.pkl')
        self.initialize_parameters()
        for i in range(0, num_iterations):
            AL, caches = self.L_model_forward(X)
            cost = compute_cost(AL, Y)
            self.costs.append(cost)
            grads = self.L_model_backward(AL, Y, caches)
            self.update_parameters(grads, learning_rate)
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if print_cost and i % 100 == 0:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        return self.costs

    def predict_cost(self,X,Y):
        AL, caches = self.L_model_forward(X)
        cost = compute_cost(AL, Y)
        print("Cost after iteration {}: {}".format(10000, np.squeeze(cost)))

    def predict(self, X):
        # 使用模型参数进行前向传播并返回预测结果
        AL, _ = self.L_model_forward(X)
        print(AL.shape)
        # 对每个样本，选择概率最大的类别，并增加一个维度
        predictions = np.expand_dims(np.argmax(AL, axis=0), axis=0)
        print(predictions.shape)
        return predictions

    def accuracy(self, X, Y):
        predictions = self.predict(X)
        # 将OneHot编码的实际结果转换为类别标签
        labels = np.expand_dims(np.argmax(Y, axis=0), axis=0)
        accuracy = np.mean(predictions == labels)
        return accuracy

    def save_parameters(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.parameters, file)

    def load_parameters(self, filename):
        with open(filename, 'rb') as file:
            self.parameters = pickle.load(file)










