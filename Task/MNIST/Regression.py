# the model for MNIST regression task
import numpy as np
class model(object):
    def __init__(self,layers_dims):
        self.parameters = {}
        self.costs = None
        self.layers_dims = layers_dims
        self.L = len(layers_dims)
    def initialize_parameters(self):
        np.random.seed(3)
        for l in range(1, self.L):
            self.parameters['W' + str(l)] = np.random.randn(self.layers_dims[l], self.layers_dims[l - 1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((self.layers_dims[l], 1))
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))
    def relu(self,Z):
        return np.maximum(0,Z)
    def tanh(self,Z):
        return np.tanh(Z)
    def sigmoid_backward(self,dA, Z):
        s = self.sigmoid(Z)
        return dA * s * (1 - s)
    def relu_backward(self,dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
    def tanh_backward(self,dA, Z):
        return dA * (1 - np.power(np.tanh(Z), 2))
