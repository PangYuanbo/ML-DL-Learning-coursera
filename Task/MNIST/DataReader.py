# This the class to reader the MNIST data
import numpy as np
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