import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import struct
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
        images = np.array(image_data).reshape(size, rows, cols,1)
        return images, labels

    def load_data(self,):
        x_train, y_train = self.read_images_labels(self.train_images_filepath, self.train_labels_filepath)
        x_train = x_train/255
        y_train = self.OneHot(y_train)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        x_test = x_test/255
        y_test = self.OneHot(y_test)
        data = (x_train, y_train), (x_test, y_test)
        return data

    def OneHot(self, y):
        y = np.array(y)
        num_classes = y.max().item() + 1
        num_samples = y.size
        onehot = np.zeros(( num_samples,num_classes))
        onehot[np.arange(num_samples),y] = 1
        print('onehot:',onehot.shape)
        return onehot
