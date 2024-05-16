# This the class to reader the MNIST data
import platform


if platform.system() == 'Windows':
    import cupy as np
elif platform.system() == 'Linux':
    import numpy as np
elif platform.system() == 'Darwin':
    import numpy as np


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
        images = np.array(image_data).reshape(size, rows, cols)
        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.train_images_filepath, self.train_labels_filepath)
        x_train = self.np_change(x_train)
        y_train = self.OneHot(y_train)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        x_test = self.np_change(x_test)
        y_test = self.OneHot(y_test)
        return (x_train, y_train), (x_test, y_test)

    def np_change(self, x):
        np_x = np.array(x)
        np_x = np_x.reshape(np_x.shape[0], -1).T
        np_x = np_x / 255
        return np_x

    def OneHot(self, y):
        y = np.array(y)
        num_classes = y.max().item() + 1
        num_samples = y.size
        onehot = np.zeros((num_classes, num_samples))
        onehot[y, np.arange(num_samples)] = 1
        return onehot

#
# import cupy as np
# import struct
# from array import array
#
#
# class ReadMNIST(object):
#     def __init__(self, train_images_filepath, train_labels_filepath, test_images_filepath, test_labels_filepath):
#         self.train_images_filepath = train_images_filepath
#         self.train_labels_filepath = train_labels_filepath
#         self.test_images_filepath = test_images_filepath
#         self.test_labels_filepath = test_labels_filepath
#
#     def read_images_labels(self, images_filepath, labels_filepath):
#         with open(labels_filepath, 'rb') as file:
#             magic, size = struct.unpack(">II", file.read(8))
#             if magic != 2049:
#                 raise ValueError(f'Magic number mismatch, expected 2049 but got {magic}')
#             labels = array('B', file.read())
#
#         with open(images_filepath, 'rb') as file:
#             magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
#             if magic != 2051:
#                 raise ValueError(f'Magic number mismatch, expected 2051 but got {magic}')
#             image_data = array('B', file.read())
#
#         images = np.array(image_data).reshape(size, rows, cols)
#         return images, np.array(labels)
#
#     def load_data(self):
#         x_train, y_train = self.read_images_labels(self.train_images_filepath, self.train_labels_filepath)
#         x_train = self.np_change(x_train)
#         y_train = self.OneHot(y_train)
#
#         x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
#         x_test = self.np_change(x_test)
#         y_test = self.OneHot(y_test)
#
#         return (x_train, y_train), (x_test, y_test)
#
#     def np_change(self, x):
#         np_x = np.array(x)
#         np_x = np_x.reshape(np_x.shape[0], -1).T
#         np_x = np_x / 255.0
#         return np_x
#

