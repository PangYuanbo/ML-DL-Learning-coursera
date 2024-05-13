# This the class to reader the MNIST data
import numpy as np
class ReadMNIST(object):
    def __init__(self,train_images_filepath,train_labels_filepath,test_images_filepath,test_labels_filepath):
        self.train_images_filepath=train_images_filepath
        self.train_labels_filepath=train_labels_filepath
        self.test_images_filepath=test_images_filepath
        self.test_labels_filepath=test_labels_filepath


