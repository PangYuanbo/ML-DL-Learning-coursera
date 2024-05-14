# This the class to reader the MNIST data
import numpy as np
import struct
import os
from array import array
class ReadMNIST(object):
    def __init__(self,train_images_filepath,train_labels_filepath,test_images_filepath,test_labels_filepath):
        self.train_images_filepath=train_images_filepath
        self.train_labels_filepath=train_labels_filepath
        self.test_images_filepath=test_images_filepath
        self.test_labels_filepath=test_labels_filepath
    def read_images_labels(self,images_filepath,labels_filepath):
        labels=[]
        with open(labels_filepath,'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic!=2049:
                raise ValueError('Magic number mismatch, expected 2049 but got {}'.format(magic))
            labels=array('B',file.read())
        with open(images_filepath,'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic!=2051:
                raise ValueError('Magic number mismatch, expected 2051 but got {}'.format(magic))
            image_data=array('B',file.read())



