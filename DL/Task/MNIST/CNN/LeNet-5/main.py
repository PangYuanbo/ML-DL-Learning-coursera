import math
import numpy as np

import h5py
import scipy
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from DataReader import ReadMNIST
def MnistModule():
    model=tf.keras.models.Sequential([
        tfl.ZeroPadding2D(padding=(2,2),input_shape=(28,28,1)),
        tfl.Conv2D(6,(5,5),strides=(1,1),activation='sigmoid'),
        tfl.AvgPool2D(pool_size=(2,2),strides=(2,2)),
        tfl.Conv2D(16,(5,5),strides=(1,1),activation='sigmoid'),
        tfl.AvgPool2D(pool_size=(2,2),strides=(2,2)),
        tfl.Flatten(),
        tfl.Dense(120,activation='sigmoid'),
        tfl.Dense(84,activation='sigmoid'),
        tfl.Dense(10,activation='softmax')
    ])
    return model
mnistmodel=MnistModule()
mnistmodel.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
mnistmodel.summary()
Data = ReadMNIST(train_images_filepath='../../Data/train-images.idx3-ubyte',
                 train_labels_filepath='../../Data/train-labels.idx1-ubyte',
                 test_images_filepath='../../Data/t10k-images.idx3-ubyte', test_labels_filepath='../../Data/t10k-labels.idx1-ubyte')
Data.read_images_labels(images_filepath='../../Data/train-images.idx3-ubyte', labels_filepath='../../Data/train-labels.idx1-ubyte')
(x_train, y_train), (x_test, y_test) = Data.load_data()
mnistmodel.fit(x_train, y_train, epochs=10, batch_size=64)
mnistmodel.evaluate(x_test, y_test)