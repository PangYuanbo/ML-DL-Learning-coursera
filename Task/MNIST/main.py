from DataReader import ReadMNIST
import numpy as np
import matplotlib.pyplot as plt
import os
import random

Data = ReadMNIST(train_images_filepath='Data/train-images.idx3-ubyte',
                 train_labels_filepath='Data/train-labels.idx1-ubyte',
                 test_images_filepath='Data/t10k-images.idx3-ubyte', test_labels_filepath='Data/t10k-labels.idx1-ubyte')
Data.read_images_labels(images_filepath='Data/train-images.idx3-ubyte', labels_filepath='Data/train-labels.idx1-ubyte')


#
# Verify Reading Dataset via MnistDataloader class
#


#
# Set file paths based on added MNIST Datasets
#

def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize=15)
        index += 1
    plt.show()


#Load the MNIST data
Data = ReadMNIST(train_images_filepath='Data/train-images.idx3-ubyte',
                 train_labels_filepath='Data/train-labels.idx1-ubyte',
                 test_images_filepath='Data/t10k-images.idx3-ubyte', test_labels_filepath='Data/t10k-labels.idx1-ubyte')
(x_train, y_train), (x_test, y_test) = Data.load_data()

#
# Show some random training and test images
#
images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

show_images(images_2_show, titles_2_show)

