from DataReader import ReadMNIST
import time

from Regression import Model

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
#
# def show_images(images, title_texts):
#     cols = 5
#     rows = int(len(images) / cols) + 1
#     plt.figure(figsize=(30, 20))
#     index = 1
#     for x in zip(images, title_texts):
#         image = x[0]
#         title_text = x[1]
#         plt.subplot(rows, cols, index)
#         plt.imshow(image, cmap=plt.cm.gray)
#         if title_text != '':
#             plt.title(title_text, fontsize=15)
#         index += 1
#     plt.show()print('Time taken: ',time.time()-t)
#

# Load the MNIST dataset
Data = ReadMNIST(train_images_filepath='Data/train-images.idx3-ubyte',
                 train_labels_filepath='Data/train-labels.idx1-ubyte',
                 test_images_filepath='Data/t10k-images.idx3-ubyte', test_labels_filepath='Data/t10k-labels.idx1-ubyte')
(x_train, y_train), (x_test, y_test) = Data.load_data()

#
# Show some random training and test images
#
# images_2_show = []
# titles_2_show = []
# for i in range(0, 10):
#     r = random.randint(1, 60000)
#     images_2_show.append(x_train[r])
#     titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))
#
# for i in range(0, 5):
#     r = random.randint(1, 10000)
#     images_2_show.append(x_test[r])
#     titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))
#
# show_images(images_2_show, titles_2_show)

import csv

# 定义一个函数来训练模型并返回成本

import csv


# 定义一个函数来训练模型并返回成本
def train_model(layers_dims, x_train, y_train, learning_rate, num_iterations):
    model = Model(layers_dims)
    costs = model.fit(x_train, y_train, learning_rate=learning_rate, num_iterations=num_iterations, print_cost=True)
    return costs


# 测试不同的隐藏层大小
hidden_layer_sizes = []
for i in range(16, 128,8):
    for j in range(128, 256,8):
        hidden_layer_sizes.append([784, i, j, 10])
learning_rate = 0.1
num_iterations = 1000

all_costs = []

# 训练模型并保存成本
for layers_dims in hidden_layer_sizes:
    t = time.time()
    costs = train_model(layers_dims, x_train, y_train, learning_rate, num_iterations)
    all_costs.append(costs)
    print('Time taken: ', time.time() - t)

# 将成本值导出到CSV文件
for i, costs in enumerate(all_costs):
    filename = f'costs_hidden_layers_{hidden_layer_sizes[i][1]}_{hidden_layer_sizes[i][2]}.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'Cost'])
        for j, cost in enumerate(costs):
            writer.writerow([j, cost])

print("Cost values have been saved to CSV files.")

# loading model
t = time.time()
layers_dims = [784, 256, 64, 10]
model = Model(layers_dims)
costs = model.fit(x_train, y_train, learning_rate=0.1, num_iterations=200, print_cost=True)
# plt.plot(costs)
# plt.xlabel('Iteration')
# plt.ylabel('Cost')
# plt.title('Cost reduction over iterations')
# plt.show()
model.save_parameters('model.pkl')
print('Time taken: ', time.time() - t)
print("test_accuracy", model.accuracy(x_test, y_test))
