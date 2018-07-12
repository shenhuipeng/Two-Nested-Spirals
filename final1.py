# -*- coding : utf-8 -*-

import numpy as np
from FClayer import *
from lossFunction import *
from data_generator import *
from plot_function import *
import matplotlib.pyplot as plt
import math

class four_layer_model():
    def __init__(self, learning_rate ):
        self.layer1 = one_FC_layer(2, 10, "relu")
        self.layer2 = one_FC_layer(10, 20, "relu")
        self.layer3 = one_FC_layer(20, 10, "relu")
        self.layer4 = one_FC_layer(10, 1, "sigmoid")
        self.learning_rate = learning_rate
    def forward(self,X):
        """
        :param X:  input (input_dim,sample num)
        :return:
        """
        A = X
        A1 = self.layer1.FC_forward(A)
        A2 = self.layer2.FC_forward(A1)
        A3 = self.layer3.FC_forward(A2)
        A4 = self.layer4.FC_forward(A3)

        return A4

    def get_loss(self,A, Y):
        return compute_loss_with_L2_regularization(A, Y, 0.00001,
                                            self.layer1.W,self.layer2.W,self.layer3.W,self.layer4.W )

    def backward(self,Y):
        dA4 = - (np.divide(Y, self.layer4.Atmp) - np.divide(1 - Y, 1 - self.layer4.Atmp))

        dA3 = self.layer4.FC_backward_with_regularization(dA4,0.0001)
        dA2 = self.layer3.FC_backward_with_regularization(dA3,0.0001)
        dA1 = self.layer2.FC_backward_with_regularization(dA2,0.0001)
        dA0 = self.layer1.FC_backward_with_regularization(dA1,0.0001)

    def update(self):
        # self.layer4.FC_update_with_momentum(0.95,self.learning_rate)
        # self.layer3.FC_update_with_momentum(0.95,self.learning_rate)
        # self.layer2.FC_update_with_momentum(0.95,self.learning_rate)
        # self.layer1.FC_update_with_momentum(0.95,self.learning_rate)

        self.layer4.FC_update_basic(self.learning_rate)
        self.layer3.FC_update_basic(self.learning_rate)
        self.layer2.FC_update_basic(self.learning_rate)
        self.layer1.FC_update_basic(self.learning_rate)

    def predict(self,A):
        assert(A.shape[0] == 1)
        m = A.shape[1]
        p = np.zeros((1, m))

        for i in range(0,m):
            if A[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        return p

    def get_accuracy(self,p,Y):
        m = Y.shape[1]
        accuracy = np.sum(p == Y) / m
        print("Accuracy: " + str(accuracy))
        return accuracy

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    batch_nums = math.floor(m/mini_batch_size)
    for k in range(0, batch_nums):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, batch_nums * mini_batch_size : ]
        mini_batch_Y = shuffled_Y[:, batch_nums * mini_batch_size : ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

data_label= get_data()
np.random.shuffle(data_label)

x= data_label[:,0:2]
y = data_label[:,2]
y = y[:,np.newaxis]

show_data_without_border(x,y)

model = four_layer_model(0.08)
loss_list = []
acc_list = []
x = x.T
y = y.T


plt.ion()   # 画图
plt.show()

epochs = 200
seed = 0
mini_batch_size = 64
for i  in range(epochs):

    if i == epochs//4*3 :
        model.learning_rate = model.learning_rate / 4

    seed = seed + 1
    minibatches = random_mini_batches(x, y, mini_batch_size, seed)

    for index, minibatch in enumerate(minibatches):
        (minibatch_X, minibatch_Y) = minibatch
        output = model.forward(minibatch_X)
        model.backward(minibatch_Y)
        model.update()

    output = model.forward(x)
    p = model.predict(output)
    acc = model.get_accuracy(p, y)
    acc_list.append(acc)
    loss = model.get_loss(output, y)
    loss_list.append(loss)


    show_training(loss_list,acc_list,acc,x,p)

plt.ioff()  # 停止画图
plt.show()

for i in range(10):

    test_data = get_test()
    x= test_data[:,0:2]
    y = test_data[:,2]
    y = y[:,np.newaxis]

    x = x.T
    y = y.T

    output = model.forward(x)
    p = model.predict(output)
    acc = model.get_accuracy(p,y)

    border_x0, border_y0, border_x1, border_y1 = get_border()
    show_test(x,p,acc,i,border_x0, border_y0, border_x1, border_y1)







