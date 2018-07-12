# -*- coding : utf-8 -*-

import numpy as np
from FClayer import *
from lossFunction import *
import matplotlib.pyplot as plt

class five_layer_model():
    def __init__(self, learning_rate ):
        self.layer1 = one_FC_layer(2, 10, "relu", learning_rate)
        self.layer2 = one_FC_layer(10, 30, "relu", learning_rate)
        self.layer3 = one_FC_layer(30, 30, "relu", learning_rate)
        self.layer4 = one_FC_layer(30, 10, "relu", learning_rate)
        self.layer5 = one_FC_layer(10, 1, "sigmoid", learning_rate)

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
        A5 = self.layer5.FC_forward(A4)

        return A5

    def get_loss(self,A, Y):
        return compute_loss_with_L2_regularization(A, Y, 0.0001,
                                            self.layer1.W,self.layer2.W,self.layer3.W,self.layer4.W,self.layer5.W )

    def backward(self,Y):
        dA5 = - (np.divide(Y, self.layer5.Atmp) - np.divide(1 - Y, 1 - self.layer5.Atmp))
        dA4 = self.layer5.FC_backward_with_regularization(dA5,0.0001)
        dA3 = self.layer4.FC_backward_with_regularization(dA4,0.0001)
        dA2 = self.layer3.FC_backward_with_regularization(dA3,0.0001)
        dA1 = self.layer2.FC_backward_with_regularization(dA2,0.0001)
        dA0 = self.layer1.FC_backward_with_regularization(dA1,0.0001)

    def update(self):
        self.layer5.FC_update_basic()
        self.layer4.FC_update_basic()
        self.layer3.FC_update_basic()
        self.layer2.FC_update_basic()
        self.layer1.FC_update_basic()


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


data_num_per_class = 300
n_data = np.ones((data_num_per_class, 2))
n_data0 = n_data
n_data0[:,0] = 1
n_data0[:,1] = 1
x0 = np.random.normal(1*n_data0, 1)
y0 = np.zeros((data_num_per_class,1))


n_data1 = n_data
n_data1[:,0] = -1
n_data1[:,1] = -1
x1 = np.random.normal(1*n_data1, 1)
y1 = np.ones((data_num_per_class,1))

x = np.concatenate((x0, x1), axis=0)
y = np.concatenate((y0, y1), axis=0)

data = np.concatenate((x, y), axis=1)

np.random.shuffle(data)

plt.scatter(data[:, 0], data[:, 1], c=data[:,2], s=30, lw=0)
plt.show()
print(data)
print("data shape", data.shape)


model = five_layer_model(0.0085)
loss_list = []
x= data[:,0:2]
y = data[:,2]
y = y[:,np.newaxis]

x = x.T
y = y.T

print(x)
plt.ion()   # 画图
plt.show()
for i  in range(200):
    output = model.forward(x)
    p = model.predict(output)
    acc = model.get_accuracy(p,y)
    loss = model.get_loss(output, y)
    loss_list.append(loss)
    model.backward(y)
    model.update()
    plt.cla()
    plt.subplot(121)
    plt.plot(np.squeeze(loss_list))
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.subplot(122)
    plt.scatter(x[0,:], x[1,:], c=p[0,:], s=30, lw=0)
    plt.text(1.5, -4, 'Accuracy=%.2f' % acc, fontdict={'size': 20, 'color': 'red'})
    plt.pause(0.001)

plt.ioff()  # 停止画图
plt.show()




