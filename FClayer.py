# -*- coding : utf-8 -*-

import numpy as np
from activeFunction import *

class one_FC_layer():
    """
    self.W : weights,  numpy array of shape (size of current layer, size of previous layer)
    self.b : bias, numpy array of shape (size of the current layer, 1)
    """
    def __init__(self,p_nodes_num, c_nodes_num, A_name):
        self.p_nodes_num = p_nodes_num
        self.c_nodes_num = c_nodes_num
        self.W = np.random.randn(c_nodes_num, p_nodes_num) / np.sqrt(p_nodes_num)
        self.b = np.zeros((c_nodes_num,1))
        self.A_name = A_name
        self.VdW = np.zeros(self.W.shape)
        self.Vdb = np.zeros(self.b.shape)
        self.dW = None
        self.db = None
        self.Ztmp = None
        self.Atmp = None
        self.A_prev_tmp = None

    def FC_forward(self,A_prev):
        """
        :param A_prev: activations from previous layer (or input data): (size of previous layer, number of examples)
        :return A:  the output of the FC layer with the activation function
         (size of current layer, number of examples)
        """
        Z = np.dot(self.W, A_prev) + self.b
        self.A_prev_tmp = A_prev
        if self.A_name == "sigmoid":
            A, Ztmp= sigmoid(Z)
        elif self.A_name == "relu":
            A, Ztmp = relu(Z)
        else:
            exit("activation name error")
        self.Ztmp = Ztmp
        self.Atmp = A
        return A

    def FC_backward(self,dA):
        """
        :param dA:
        :return dA_prev:
        """
        if self.A_name == "sigmoid":
            dZ = sigmoid_backward(dA, self.Ztmp)
        elif self.A_name == "relu":
            dZ = relu_backward(dA, self.Ztmp)
        else:
            exit("activation name error")

        m = self.A_prev_tmp.shape[1]
        self.dW = 1. / m * np.dot(dZ, self.A_prev_tmp.T)
        self.db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)

        return dA_prev

    def FC_backward_with_regularization(self,dA,lambd):
        """
        :param dA:
        :return dA_prev:
        :return dW:
        :return db:
        """
        if self.A_name == "sigmoid":
            dZ = sigmoid_backward(dA, self.Ztmp)
        elif self.A_name == "relu":
            dZ = relu_backward(dA, self.Ztmp)
        else:
            exit("activation name error")

        m = self.A_prev_tmp.shape[1]
        self.dW = 1. / m * np.dot(dZ, self.A_prev_tmp.T) + lambd/m * self.W
        self.db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)

        return dA_prev

    def FC_update_basic(self,learning_rate):
        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db

    def FC_update_with_momentum(self, beta,learning_rate):
        self.VdW = beta * self.VdW + (1 - beta) * self.dW
        self.Vdb = beta * self.Vdb + (1 - beta) * self.db

        self.W = self.W - learning_rate * self.VdW
        self.b = self.b - learning_rate * self.Vdb





