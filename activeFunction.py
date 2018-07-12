# -*- coding : utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(Z):
    """
    :param Z:    a numpy array with any shape
    :return A:   sigmoid(Z) the size of A is  same as  Z
    """
    A = 1/(1+np.exp(-Z))
    tmp = Z
    return A, tmp

def relu(Z):
    """
    :param Z: a numpy array with any shape
    :return A:  relu(Z) the size of A is  same as  Z
    :return tmp:   the same with Z
    """
    A = np.maximum(0, Z)
    tmp  = Z
    return A, tmp

def sigmoid_backward(dA, tmp):
    """
    :param dA: backward  gradient from post layer, in  any shape
    :param tmp:  tmp of Z
    :return dZ:  gradient of  Z
    """
    Z = tmp
    tmp1 = 1/(1+np.exp(-Z))
    dZ = dA * tmp1 * (1 - tmp1)

    return dZ

def relu_backward(dA, tmp):
    """
    :param dA: backward  gradient from post layer, in  any shape
    :param tmp:  tmp of Z
    :return dZ:  gradient of  Z
    """
    Z = tmp
    dZ = np.array(dA, copy=True)
    # when z <= 0, the gradient is zero .
    dZ[Z <= 0] = 0

    return dZ
