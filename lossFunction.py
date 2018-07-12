# -*- coding : utf-8 -*-

import numpy as np

#  cross entropy loss
def compute_loss(A_L, Y):
    """
    :param A_L: probability vector corresponding to your label predictions, shape (1, number of examples)
    :param Y: true "label" vector , shape (1, number of examples)
    :return loss:
    """
    m = Y.shape[1]
    loss = -1. / m * np.sum(np.multiply(np.log(A_L), Y) + np.multiply(np.log(1 - A_L), 1 - Y))
    loss = np.squeeze(loss)  # To make sure your cost's shape is what we expect (e.g. this turns [[10]] into 10).
    assert (loss.shape == ())

    return loss

def compute_loss_with_L2_regularization(A_L, Y, lambd, W1,W2,W3,W4):
    """
    :param A_L:
    :param Y:
    :param lambd:
    :param W1:
    :param W2:
    :param W3:
    :param W4:
    :return cost:  loss with L2  regularization
    """
    cross_entropy_cost = compute_loss(A_L, Y)
    m = Y.shape[1]
    l2_regularization_cost = lambd / (2 * m) * (np.sum(np.square(W1))  + np.sum(np.square(W2)) +
                                                np.sum(np.square(W3)) + np.sum(np.square(W4)))

    cost = cross_entropy_cost + l2_regularization_cost

    return cost