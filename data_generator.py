# -*- coding : utf-8 -*-
import numpy as np
from plot_function import *

def get_border():
    """
    generate the border of the class
    :return border_x0,border_y0,border_x1,border_y1:
    """
    border_points = np.arange(0, 100)
    border_alpha1 = np.pi * (border_points) / 25
    border_beta = 0.4 * ((99 - border_points) / 99)
    border_x0 = 0. + border_beta * np.cos(border_alpha1)
    border_y0 = 0. + border_beta * np.sin(border_alpha1)
    border_x1 = 0. - border_beta * np.cos(border_alpha1)
    border_y1 = 0. - border_beta * np.sin(border_alpha1)
    return border_x0,border_y0,border_x1,border_y1

def get_data():
    """
    generate training data
    :return  data_label:   (x,y,label)
    """
    ######################################################
    studytra_num=100
    points = np.arange(0,studytra_num,4)
    alpha1=np.pi*(points)/25
    beta=0.4*((106-points)/99)
    x0=0.+beta * np.cos(alpha1)
    y0=0.+beta * np.sin(alpha1)
    z0=np.zeros((points.shape[0],1))
    x1=0.-beta*np.cos(alpha1)
    y1=0.-beta*np.sin(alpha1)
    z1=np.ones((points.shape[0],1))

    x0 = x0[:,np.newaxis]
    x1 = x1[:,np.newaxis]
    y0 = y0[:,np.newaxis]
    y1 = y1[:,np.newaxis]

    data0 = np.concatenate((x0,y0),axis=1)
    data1 = np.concatenate((x1,y1),axis=1)
    data = np.concatenate((data0,data1),axis=0)

    label = np.concatenate((z0,z1),axis=0)

    for  i in range(14):
        if 106+i ==110:
            continue
        for j in range(-6,4):
            alpha1 = np.pi * (points+j/5*4) / 25
            beta = 0.4 * ((107+i - points) / 99)  # 155
            x0 = 0. + beta * np.cos(alpha1)
            y0 = 0. + beta * np.sin(alpha1)
            z0 = np.zeros((points.shape[0], 1))
            x1 = 0. - beta * np.cos(alpha1)
            y1 = 0. - beta * np.sin(alpha1)
            z1 = np.ones((points.shape[0], 1))

            x0 = x0[:, np.newaxis]
            x1 = x1[:, np.newaxis]
            y0 = y0[:, np.newaxis]
            y1 = y1[:, np.newaxis]

            data0 = np.concatenate((x0, y0), axis=1)
            data1 = np.concatenate((x1, y1), axis=1)
            data = np.concatenate((data, data0,data1,), axis=0)

            label = np.concatenate((label, z0,z1), axis=0)
    border_x0, border_y0, border_x1, border_y1 = get_border()
    show_data_with_border(data, label,border_x0, border_y0, border_x1, border_y1)
    data_label = np.concatenate((data,label), axis=1)

    print(data_label.shape)

    return data_label


def add_noise(X):
    """
    add noise to points
    :param X: (size of points, num of points)
    :return X+noise:
    """
    r,c = X.shape
    mu, sigma = 0, 0.01
    noise = np.random.normal(mu, sigma, (r, c))

    return X+noise

def get_test():
    """
    generate test data randomly in the border
    :return  data_label:   (x,y,label)
    """
    tmp = np.random.randint(110,115)
    step = np.random.randint(1,5)
    noise1 = (np.random.rand()-0.5)/20
    noise2 = (np.random.rand()-0.5)/20
    studytra_num = 100
    points = np.arange(0, studytra_num, step)
    alpha1 = np.pi * (points) / 25
    beta = 0.4 * ((tmp - points) / 99)
    x0 = 0.+ noise1 + beta * np.cos(alpha1)
    y0 = 0.+ noise2 + beta * np.sin(alpha1)
    z0 = np.zeros((points.shape[0], 1))
    x1 = 0. - beta * np.cos(alpha1)
    y1 = 0. - beta * np.sin(alpha1)
    z1 = np.ones((points.shape[0], 1))

    x0 = x0[:, np.newaxis]
    x1 = x1[:, np.newaxis]
    y0 = y0[:, np.newaxis]
    y1 = y1[:, np.newaxis]

    data0 = np.concatenate((x0, y0), axis=1)
    data1 = np.concatenate((x1, y1), axis=1)

    data0 = add_noise(data0)
    data1 = add_noise(data1)
    data = np.concatenate((data0, data1), axis=0)

    label = np.concatenate((z0, z1), axis=0)

    data_label = np.concatenate((data, label), axis=1)

    print(data_label.shape)

    return data_label


