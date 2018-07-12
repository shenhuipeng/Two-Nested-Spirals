# -*- coding : utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def show_data_without_border(data,label):
    plt.scatter(data[:, 0], data[:, 1], c=label[:, 0], s=30, lw=0)
    plt.show()

def show_data_with_border(data,label,border_x0, border_y0, border_x1, border_y1):
    plt.plot(border_x0, border_y0)
    plt.plot(border_x1, border_y1)
    plt.scatter(data[:, 0], data[:, 1], c=label[:, 0], s=30, lw=0)
    plt.show()

def show_training(loss_list,acc_list,acc,x,p):
    #plt.figure(figsize=(60, 1))
    plt.cla()
    plt.subplot(131)
    #plt.subplots(figsize=(1, 1))
    plt.plot(np.squeeze(loss_list))
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.subplot(132)
    #plt.subplots(figsize=(1, 1))
    plt.plot(np.squeeze(acc_list))
    plt.ylabel('accuracy')
    plt.xlabel('iterations')


    plt.subplot(133)
    #plt.subplots(figsize=(1, 1))
    plt.scatter(x[0, :], x[1, :], c=p[0, :], s=30, lw=0)
    plt.text(-0.4, 0.5, 'Accuracy=%.5f' % acc, fontdict={'size': 20, 'color': 'red'})
    plt.pause(0.001)


def show_test(x,p,acc,i,border_x0, border_y0, border_x1, border_y1):
    plt.plot(border_x0, border_y0)
    plt.plot(border_x1, border_y1)

    plt.scatter(x[0, :], x[1, :], c=p[0, :], s=30, lw=0)
    plt.text(-0.4, 0.5, "test %i Accuracy= %.5f" % (i, acc), fontdict={'size': 20, 'color': 'red'})
    plt.show()