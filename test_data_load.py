# -*- coding : utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
#
# a = np.loadtxt('test_data1.txt')
# print(a)
# print("a shape", a.shape)
#
# a[:,2] = np.round(a[:,2])
# print(a)
#
# plt.scatter(a[:, 0], a[:, 1], c=a[:,2], s=30, lw=0)
# plt.show()

##################################################
# studytra_num=100
# points = np.arange(0,studytra_num,4)
# alpha1=np.pi*(points)/25
# beta=0.4*((106-points)/99)
# x0=0.+beta * np.cos(alpha1)
# y0=0.+beta * np.sin(alpha1)
# z0=np.zeros((points.shape[0],1))
# x1=0.-beta*np.cos(alpha1)
# y1=0.-beta*np.sin(alpha1)
# z1=np.ones((points.shape[0],1))
#
# x0 = x0[:,np.newaxis]
# x1 = x1[:,np.newaxis]
# y0 = y0[:,np.newaxis]
# y1 = y1[:,np.newaxis]
#
# data0 = np.concatenate((x0,y0),axis=1)
# data1 = np.concatenate((x1,y1),axis=1)
# data = np.concatenate((data0,data1),axis=0)
#
# label = np.concatenate((z0,z1),axis=0)
#
# plt.scatter(data0[:, 0], data0[:, 1], c=z0[:, 0], s=30, lw=0)
# plt.show()


###########################################
from data_generator import *
from plot_function import *
border_x0, border_y0, border_x1, border_y1 = get_border()
plt.plot(border_x0, border_y0)
plt.plot(border_x1, border_y1)

test_data = get_test()
x= test_data[:,0:2]
y = test_data[:,2]
y = y[:,np.newaxis]
plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], s=30, lw=0)
plt.show()

############################################################

