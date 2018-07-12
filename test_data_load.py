# -*- coding : utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt('test_data1.txt')
print(a)
print("a shape", a.shape)

a[:,2] = np.round(a[:,2])
print(a)

plt.scatter(a[:, 0], a[:, 1], c=a[:,2], s=30, lw=0)
plt.show()