# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Fri Aug 30 16:34:59 2019

@author: Admin
"""
#import numpy as np
import os
import matplotlib.pyplot as plt
epochs = 7001 #here, the epoch should add 1 


file_1 = 'train_loss.txt'
file_2 = 'val_loss.txt'

file_dic_path = os.path.join(os.getcwd(), 'gypsum_model')
file_1_path = os.path.join(file_dic_path, file_1)
file_2_path = os.path.join(file_dic_path, file_2)

mat_list_train_loss = []
with open(file_1_path,'r') as f1:
    data1 = f1.readlines()
    for line in data1:
        odom = line.strip('[]\n').split(',')
        num_float = list(map(float, odom))
        mat_list_train_loss.append(num_float)
f1.close()
train_loss_axis = sum(mat_list_train_loss, [])

mat_list_val_loss = []
with open(file_2_path,'r') as f3:
    data3 = f3.readlines()
    for line in data3:
        odom = line.strip('[]\n').split(',')
        num_float = list(map(float, odom))
        mat_list_val_loss.append(num_float)
f3.close()
val_loss_axis = sum(mat_list_val_loss, [])

x_axis = range(1,epochs)

plt.plot(x_axis, train_loss_axis, 'r-.', label = "train_loss")
plt.plot(x_axis, val_loss_axis, 'b-.', label = "val_loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc = 'best')
plt.grid(linestyle = '-.', c = 'k')
plt.show()
