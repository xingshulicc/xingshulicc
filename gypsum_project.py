# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Mon Nov 26 09:48:46 2018

@author: xingshuli
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import AveragePooling1D
from keras import backend as K


file_name = 'gypsum_project.xlsx'
file_path = os.path.join(os.getcwd(), file_name)

#load data
cement_100 = pd.read_excel(file_path, skiprows = 2, usecols = 'B')
gypsum_3 = pd.read_excel(file_path, skiprows = 2, usecols = 'F')
gypsum_5 = pd.read_excel(file_path, skiprows = 2, usecols = 'J')
gypsum_10 = pd.read_excel(file_path, skiprows = 2, usecols = 'N')
gypsum_20 = pd.read_excel(file_path, skiprows = 2, usecols = 'R')
gypsum_100 = pd.read_excel(file_path, skiprows = 2, usecols = 'V')

x_1 = np.arange(1, 334, dtype = int)
x_2 = np.arange(1, 335, dtype = int)
x_3 = np.arange(1, 345, dtype = int)

plt.plot(x_1, cement_100, color = 'green', label = 'cement_100')
plt.plot(x_2, gypsum_3, color = 'red', label = 'gypsum_3')
plt.plot(x_1, gypsum_5, color = 'blue', label = 'gypsum_5')
plt.plot(x_2, gypsum_10, color = 'orange', label = 'gypsum_10')
plt.plot(x_3, gypsum_20, color = 'purple', label = 'gypsum_20')
plt.plot(x_1, gypsum_100, color = 'brown', label = 'gypsum_100')
plt.legend()


plt.xlabel('Number of Measurements')
plt.ylabel('Radon')
plt.grid()
plt.show()

#convert dataframe to keras tensor
gypsum_3 = np.array(gypsum_3, dtype = float)
gypsum_3 = np.expand_dims(gypsum_3, axis = 0)
Gypsum_3 = K.variable(gypsum_3)

gypsum_5 = np.array(gypsum_5, dtype = float)
gypsum_5 = np.expand_dims(gypsum_5, axis = 0)
Gypsum_5 = K.variable(gypsum_5)

gypsum_10 = np.array(gypsum_10, dtype = float)
gypsum_10 = np.expand_dims(gypsum_10, axis = 0)
Gypsum_10 = K.variable(gypsum_10)

gypsum_20 = np.array(gypsum_20, dtype = float)
gypsum_20 = np.expand_dims(gypsum_20, axis = 0)
Gypsum_20 = K.variable(gypsum_20)

gypsum_100 = np.array(gypsum_100, dtype = float)
gypsum_100 = np.expand_dims(gypsum_100, axis = 0)
Gypsum_100 = K.variable(gypsum_100)

cement_100 = np.array(cement_100, dtype = float)
cement_100 = np.expand_dims(cement_100, axis = 0)
Cement_100 = K.variable(cement_100)

#define hyper-parameters for window function
pool_size = 30
strides = 50
padding = 'valid'


Gypsum_3 = AveragePooling1D(pool_size = pool_size, 
                            strides = strides, 
                            padding = padding)(Gypsum_3)
Gypsum_5 = AveragePooling1D(pool_size = pool_size, 
                            strides = strides, 
                            padding = padding)(Gypsum_5)
Gypsum_10 = AveragePooling1D(pool_size = pool_size, 
                            strides = strides, 
                            padding = padding)(Gypsum_10)
Gypsum_20 = AveragePooling1D(pool_size = pool_size, 
                            strides = strides, 
                            padding = padding)(Gypsum_20)
Gypsum_100 = AveragePooling1D(pool_size = pool_size, 
                            strides = strides, 
                            padding = padding)(Gypsum_100)
Cement_100 = AveragePooling1D(pool_size = pool_size, 
                            strides = strides, 
                            padding = padding)(Cement_100)

#convert keras tensor to numpy array
array_g3 = K.eval(Gypsum_3)
array_g3 = np.reshape(array_g3, -1)

array_g5 = K.eval(Gypsum_5)
array_g5 = np.reshape(array_g5, -1)

array_g10 = K.eval(Gypsum_10)
array_g10 = np.reshape(array_g10, -1)

array_g20 = K.eval(Gypsum_20)
array_g20 = np.reshape(array_g20, -1)

array_g100 = K.eval(Gypsum_100)
array_g100 = np.reshape(array_g100, -1)

array_c100 = K.eval(Cement_100)
array_c100 = np.reshape(array_c100, -1)

x_4 = np.arange(1, len(array_c100) + 1, dtype = int)
x_5 = np.arange(1, len(array_g20) + 1, dtype = int)

plt.plot(x_4, array_c100, color = 'green', label = 'cement_100')
plt.plot(x_4, array_g3, color = 'red', label = 'gypsum_3')
plt.plot(x_4, array_g5, color = 'blue', label = 'gypsum_5')
plt.plot(x_4, array_g10, color = 'orange', label = 'gypsum_10')
plt.plot(x_5, array_g20, color = 'purple', label = 'gypsum_20')
#plt.plot(x_4, array_g100, color = 'brown', label = 'gypsum_100')
plt.legend()
plt.figure(1)

plt.xlabel('Number of Measurements')
plt.ylabel('Radon')
plt.grid()
plt.show()



#TODO: Classification and Prediction


















