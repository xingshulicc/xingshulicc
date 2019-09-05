# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Thu Aug 22 14:35:08 2019

@author: Admin
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import AveragePooling1D
from keras import backend as K

file_folder = os.path.join(os.getcwd(), 'gypsum_project')

#file_name = 'gypsum_6.xlsx'
#file_name = 'gypsum_7.xlsx'
file_name = 'gypsum_8.xlsx'
#file_name = 'gypsum_9.xlsx'
#file_name = 'gypsum_10.xlsx'
file_path = os.path.join(file_folder, file_name)

#load data
cement_100 = pd.read_excel(file_path, skiprows = 2, usecols = 'B')
gypsum_3 = pd.read_excel(file_path, skiprows = 2, usecols = 'F')
gypsum_5 = pd.read_excel(file_path, skiprows = 2, usecols = 'J')
gypsum_10 = pd.read_excel(file_path, skiprows = 2, usecols = 'N')
gypsum_20 = pd.read_excel(file_path, skiprows = 2, usecols = 'R')
gypsum_100 = pd.read_excel(file_path, skiprows = 2, usecols = 'V')

length = len(cement_100)
x_range = np.arange(1, length + 1, dtype = int)


plt.plot(x_range, cement_100, color = 'green', label = 'cement_100')
plt.plot(x_range, gypsum_3, color = 'red', label = 'gypsum_3')
plt.plot(x_range, gypsum_5, color = 'blue', label = 'gypsum_5')
plt.plot(x_range, gypsum_10, color = 'orange', label = 'gypsum_10')
plt.plot(x_range, gypsum_20, color = 'purple', label = 'gypsum_20')
#plt.plot(x_range, gypsum_100, color = 'brown', label = 'gypsum_100')

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
pool_size = 15
strides = 15
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
array_g3 = np.reshape(array_g3, (-1, 1))

array_g5 = K.eval(Gypsum_5)
array_g5 = np.reshape(array_g5, (-1, 1))

array_g10 = K.eval(Gypsum_10)
array_g10 = np.reshape(array_g10, (-1, 1))

array_g20 = K.eval(Gypsum_20)
array_g20 = np.reshape(array_g20, (-1, 1))

array_g100 = K.eval(Gypsum_100)
array_g100 = np.reshape(array_g100, (-1, 1))

array_c100 = K.eval(Cement_100)
array_c100 = np.reshape(array_c100, (-1, 1))

x_4 = np.arange(1, len(array_c100) + 1, dtype = int)
x_5 = np.arange(1, len(array_g20) + 1, dtype = int)

plt.plot(x_4, array_c100, color = 'green', marker = 'o', label = 'cement_100')
plt.plot(x_4, array_g3, color = 'red', marker = 'o', label = 'gypsum_3')
plt.plot(x_4, array_g5, color = 'blue', marker = 'o', label = 'gypsum_5')
plt.plot(x_4, array_g10, color = 'orange', marker = 'o', label = 'gypsum_10')
plt.plot(x_5, array_g20, color = 'purple', marker = 'o', label = 'gypsum_20')
#plt.plot(x_4, array_g100, color = 'brown', label = 'gypsum_100')
plt.legend()
plt.figure(1)

plt.xlabel('Number of Measurements')
plt.ylabel('Radon')
plt.grid()
plt.show()

#load humidity information

humi_cement_100 = pd.read_excel(file_path, skiprows = 2, usecols = 'D')
humi_gypsum_3 = pd.read_excel(file_path, skiprows = 2, usecols = 'H')
humi_gypsum_5 = pd.read_excel(file_path, skiprows = 2, usecols = 'L')
humi_gypsum_10 = pd.read_excel(file_path, skiprows = 2, usecols = 'P')
humi_gypsum_20 = pd.read_excel(file_path, skiprows = 2, usecols = 'T')
humi_gypsum_100 = pd.read_excel(file_path, skiprows = 2, usecols = 'X')

humi_cement_100 = np.array(humi_cement_100, dtype = float) * 100
humi_gypsum_3 = np.array(humi_gypsum_3, dtype = float) * 100
humi_gypsum_5 = np.array(humi_gypsum_5, dtype = float) * 100
humi_gypsum_10 = np.array(humi_gypsum_10, dtype = float) * 100
humi_gypsum_20 = np.array(humi_gypsum_20, dtype = float) * 100
humi_gypsum_100 = np.array(humi_gypsum_100, dtype = float) * 100

#convert humidity dataframe to keras tensor

humi_gypsum_3 = np.expand_dims(humi_gypsum_3, axis = 0)
humi_gypsum_3 = K.variable(humi_gypsum_3)

humi_gypsum_5 = np.expand_dims(humi_gypsum_5, axis = 0)
humi_gypsum_5 = K.variable(humi_gypsum_5)

humi_gypsum_10 = np.expand_dims(humi_gypsum_10, axis = 0)
humi_gypsum_10 = K.variable(humi_gypsum_10)

humi_gypsum_20 = np.expand_dims(humi_gypsum_20, axis = 0)
humi_gypsum_20 = K.variable(humi_gypsum_20)

humi_gypsum_100 = np.expand_dims(humi_gypsum_100, axis = 0)
humi_gypsum_100 = K.variable(humi_gypsum_100)

humi_cement_100 = np.expand_dims(humi_cement_100, axis = 0)
humi_cement_100 = K.variable(humi_cement_100)

humi_g3 = AveragePooling1D(pool_size = pool_size, 
                            strides = strides, 
                            padding = padding)(humi_gypsum_3)
humi_g5 = AveragePooling1D(pool_size = pool_size, 
                            strides = strides, 
                            padding = padding)(humi_gypsum_5)
humi_g10 = AveragePooling1D(pool_size = pool_size, 
                            strides = strides, 
                            padding = padding)(humi_gypsum_10)
humi_g20 = AveragePooling1D(pool_size = pool_size, 
                            strides = strides, 
                            padding = padding)(humi_gypsum_20)
humi_g100 = AveragePooling1D(pool_size = pool_size, 
                            strides = strides, 
                            padding = padding)(humi_gypsum_100)
humi_c100 = AveragePooling1D(pool_size = pool_size, 
                            strides = strides, 
                            padding = padding)(humi_cement_100)

#convert keras tensor to numpy array
humi_g3 = K.eval(humi_g3)
humi_g3 = np.reshape(humi_g3, (-1, 1))

humi_g5 = K.eval(humi_g5)
humi_g5 = np.reshape(humi_g5, (-1, 1))

humi_g10 = K.eval(humi_g10)
humi_g10 = np.reshape(humi_g10, (-1, 1))

humi_g20 = K.eval(humi_g20)
humi_g20 = np.reshape(humi_g20, (-1, 1))

humi_g100 = K.eval(humi_g100)
humi_g100 = np.reshape(humi_g100, (-1, 1))

humi_c100 = K.eval(humi_c100)
humi_c100 = np.reshape(humi_c100, (-1, 1))


#create data frame
gypsum_proportion = [0, 3, 5, 10, 20, 100]
cement_proportion = [100, 97, 95, 90, 80, 0]

type_3_cement = np.array([cement_proportion[1]] * (len(array_g3) - 1))
type_3_gypsum = np.array([gypsum_proportion[1]] * (len(array_g3) - 1))
type_3 = np.stack((type_3_cement, type_3_gypsum), axis = 1)
array_g3_in = array_g3[0:(len(array_g3) - 1)]
array_g3_out = array_g3[1:len(array_g3)]
humi_g3_in = humi_g3[0:(len(humi_g3) - 1)]
type_3 = np.concatenate((type_3, array_g3_in, humi_g3_in, 
                         array_g3_out), axis = 1)

type_5_cement = np.array([cement_proportion[2]] * (len(array_g5) - 1))
type_5_gypsum = np.array([gypsum_proportion[2]] * (len(array_g5) - 1))
type_5 = np.stack((type_5_cement, type_5_gypsum), axis = 1)
array_g5_in = array_g5[0:(len(array_g5) - 1)]
array_g5_out = array_g5[1:len(array_g5)]
humi_g5_in = humi_g5[0:(len(humi_g5) - 1)]
type_5 = np.concatenate((type_5, array_g5_in, humi_g5_in, 
                         array_g5_out), axis = 1)

type_10_cement = np.array([cement_proportion[3]] * (len(array_g10) - 1))
type_10_gypsum = np.array([gypsum_proportion[3]] * (len(array_g10) - 1))
type_10 = np.stack((type_10_cement, type_10_gypsum), axis = 1)
array_g10_in = array_g10[0:(len(array_g10) - 1)]
array_g10_out = array_g10[1:len(array_g10)]
humi_g10_in = humi_g10[0:(len(humi_g10) - 1)]
type_10 = np.concatenate((type_10, array_g10_in, humi_g10_in, 
                          array_g10_out), axis = 1)

type_20_cement = np.array([cement_proportion[4]] * (len(array_g20) - 1))
type_20_gypsum = np.array([gypsum_proportion[4]] * (len(array_g20) - 1))
type_20 = np.stack((type_20_cement, type_20_gypsum), axis = 1)
array_g20_in = array_g20[0:(len(array_g20) - 1)]
array_g20_out = array_g20[1:len(array_g20)]
humi_g20_in = humi_g20[0:(len(humi_g20) - 1)]
type_20 = np.concatenate((type_20, array_g20_in, humi_g20_in, 
                          array_g20_out), axis = 1)

type_100_cement = np.array([cement_proportion[5]] * (len(array_g100) - 1))
type_100_gypsum = np.array([gypsum_proportion[5]] * (len(array_g100) - 1))
type_100 = np.stack((type_100_cement, type_100_gypsum), axis = 1)
array_g100_in = array_g100[0:(len(array_g100) - 1)]
array_g100_out = array_g100[1:len(array_g100)]
humi_g100_in = humi_g100[0:(len(humi_g100) - 1)]
type_100 = np.concatenate((type_100, array_g100_in, humi_g100_in, 
                           array_g100_out), axis = 1)

type_0_cement = np.array([cement_proportion[0]] * (len(array_c100) - 1))
type_0_gypsum = np.array([gypsum_proportion[0]] * (len(array_c100) - 1))
type_0 = np.stack((type_0_cement, type_0_gypsum), axis = 1)
array_c100_in = array_c100[0:(len(array_c100) - 1)]
array_c100_out = array_c100[1:len(array_c100)]
humi_c100_in = humi_c100[0:(len(humi_c100) - 1)]
type_0 = np.concatenate((type_0, array_c100_in, humi_c100_in, 
                         array_c100_out), axis = 1)


df_data = np.concatenate((type_0, type_3, type_5, type_10, type_20, type_100), axis = 0)
data_df = pd.DataFrame(df_data)
data_df.columns = ['Cement','Gypsum','Radon_x', 'Humidity', 'Radon_y']


#excel_name = 'Excel_6.xlsx'
#excel_name = 'Excel_7.xlsx'
excel_name = 'Excel_8.xlsx'
#excel_name = 'Excel_9.xlsx'
#excel_name = 'Excel_10.xlsx'

results_folder = os.path.join('gypsum_model', excel_name)
excel_path = os.path.join(os.getcwd(), results_folder)
writer = pd.ExcelWriter(excel_path)
data_df.to_excel(writer, 'Sheet1', float_format='%.3f')
writer.save()
writer.close()



