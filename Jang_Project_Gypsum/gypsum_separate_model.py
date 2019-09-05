# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Sep  4 09:25:51 2019

@author: Admin
"""
import pandas as pd
import os
import numpy as np
#import matplotlib.pyplot as plt

from sklearn import preprocessing

from keras.layers import Dense
from keras.layers import Dropout

from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import regularizers

weight_decay = 0.00001


file_folder = os.path.join(os.getcwd(), 'gypsum_model')
#excel_name = 'ratio_0.xlsx'
#excel_name = 'ratio_3.xlsx'
#excel_name = 'ratio_5.xlsx'
#excel_name = 'ratio_10.xlsx'
#excel_name = 'ratio_20.xlsx'
excel_name = 'ratio_100.xlsx'
file_path = os.path.join(file_folder, excel_name)

data = pd.read_excel(file_path, skiprows = 0, usecols = 'D:F')
data = np.array(data, dtype = 'float64')

inputs = data[:, :2]
inputs = np.around(inputs, decimals = 1)
outputs = data[:, 2]
outputs = np.reshape(outputs, (-1, 1))
outputs = np.around(outputs, decimals = 1)

feature_range = (0, 1)
scaler_inputs = preprocessing.MinMaxScaler(feature_range = feature_range, copy = True)
scaler_outputs = preprocessing.MinMaxScaler(feature_range = feature_range, copy = True)

inputs_norm = scaler_inputs.fit_transform(inputs)
outputs_norm = scaler_outputs.fit_transform(outputs)

optimizer = Adam(lr = 0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, decay = 0.0, amsgrad = True)

model = Sequential()
model.add(Dense(20, input_dim = 2, activation = 'sigmoid', 
                kernel_initializer = 'he_normal', 
                kernel_regularizer = regularizers.l2(weight_decay)))
model.add(Dropout(rate = 0.5))
model.add(Dense(1, activation = 'linear', 
                kernel_initializer = 'he_normal', 
                kernel_regularizer = regularizers.l2(weight_decay)))
model.compile(loss = 'mse', optimizer = optimizer)

#model_name = 'Ratio_0_model.h5'
#model_name = 'Ratio_3_model.h5'
#model_name = 'Ratio_5_model.h5'
#model_name = 'Ratio_10_model.h5'
#model_name = 'Ratio_20_model.h5'
model_name = 'Ratio_100_model.h5'

model_save_path = os.path.join(file_folder, model_name)

checkpoint = ModelCheckpoint(model_save_path, 
                             monitor = 'val_loss', 
                             verbose = 1, 
                             save_best_only = True, 
                             save_weights_only = False, 
                             mode = 'min', 
                             period = 10)

callbacks_list = [checkpoint]

hist = model.fit(inputs_norm, 
                 outputs_norm,
                 validation_split = 0.2, 
                 epochs = 7000, 
                 batch_size = 10, 
                 callbacks = callbacks_list)

train_loss = hist.history['loss']
validation_loss = hist.history['val_loss']


#print acc and stored into acc.txt
f = open(os.path.join(file_folder, 'train_loss.txt'),'w')
f.write(str(train_loss))
f.close()
#print val_acc and stored into val_acc.txt
f = open(os.path.join(file_folder, 'val_loss.txt'),'w')
f.write(str(validation_loss))
f.close()


