# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue Oct  6 16:23:04 2020

@author: Admin
"""
import numpy as np
import pandas as pd
import math
import os
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential

import matplotlib.pyplot as plt

#load data
filename = 'international-airline-passengers.csv'
filepath = os.path.join(os.getcwd(), filename)
dataframe = pd.read_csv(filepath, 
                        usecols = [1], 
                        engine = 'python')
dataset = dataframe.values
#convert dataframe to numpy array
dataset = dataset.astype('float32')
#the shape of dataset: num_samples, features

#normalise the dataset
feature_range = (0, 1)
scaler = MinMaxScaler(feature_range = feature_range)
dataset = scaler.fit_transform(dataset)

#split the dataset into training and test set
i_split = 0.8
train_size = int(len(dataset) * i_split)
#print(train_size)
test_size = len(dataset) - train_size
#print(test_size)
train_set = dataset[0:train_size, :]
test_set = dataset[train_size:, :]

#convert an array values into a dataset matrix for LSTM
def create_dataset(dataset, look_back):
    dataX = []
    dataY = []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i+look_back), 0]
        b = dataset[i+look_back, 0]
        dataX.append(a)
        dataY.append(b)
        
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    
    return dataX, dataY

look_back = 1
#look_back = time_steps:  the number of previous time steps
trainX, trainY = create_dataset(train_set, look_back)
testX, testY = create_dataset(test_set, look_back)

#reshape input to be [samples, time_steps, features]
time_steps = look_back
features = dataset.shape[1]
trainX = np.reshape(trainX, (trainX.shape[0], time_steps, features))
testX = np.reshape(testX, (testX.shape[0], time_steps, features))

#create and fit the LSTM 
input_shape = (time_steps, features)
lstm_neurons = 4
#lstm_neurons is a hyper-parameter
dense_neurons = 1
#dense_neurions is equal to the shape of trainY(= 1)
batch_size = 1
epochs = 100
lr = 0.001
optimizer = Adam(lr = lr, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, decay = 0.0, amsgrad = True)

model = Sequential()
model.add(LSTM(lstm_neurons, input_shape = input_shape, return_sequences = False))
model.add(Dense(dense_neurons, activation = 'linear'))
model.compile(loss = 'mean_squared_error', optimizer = optimizer)
model.fit(trainX, 
          trainY, 
          batch_size = batch_size, 
          epochs = epochs, 
          verbose = 1, 
          shuffle = True)

#make predictions
trainPredict = model.predict(trainX, batch_size = batch_size)
testPredict = model.predict(testX, batch_size = batch_size)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

'''
the most important hyper-parameter is look_back and batch_size
researchers should try few times to determine the best values
'''
