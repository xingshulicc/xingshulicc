# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Fri May  3 11:00:34 2019

@author: Admin
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

#get file path
file_name = 'ammonia_emission.xlsx'
file_dic = os.getcwd()
file_path = os.path.join(file_dic, file_name)

#load data
ammonia_emission = pd.read_excel(file_path, skiprows = 1, usecols = 'B:M')
ammonia_emission = np.array(ammonia_emission, dtype = float)

average_ammonia = pd.read_excel(file_path, skiprows = 1, usecols = 'N')
average_ammonia = np.array(average_ammonia, dtype = float)

def load_data(data, num, begin, skip):
    data_split = []
    i = 0
    while i < num:
        data_split.append(data[begin])
        begin = begin + skip
        i = i + 1
    
    return np.array(data_split)

Opc = load_data(ammonia_emission, 8, 0, 4)
fly_ash_20 = load_data(ammonia_emission, 8, 1, 4)
fly_ash_40 = load_data(ammonia_emission, 8, 2, 4)
fly_ash_60 = load_data(ammonia_emission, 8, 3, 4)

#Data Visualization
plt.figure(1)
x_axis = np.ones((Opc.shape[0], Opc.shape[1]), dtype = float)
time_series = np.array([0.125, 0.25, 0.375, 0.5, 1, 3, 10, 30])
color_array = np.array(['r', 'g', 'b', 'y', 'm', 'k', 'hotpink', 'orange'])
annotations = np.array(['3h', '6h', '9h', '12h', '1d', '3d', '10d', '30d'])
figures = []
for i in range(0, len(time_series)):
    x_axis[i] = x_axis[i] * time_series[i]
    p = plt.scatter(x_axis[i], Opc[i], c = color_array[i], marker = 'o')
    figures.append(p)

plt.title('OPC Ammonia Emission in Different Moments')
plt.xlabel('Time / day')
plt.ylabel('Ammonia emission / ppm')
plt.legend((i for i in figures), (j for j in annotations), loc = 'best')
plt.show()

plt.figure(2)
x_axis_flat = x_axis.flatten()
Opc_flat = Opc.flatten()
fly_ash_20_flat = fly_ash_20.flatten()
fly_ash_40_flat = fly_ash_40.flatten()
fly_ash_60_flat = fly_ash_60.flatten()
p0 = plt.scatter(x_axis_flat, Opc_flat, c = 'r', marker = 'o')
p1 = plt.scatter(x_axis_flat, fly_ash_20_flat, c = 'g', marker = 'o')
p2 = plt.scatter(x_axis_flat, fly_ash_40_flat, c = 'b', marker = 'o')
p3 = plt.scatter(x_axis_flat, fly_ash_60_flat, c = 'm', marker = 'o')

plt.title('Ammonia Emission of Different Percentage Fly Ash')
plt.xlabel('Time / day')
plt.ylabel('Ammonia emission / ppm')
plt.legend((p0, p1, p2, p3), ('OPC', 'Fly_20', 'Fly_40', 'Fly_60'), loc = 'best')
plt.show()

#Generate train data
time_data = x_axis
Opc_train = np.array(list(zip(Opc, time_data)))
Opc_train = np.transpose(Opc_train, (0, 2, 1))
Opc_train = np.reshape(Opc_train, (-1, 2))

fly_ash_20_train = np.array(list(zip(fly_ash_20, time_data)))
fly_ash_20_train = np.transpose(fly_ash_20_train, (0, 2, 1))
fly_ash_20_train = np.reshape(fly_ash_20_train, (-1, 2))

fly_ash_40_train = np.array(list(zip(fly_ash_40, time_data)))
fly_ash_40_train = np.transpose(fly_ash_40_train, (0, 2, 1))
fly_ash_40_train = np.reshape(fly_ash_40_train, (-1, 2))

fly_ash_60_train = np.array(list(zip(fly_ash_60, time_data)))
fly_ash_60_train = np.transpose(fly_ash_60_train, (0, 2, 1))
fly_ash_60_train = np.reshape(fly_ash_60_train, (-1, 2))
    
train_data = np.concatenate((Opc_train, fly_ash_20_train, fly_ash_40_train, fly_ash_60_train), 
                            axis = 0)

#Generate labels
def generate_labels(data, begin):
    count = len(data)
    i = 0
    labels = []
    while i < count:
        num_labels = len(data[i])
        init_labels = np.ones(num_labels)
        labels.append(init_labels * (i + begin))
        i = i + 1
    
    return(np.array(labels))

Opc_labels = generate_labels(Opc, 0)
fly_ash20_labels = generate_labels(fly_ash_20, 8) 
fly_ash40_labels = generate_labels(fly_ash_40, 16)
fly_ash60_labels = generate_labels(fly_ash_60, 24)

train_labels = np.concatenate((Opc_labels, fly_ash20_labels, fly_ash40_labels, fly_ash60_labels), 
                              axis = 0)
train_labels = np.reshape(train_labels, (-1))


#Data Normalization
feature_range = (0, 1)
min_max_scaler = preprocessing.MinMaxScaler(feature_range = feature_range, copy = True)
train_data_norm = min_max_scaler.fit_transform(train_data)

#classification using MLP
inputs = train_data_norm
outputs = train_labels

hidden_layer_neurons = list(range(10, 110, 10))
MLP_train_results = []

for h in hidden_layer_neurons:
    classifier_MLP = MLPClassifier(hidden_layer_sizes = (h,), 
                                   activation = 'logistic', 
                                   solver = 'lbfgs', 
                                   max_iter = 10000, 
                                   warm_start = False)
    
    classifier_MLP.fit(inputs, outputs)
    classifier_accuracy = accuracy_score(y_true = outputs, 
                                         y_pred = classifier_MLP.predict(inputs), 
                                         normalize = True)
    MLP_train_results.append(classifier_accuracy)
    
MLP_train_results = np.array(MLP_train_results)
Max_value = np.max(MLP_train_results)
Max_value_index = np.where(MLP_train_results == Max_value)

#Time series prediction using MLP
#OPC prediction

def generate_regressor_inputs_outputs(data, time):
    Regressor_inputs = []
    Regressor_outputs = []
    rows = data.shape[0]
    cols = data.shape[1]
    for i in range(0, (rows - 1)):
        for j in range(0, cols):
            input_data = data[i][j]
            output_data = data[i + 1][j]
            Regressor_inputs.append([input_data, time[i]])
            Regressor_outputs.append([output_data, time[i + 1]])
    
    return np.array(Regressor_inputs), np.array(Regressor_outputs)
time = time_series
OPC_regressor_inputs, OPC_regressor_outputs = generate_regressor_inputs_outputs(Opc, time)

#Data Normalization
OPC_regressor_inputs = min_max_scaler.fit_transform(OPC_regressor_inputs)
OPC_regressor_outputs = min_max_scaler.fit_transform(OPC_regressor_outputs)



regressor_hidden_neurons = list(range(10, 110, 10))
regressor_results = []
for r in regressor_hidden_neurons:
    Regressor_MLP = MLPRegressor(hidden_layer_sizes = (r,), 
                                 activation = 'logistic', 
                                 solver = 'lbfgs', 
                                 max_iter = 10000, 
                                 warm_start = False)
    
    Regressor_MLP.fit(OPC_regressor_inputs, OPC_regressor_outputs)
    Regressor_error = mean_squared_error(y_true = OPC_regressor_outputs, 
                                         y_pred = Regressor_MLP.predict(OPC_regressor_inputs))
    
    regressor_results.append(Regressor_error)

regressor_results = np.array(regressor_results)
Min_value = np.min(regressor_results)
Min_value_index = np.where(regressor_results == Min_value)

Min_value_index = list(Min_value_index)
index = Min_value_index[0][0]

Regressor_final = regressor_hidden_neurons[index]
Regressor_MLP_final = MLPRegressor(hidden_layer_sizes = (Regressor_final,), 
                                   activation = 'logistic', 
                                   solver = 'lbfgs', 
                                   max_iter = 10000, 
                                   warm_start = False)

Regressor_MLP_final.fit(OPC_regressor_inputs, OPC_regressor_outputs)

Regressor_pre = Regressor_MLP_final.predict(OPC_regressor_inputs)
Regressor_pre = min_max_scaler.inverse_transform(Regressor_pre)
Regressor_pre = np.transpose(Regressor_pre, (1, 0))

plt.figure(3)
p_pre = plt.scatter(Regressor_pre[1], Regressor_pre[0], c = 'r', marker = 'o')
p_real = plt.scatter(x_axis_flat, Opc_flat, c = 'b', marker = 'o')

plt.title('OPC Comparison')
plt.xlabel('Time / day')
plt.ylabel('Ammonia emission / ppm')
plt.legend((p_pre, p_real), ('prediction', 'real'), loc = 'best')
plt.show()



