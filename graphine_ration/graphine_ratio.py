# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Fri Jul  5 14:36:12 2019

@author: Admin
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

#get file path
file_name = 'graphine_ratio.xlsx'
file_dic = os.getcwd()
file_path = os.path.join(file_dic, file_name)

#load data
graphine = pd.read_excel(file_path, skiprows = 0, usecols = 'F:H')
graphine = np.array(graphine, dtype = float)

Wg00 = graphine[:40]
Wg01 = graphine[40:80]
Wg03 = graphine[80:120]
Wg05 = graphine[120:160]
Wg10 = graphine[160:200]
Wg50 = graphine[200:240]
Wg100 = graphine[240:280]
Wg200 = graphine[280:320]

train_data = np.concatenate((Wg00, Wg01, Wg03, Wg05, Wg10, Wg50, Wg100, Wg200), 
                            axis = 0)
#Generate labels
def generate_labels(data, label):
    init_labels = np.ones(len(data))
    labels = init_labels * label
    return labels

Wg00_labels = generate_labels(Wg00, 0)
Wg01_labels = generate_labels(Wg01, 1)
Wg03_labels = generate_labels(Wg03, 2)
Wg05_labels = generate_labels(Wg05, 3)
Wg10_labels = generate_labels(Wg10, 4)
Wg50_labels = generate_labels(Wg50, 5)
Wg100_labels = generate_labels(Wg100, 6)
Wg200_labels = generate_labels(Wg200, 7)

train_labels = np.concatenate((Wg00_labels, Wg01_labels, Wg03_labels, Wg05_labels, Wg10_labels, Wg50_labels, Wg100_labels, Wg200_labels), 
                              axis = 0)

#Data Normalization
feature_range = (0, 1)
min_max_scaler = preprocessing.MinMaxScaler(feature_range = feature_range, copy = True)
train_data_norm = min_max_scaler.fit_transform(train_data)

#split whole dataset into train set and test set
X_train, X_test, y_train, y_test = train_test_split(train_data_norm, 
                                                    train_labels, 
                                                    test_size = 0.25, 
                                                    random_state = 42)

#Classification
hidden_layer_neurons = list(range(10, 55, 5))
MLP_train_results = []
MLP_test_results = []

for h in hidden_layer_neurons:
    classifier_MLP = MLPClassifier(hidden_layer_sizes = (h,), 
                                   activation = 'logistic', 
                                   solver = 'lbfgs', 
                                   max_iter = 10000,
                                   shuffle = True, 
                                   verbose = False,
                                   warm_start = False, 
                                   early_stopping = True
                                   )
    
    classifier_MLP.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_true = y_train, 
                                    y_pred = classifier_MLP.predict(X_train), 
                                    normalize = True)
    MLP_train_results.append(train_accuracy)
    
    test_accuracy = accuracy_score(y_true = y_test, 
                                   y_pred = classifier_MLP.predict(X_test), 
                                   normalize = True)
    MLP_test_results.append(test_accuracy)
    
MLP_train_results = np.array(MLP_train_results)
MLP_test_results = np.array(MLP_test_results)



