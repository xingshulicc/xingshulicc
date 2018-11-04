# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue Oct 23 15:22:53 2018

@author: xingshuli
"""
import numpy as np
import pandas as pd
import os

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

file_name = 'project_output.txt'
file_path = os.path.join(os.getcwd(), file_name)

table = pd.read_csv(file_path, delim_whitespace = True, header = 0)

#Read data of specified columns: features
Hardness = table['Hardness']
Adhesiveness = table['Adhesiveness']
Cohesiveness = table['Cohesiveness']
Gumminess = table['Gumminess']
Chewiness = table['Chewiness']
Resilience = table['Resilience']

#Read data of specified rows: classes
kimpo = table.loc[0:8, :]
yeoju = table.loc[9:17, :]
hongcheon = table.loc[18:26, :]
danyang = table.loc[27:35, :]
boryeng = table.loc[36:44, :]
uiseong = table.loc[45:53, :]
sancheong = table.loc[54:62, :]
iksan = table.loc[63:71, :]
gochang = table.loc[72:80, :]
naju = table.loc[81:89, :]


#data normalization
feature_range = (-1, 1)
min_max_scaler = preprocessing.MinMaxScaler(feature_range = feature_range, copy = True)
table_norm = min_max_scaler.fit_transform(table)

#PCA analysis
#Determine which dimensions are not important 
pca = PCA(n_components = 6, svd_solver = 'auto', copy = True)
pca.fit(table_norm)
pca_variance = pca.explained_variance_ 
pca_ratio = pca.explained_variance_ratio_

#Dimensional reduction
pca_1 = PCA(n_components = 0.98, svd_solver = 'auto', copy = True)
table_pca = pca_1.fit_transform(table_norm)
pca_1_variance = pca_1.explained_variance_
pca_1_ratio = pca_1.explained_variance_ratio_

#generate training data and labels 
train_data = table_pca

#the labels are 0~9: we have ten classes in total
labels = []
for i in range(0, 10):
    label_index = [i] * 9
    labels.append(label_index)
labels = np.array(labels)
labels = np.reshape(labels, (-1))
#print(labels)

#classification by SVM
#how to choose gamma and C: 
#http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
clf = svm.SVC(C = 100.0, 
              kernel = 'rbf', 
              gamma = 0.7, 
              decision_function_shape = 'ovr')

clf.fit(train_data, labels)

#print the confusion matrix and training accuracy for training dataset  
y_pred = clf.predict(train_data)
cnf_matrix = confusion_matrix(labels, y_pred)
accuracy = accuracy_score(labels, y_pred)


#classification by KNN
#firstly, we should determine the best value of neighbor
neighbors = list(xrange(5, 30))
train_results = []
for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(train_data, labels)
    
    knn_pred = knn.predict(train_data)
    knn_accuracy = accuracy_score(labels, knn_pred)
    
    train_results.append(knn_accuracy)

plt.plot(neighbors, train_results, 'bo--')
plt.xlabel('The value of neighbor')
plt.ylabel('Training accuracy')
plt.show()

#get the KNN classifier confusion matrix and training accuracy
neighbor = np.argmax(np.array(train_results)) + 5
KNN = KNeighborsClassifier(n_neighbors = neighbor)
KNN.fit(train_data, labels)
KNN_accuracy = accuracy_score(labels, KNN.predict(train_data))
KNN_cnf_matrix = confusion_matrix(labels, KNN.predict(train_data))

#classification by MLP 
#firstly, we should determine the number of hidden neurons
hidden_layer1_neurons = list(xrange(4, 16))
hidden_layer2_neurons = list(xrange(4, 16))

mlp_train_results = []

for h_1 in hidden_layer1_neurons:
    for h_2 in hidden_layer2_neurons:
        mlp = MLPClassifier(hidden_layer_sizes = (h_1, h_2), 
                            activation = 'tanh', 
                            solver = 'adam', 
                            learning_rate_init = 0.01, 
                            warm_start = False)
        mlp.fit(train_data, labels)
        mlp_pred = mlp.predict(train_data)
        mlp_accuracy = accuracy_score(labels, mlp_pred)
        
        mlp_train_results.append(mlp_accuracy)

#Get the index of highest accuracy
mlp_train_results = np.array(mlp_train_results)

newshape = (len(hidden_layer1_neurons), len(hidden_layer2_neurons))
mlp_train_results = np.reshape(mlp_train_results, newshape = newshape)

maximum_value = np.max(mlp_train_results)

maximum_value_index = np.where(mlp_train_results == maximum_value)

#determine the best structure of DNN model
#choose the samllest values
layer_1_neurons = maximum_value_index[0][0] + 4
layer_2_neurons = maximum_value_index[1][0] + 4

print('The number of neurons in the first hidden layer is: %d' % layer_1_neurons)
print('The number of neurons in the second hidden layer is: %d' % layer_2_neurons)

#get the MLP classifier confusion matrix and training accuracy
DNN = MLPClassifier(hidden_layer_sizes = (layer_1_neurons, layer_2_neurons), 
                    activation = 'tanh', 
                    solver = 'adam', 
                    learning_rate_init = 0.01, 
                    warm_start = False)

DNN.fit(train_data, labels)
DNN_accuracy = accuracy_score(labels, DNN.predict(train_data))
DNN_cnf_matrix = confusion_matrix(labels, DNN.predict(train_data))





