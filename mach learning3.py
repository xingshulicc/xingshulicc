# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue Nov 28 18:25:53 2017

@author: xingshuli
"""
#import numpy as np

from sklearn import datasets
from sklearn import svm
from sklearn import metrics

import matplotlib.pyplot as plt

#load dataset
digits = datasets.load_digits()
#flatten the image, to turn the data in a (samples, feature) matrix
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

#create menu
print('*********************** Menu ************************')
print('******** Welcome to use svm.SCV in sklearn ******** \n')
print('******** The available kernels are "linear" and "rbf" ******** \n')
print('******** Please choose one option from the above kernels ********')
kernel_option = input()
assert kernel_option in ['linear', 'rbf'], 'Incorrect Input'

# creat svm classifier
clf = svm.SVC(C = 100.0, gamma = 0.001, kernel= kernel_option)
#the ratio of training set and testing set is 2:1
clf.fit(data[n_samples//3:], digits.target[n_samples//3:])
#predict the value of digit on test set
expected = digits.target[:n_samples//3]
predicted = clf.predict(data[:n_samples//3])

print('classification performance: \n%s\n' %
      metrics.classification_report(expected, predicted))

print('confusion matrix: \n%s' %
      metrics.confusion_matrix(expected, predicted))

judge = (expected == predicted)

#print(sum(judge)) #the number of correct classification digits
#print(len(judge) - sum(judge)) #the number of incorrect classification digits

incorrect_list = []
for i in range(n_samples//3):
    if judge[i] == False:
        incorrect_list.append(i)

#print(incorrect_list)
index = 0
for j in incorrect_list:
    index += 1
    plt.figure(index)
    plt.axis('off')
    plt.imshow(digits.images[j], cmap = plt.cm.gray_r, interpolation='nearest')
    plt.title('label: %i\n prediction: %i\n' %
              (expected[j], predicted[j]))
    plt.rcParams['font.size'] = 10

plt.show()
    
    


