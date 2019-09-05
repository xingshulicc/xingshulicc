# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue Sep  3 11:46:58 2019

@author: Admin
"""
import pandas as pd
import os
import numpy as np

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from keras.models import load_model

import matplotlib.pyplot as plt

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

test_file = os.path.join(file_folder, 'gypsum_separate_test.xlsx')
test_data = pd.read_excel(test_file, skiprows = 0, usecols = 'B:D')
test_data = np.array(test_data, dtype = 'float64')

test_inputs = test_data[:, :2]
test_outputs = test_data[:, 2]
test_outputs = np.reshape(test_outputs, (-1, 1))

#load model 
#model_name = 'Ratio_0_model.h5'
#model_name = 'Ratio_3_model.h5'
#model_name = 'Ratio_5_model.h5'
#model_name = 'Ratio_10_model.h5'
#model_name = 'Ratio_20_model.h5'
model_name = 'Ratio_100_model.h5'

model_save_path = os.path.join(file_folder, model_name)
model = load_model(model_save_path)

#model test

separate_inputs = test_inputs[70:84]
separate_outputs = test_outputs[70:84]
separate_predict_outputs = model.predict(separate_inputs)

#inverse predicted outputs
inv_separate_predict_outputs = scaler_outputs.inverse_transform(separate_predict_outputs)


'''
actual outputs: separate_outputs

predicted outputs: separate_predict_outputs

'''
compare_file = os.path.join(file_folder, 'Excel_10.xlsx')

actual_data = pd.read_excel(compare_file, skiprows = 0, usecols = 'F')
actual_data = np.array(actual_data, dtype = 'float64')

prediction = pd.read_excel(compare_file, skiprows = 0, usecols = 'G')
prediction = np.array(prediction, dtype = 'float64')
prediction = np.around(prediction, decimals = 1)

r_square = r2_score(actual_data, prediction)
radon_mse = mean_squared_error(actual_data, prediction)

#draw fit plot figure
plt.scatter(actual_data, prediction, s = 40, c = '', marker = 'o', edgecolors = 'k')
lr = LinearRegression()
lr.fit(actual_data, prediction)
Y_pred = lr.predict(actual_data)
plt.plot(actual_data, Y_pred, color = 'r', linewidth = 4.0, label = "Fit")
plt.xlabel('Measured')
plt.ylabel('Predict')
plt.legend(loc = 'best')
plt.grid(linestyle = '-.', c = 'k')
plt.show()

print(r_square)
print(radon_mse)

