#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue Sep  1 13:06:23 2020

@author: xingshuli
"""

import os
import numpy as np

from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image

#import cv2

import matplotlib.pyplot as plt

#load model
model_save_dir = os.path.join(os.getcwd(), 'Multi_Scale_Model')
model_name = 'keras_trained_model.h5'
model_save_path = os.path.join(model_save_dir, model_name)
model = load_model(model_save_path)

#get output of ith layer: 68, 69, 70, color = b, m, r
i = 70
get_ith_layer_output = K.function([model.layers[0].input, K.learning_phase()], 
                                  [model.layers[i].output])

#load image
img_path = '/home/xingshuli/Desktop/test_pictures/green_2.jpg'

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /=255.
print('Input image shape:', x.shape)

#output in test mode = 0
layer_output = get_ith_layer_output([x, 0])[0]
length = layer_output.shape[3]
layer_output = layer_output.reshape(length)
layer_output = np.array(layer_output)

mu = np.mean(layer_output)
mu = round(mu, 4)
sigma = np.std(layer_output)
sigma = round(sigma, 4)

#draw histogram

plt.hist(layer_output, 
         bins = 40, 
         edgecolor = 'k', 
         color = 'r')
title_name = 'mu = ' + str(mu) + ', ' + 'sigma = ' + str(sigma)
plt.title(title_name, fontsize = 16)
plt.xlabel('Value', fontsize = 14)
plt.ylabel('Count', fontsize = 14)

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
 
plt.savefig('s.png', dpi = 800)





