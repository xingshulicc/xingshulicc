# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Sun Dec 31 13:13:11 2017

@author: xingshuli
"""
import os
import numpy as np

from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image

import cv2

import matplotlib.pyplot as plt

#load model
model_save_dir = os.path.join(os.getcwd(), 'resnet_model')
model_name = 'keras_resnet_trained_model.h5'
model_save_path = os.path.join(model_save_dir, model_name)
model = load_model(model_save_path)

#get output of ith layer
i = 37
get_ith_layer_output = K.function([model.layers[0].input, K.learning_phase()], 
                                  [model.layers[i].output])

#load image
img_path = '/home/xingshuli/Desktop/test_pictures/luna_female.jpeg'

'''
the test insect name included: luna_female, Delias_female, Hyaloph_female
Forktailed_Female

'''
img = image.load_img(img_path, target_size=(200, 200))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /=255.
print('Input image shape:', x.shape)

#output in test mode = 0
layer_output = get_ith_layer_output([x, 0])[0]
merged = cv2.merge(layer_output)

index = -1
Avg_feature_maps = []
num_feature_maps = 256  #the number of feature maps 
#show the output feature map
for j in range(num_feature_maps):
    channel_list = merged[:, :, j]
    avg_channel_list = channel_list.mean()
    Avg_feature_maps.append(avg_channel_list)
    index += 1
    plt.axis('off')
    plt.imshow(channel_list)
    #save feature maps in given folder
    img_save_dir = os.path.join(os.getcwd(), 'output_feature_map')
    img_save_name = 'feature_map' + '_' + str(index)
    img_save_path = os.path.join(img_save_dir, img_save_name)
    plt.savefig(img_save_path)
    plt.show()

print('the output feature maps are saved\n')

reverse_sorted = sorted(Avg_feature_maps, reverse = True)
top_index = []

for i_1 in reverse_sorted:
    for j_1 in Avg_feature_maps:
        if i_1 == j_1:
            top_index.append(Avg_feature_maps.index(j_1))

num_top_activation = 10
for h in range(num_top_activation):
    x_1 = top_index[h]
    y_1 = merged[:, :, x_1]
    plt.axis('off')
    plt.imshow(y_1)
    #save the top 10 strongest response activations
    img_save_dir = os.path.join(os.getcwd(), 'top_10_maps')
    img_save_name = 'feature_map' + '_' + str(x_1)
    img_save_path = os.path.join(img_save_dir, img_save_name)
    plt.savefig(img_save_path)
    plt.show()

print('the feature maps of top-10 activations are saved')
