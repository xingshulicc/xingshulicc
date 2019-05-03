# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Feb 20 10:18:17 2019

@author: xingshuli
"""
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D

from keras.layers import Input
from keras.layers import Dropout
from keras.layers import BatchNormalization

from keras import backend as K
from keras.models import Model

def _block1(inputs, filters, block):
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = -1
    
    base_name = 'block' + '_' + str(block)
    
    x = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', 
               name = base_name + '_conv_1')(inputs)
    x = BatchNormalization(axis = bn_axis, name = base_name + '_bn_1')(x)
    x = Activation('relu', name = base_name + '_relu_1')(x)
    
    x = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', 
               name = base_name + '_conv_2')(x)
    x = BatchNormalization(axis = bn_axis, name = base_name + '_bn_2')(x)
    x = Activation('relu', name = base_name + '_relu_2')(x)
    
    x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                     name = base_name + '_MaxPool')(x)
    
    return x

def _block2(inputs, filters, block):
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = -1
    
    base_name = 'block' + '_' + str(block)
    
    x = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', 
               name = base_name + '_conv_1')(inputs)
    x = BatchNormalization(axis = bn_axis, name = base_name + '_bn_1')(x)
    x = Activation('relu', name = base_name + '_relu_1')(x)
    
    x = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', 
               name = base_name + '_conv_2')(x)
    x = BatchNormalization(axis = bn_axis, name = base_name + '_bn_2')(x)
    x = Activation('relu', name = base_name + '_relu_2')(x)
    
    x = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', 
               name = base_name + '_conv_3')(x)
    x = BatchNormalization(axis = bn_axis, name = base_name + '_bn_3')(x)
    x = Activation('relu', name = base_name + '_relu_3')(x)
    
    x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                     name = base_name + '_MaxPool')(x)
    
    return x

def VGG16(input_shape, classes):
    inputs = Input(shape = input_shape)
    
    x = _block1(inputs, 64, 1)
    x = _block1(x, 128, 2)
    x = _block2(x, 256, 3)
    x = _block2(x, 512, 4)
    x = _block2(x, 512, 5)
    
    output = GlobalAveragePooling2D()(x)
    output = Dense(512, activation = 'relu', name = 'fc_1')(output)
    output = Dropout(0.5)(output)
    output = Dense(classes, activation = 'softmax', name = 'fc_2')(output)
    
    model = Model(inputs = inputs, outputs = output, name = 'VGG16')
    
    return model


