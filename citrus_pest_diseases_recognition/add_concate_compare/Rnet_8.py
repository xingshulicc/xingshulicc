# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Jun 19 11:57:11 2019

@author: Admin
"""
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Activation

#from keras.layers import Add
from keras.layers import Concatenate

from keras.layers import Dropout
from keras.layers import Input

from keras import backend as K
from keras.models import Model

from keras.utils import plot_model

def _init_conv_block(inputs, filters, kernel_size, strides):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    x = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, 
               padding = 'same', name = 'init_conv')(inputs)
    x = BatchNormalization(axis = channel_axis, name = 'init_conv_bn')(x)
    x = Activation('relu', name = 'init_conv_relu')(x)
    
    return x

def _trans_layer(inputs, filters, block):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    base_name = 'trans_layer' + '_' + str(block)
    
    x = Conv2D(filters = filters, kernel_size = (1, 1), strides = 1, 
               padding = 'same', name = base_name + '_conv')(inputs)
    x = BatchNormalization(axis = channel_axis, name = base_name + '_conv_bn')(x)
    x = Activation('relu', name = base_name + '_conv_relu')(x)
    x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                     name = base_name + '_MaxPool')(x)
    
    return x

def _concatenate_block(inputs, filters, block):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    base_name = 'concate_layer' + '_' + str(block)
    
    x = Conv2D(filters = filters, kernel_size = (3, 3), strides = 1, 
               padding = 'same', name = base_name + '_conv')(inputs)
    x = BatchNormalization(axis = channel_axis, name = base_name + '_conv_bn')(x)
    x = Activation('relu', name = base_name + '_conv_relu')(x)
    
    x = Concatenate(axis = channel_axis)([inputs, x])
    
    return x

def Rnet_8(input_shape, classes):
    inputs = Input(shape = input_shape)
#    The input_shape: 224 x 224 x 3
     
    x_1 = _init_conv_block(inputs = inputs, filters = 16, kernel_size = (5, 5), strides = (2, 2))
#    The shape of x_1: 112 x 112 x 16
    
    x_2 = _concatenate_block(inputs = x_1, filters = 16, block = 1)
#    The shape of x_2: 112 x 112 x 32
    
    x_3 = _trans_layer(inputs = x_2, filters = 32, block = 1)
#    The shape of x_3: 56 x 56 x 32
    
    x_4 = _concatenate_block(inputs = x_3, filters = 32, block = 2)
#    The shape of x_4: 56 x 56 x 64
    
    x_5 = _trans_layer(inputs = x_4, filters = 64, block = 2)
#    The shape of x_5: 28 x 28 x 64
    
    x_6 = _concatenate_block(inputs = x_5, filters = 64, block = 3)
#    The shape of x_6: 28 x 28 x 128
    
    x_7 = _trans_layer(inputs = x_6, filters = 128, block = 3)
#    The shape of x_7: 14 x 14 x 128
    
    x_8 = _concatenate_block(inputs = x_7, filters = 128, block = 4)
#    The shape of x_8: 14 x 14 x 256
    
    x_9 = _trans_layer(inputs = x_8, filters = 256, block = 4)
#    The shape of x_9: 7 x 7 x 256
    
    x_10 = _concatenate_block(inputs = x_9, filters = 256, block = 5)
#    The shape of x_10: 7 x 7 x 512
    
    x = GlobalAveragePooling2D()(x_10)
#    The shape of x: 1 x 512
    
    x = Dense(512, activation = 'relu', name = 'fc_1')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation = 'softmax', name = 'fc_2')(x)
    
    model = Model(inputs = inputs, outputs = x, name = 'RNet_small')
    plot_model(model, to_file = 'RNet_small.png',show_shapes = True, show_layer_names = True)
    
    return model
    


