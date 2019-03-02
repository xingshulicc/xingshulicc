# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Thu Nov 29 13:56:49 2018
@author: xingshuli
"""
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D

from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Input
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model

from keras import regularizers

weight_decay = 0.001


def _initial_conv_block(input):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Conv2D(filters = 32, kernel_size = (7, 7), strides = (2, 2), padding = 'same', 
               kernel_regularizer = regularizers.l2(weight_decay), name = 'init_conv')(input)
    x = BatchNormalization(axis = channel_axis, name = 'init_conv_bn')(x)
    x = Activation('relu', name = 'init_conv_relu')(x)
    
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same', 
                     name = 'init_MaxPool')(x)
    
    return x


def _block(inputs, filters, block):
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = -1
    
    base_name = 'block' + '_' + str(block)
    
    x = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', 
               kernel_regularizer = regularizers.l2(weight_decay), name = base_name + '_conv_1_3')(inputs)
    x = BatchNormalization(axis = bn_axis, name = base_name + '_bn_1')(x)
    x = Activation('relu', name = base_name + '_relu_1')(x)
    
    x = Conv2D(filters = filters, kernel_size = (1, 1), strides = (1, 1), padding = 'same', 
               kernel_regularizer = regularizers.l2(weight_decay), name = base_name + '_conv_2_1')(x)
    x = BatchNormalization(axis = bn_axis, name = base_name + '_bn_2')(x)
    x = Activation('relu', name = base_name + '_relu_2')(x)
    
    x = Conv2D(filters = filters, kernel_size = (1, 1), strides = (1, 1), padding = 'same', 
               kernel_regularizer = regularizers.l2(weight_decay), name = base_name + '_conv_3_1')(x)
    x = BatchNormalization(axis = bn_axis, name = base_name + '_bn_3')(x)  
    x = Activation('relu', name = base_name + '_relu_3')(x)
    
    return x

def NIN16(input_shape, classes):
    inputs = Input(shape = input_shape)
    
    x = _initial_conv_block(inputs)
    
    x = _block(x, 64, 1)
    x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                     name = 'MaxPool_1')(x)
    
    x = _block(x, 128, 2)
    x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                     name = 'MaxPool_2')(x)
    
    x = _block(x, 256, 3)
    x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                     name = 'MaxPool_3')(x)
    
    x = _block(x, 512, 4)
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(512, activation = 'relu', name = 'fc_1')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation = 'softmax', name = 'fc_2')(x)
    
    model = Model(inputs = inputs, outputs = x, name = 'NIN16_model')
    plot_model(model, to_file = 'model.png',show_shapes = True, show_layer_names = True)
    return model



