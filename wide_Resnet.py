# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Feb 27 13:53:45 2019

@author: xingshuli
"""
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D

from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Add
from keras.layers import Input

from keras import backend as K
from keras.models import Model

def _initial_conv_block(input):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Conv2D(filters = 32, kernel_size = (7, 7), strides = (2, 2), 
               padding = 'same', name = 'init_conv')(input)
    x = BatchNormalization(axis = channel_axis, name = 'init_conv_bn')(x)
    x = Activation('relu', name = 'init_conv_relu')(x)
    
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same', 
                     name = 'init_MaxPool')(x)
    
    return x

def conv1_block(inputs, filters, block):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    base_name = 'block' + '_' + str(block)
    
    x = Conv2D(filters = filters, kernel_size = (1, 1), strides = (1, 1), 
               padding = 'same', name = base_name + '_conv_1x1')(inputs)
    x = BatchNormalization(axis = channel_axis, name = base_name + '_bn_1x1')(x)
    x = Activation('relu', name = base_name + '_relu_1x1')(x)
    
    return x

def wide_res_block(inputs, filters, block):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    base_name = 'block' + '_' + str(block)
    
    x = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), 
               padding = 'same', name = base_name + '_conv_3_1')(inputs)
    x = BatchNormalization(axis = channel_axis, name = base_name + '_bn_3_1')(x)
    x = Activation('relu', name = base_name + '_relu_3_1')(x)
    
    x = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), 
               padding = 'same', name = base_name + '_conv_3_2')(x)
    x = BatchNormalization(axis = channel_axis, name = base_name + '_bn_3_2')(x)
    x = Activation('relu', name = base_name + '_relu_3_2')(x)
    
    res_out = Add()([inputs, x])
    
    return res_out

def wide_resnet(input_shape, classes):
    inputs = Input(shape = input_shape)
    
    x_1 = _initial_conv_block(inputs)
#    The shape of x_1: 56 x 56 x 32
    
    x_2 = conv1_block(x_1, 64, 1)
#    The shape of x_2: 56 x 56 x 64
    
    x_3 = wide_res_block(x_2, 64, 1)
#    The shape of x_3: 56 x 56 x 64
    
    Pool_1 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                          name = 'MaxPool_1')(x_3)
#    The shape of Pool_1: 28 x 28 x 64
    
    x_4 = conv1_block(Pool_1, 128, 2)
#    The shape of x_4: 28 x 28 x 128
    
    x_5 = wide_res_block(x_4, 128, 2)
#    The shape of x_5: 28 x 28 x 128
    
    Pool_2 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                          name = 'MaxPool_2')(x_5)
#    The shape of Pool_2: 14 x 14 x 128
    
    x_6 = conv1_block(Pool_2, 256, 3)
#    The shape of x_6: 14 x 14 x 256
    
    x_7 = wide_res_block(x_6, 256, 3)
#    The shape of x_7: 14 x 14 x 256
    
    Pool_3 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                          name = 'MaxPool_3')(x_7)
#    The shape of Pool_3: 7 x 7 x 256
    
    x_8 = conv1_block(Pool_3, 512, 4)
#    The shape of x_8: 7 x 7 x 512
    
    x_9 = wide_res_block(x_8, 512, 4)
#    The shape of x_9: 7 x7 x 512
    
    output = GlobalAveragePooling2D()(x_9)
    output = Dense(512, activation = 'relu', name = 'fc_1')(output)
    output = Dropout(0.5)(output)
    output = Dense(classes, activation = 'softmax', name = 'fc_2')(output)
    
    model = Model(inputs = inputs, outputs = output, name = 'Wide_ResNet')
    
    return model

    