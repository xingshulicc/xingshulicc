# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Jul 24 15:04:44 2019

@author: Admin
"""
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D

from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Concatenate
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

def _trans_layer(inputs, block):
    base_name = 'trans_layer' + '_' + str(block)
    
    x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                     name = base_name + '_MaxPool')(inputs)
    
    return x

def _block(inputs, filters, block):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    base_name = 'block' + '_' + str(block)
    
    x = Conv2D(filters = filters, kernel_size = (3, 3), strides = 1, 
               padding = 'same', name = base_name + '_conv')(inputs)
    x = BatchNormalization(axis = channel_axis, name = base_name + '_conv_bn')(x)
    x = Activation('relu', name = base_name + '_conv_relu')(x)
    
    return x

def _bridge_block(inputs, filters, block):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    base_name = 'bridge_block' + '_' + str(block)
    
    x = Conv2D(filters = filters, kernel_size = (1, 1), strides = (1, 1), 
               padding = 'same', use_bias = False, name = base_name + '_conv_1')(inputs)
    x = BatchNormalization(axis = channel_axis, name = base_name + '_bn_1')(x)
    x = Activation('relu', name = base_name + '_relu_1')(x)
    
    x = Conv2D(filters = filters, kernel_size = (1, 1), strides = (1, 1), 
               padding = 'same', use_bias = False, name = base_name + '_conv_2')(x)
    x = BatchNormalization(axis = channel_axis, name = base_name + '_bn_2')(x)
    x = Activation('relu', name = base_name + '_relu_2')(x)
    
    return x

def test_net(input_shape, classes):
    inputs = Input(shape = input_shape)
#    The input_shape: 224 x 224 x 3
    
    x_1 = _init_conv_block(inputs = inputs, filters = 32, kernel_size = (5, 5), strides = (2, 2))
#    The shape of x_1: 112 x 112 x 32
    
    x_2 = _block(inputs = x_1, filters = 32, block = 1)
#    The shape of x_2: 112 x 112 x 32
    
    bridge_1 = _bridge_block(inputs = x_1, filters = 32, block = 1)
#    The shape of bridge_1: 112 x 112 x 32
    
    concatenate_1 = Concatenate(axis = -1)([x_2, bridge_1])
#    The shape of concatenate_1: 112 x 112 x 64
    
    x_3 = _trans_layer(inputs = concatenate_1, block = 1)
#    The shape of x_3: 56 x 56 x 64
    
    x_4 = _block(inputs = x_3, filters = 64, block = 2)
#    The shape of x_4: 56 x 56 x 64
    
    bridge_2 = _bridge_block(inputs = x_3, filters = 64, block = 2)
#    The shape of bridge_2: 56 x 56 x 64
    
    concatenate_2 = Concatenate(axis = -1)([x_4, bridge_2])
#    The shape of concatenate_2: 56 x 56 x 128
    
    x_5 = _trans_layer(inputs = concatenate_2, block = 2)
#    The shape of x_5: 28 x 28 x 128
    
    x_6 = _block(inputs = x_5, filters = 128, block = 3)
#    The shape of x_6: 28 x 28 x 128
    
    bridge_3 = _bridge_block(inputs = x_5, filters = 128, block = 3)
#    The shape of bridge_3: 28 x 28 x 128
    
    concatenate_3 = Concatenate(axis = -1)([x_6, bridge_3])
#    The shape of concatenate_3: 28 x 28 x 256
    
    x_7 = _trans_layer(inputs = concatenate_3, block = 3)
#    The shape of x_7: 14 x 14 x 256
    
    x_8 = _block(inputs = x_7, filters = 256, block = 4)
#    The shape of x_8: 14 x 14 x 256
    
    bridge_4 = _bridge_block(inputs = x_7, filters = 256, block = 4)
#    The shape of bridge_4: 14 x 14 x 256
    
    concatenate_4 = Concatenate(axis = -1)([x_8, bridge_4])
#    The shape of concatenate_4: 14 x 14 x 512
    
    x_9 = _trans_layer(inputs = concatenate_4, block = 4)
#    The shape of x_9: 7 x 7 x 512
    
    x_10 = _block(inputs = x_9, filters = 512, block = 5)
#    The shape of x_10: 7 x 7 x 512
    
    bridge_5 = _bridge_block(inputs = x_9, filters = 512, block = 5)
#    The shape of bridge_5: 7 x 7 x 512
    
    concatenate_5 = Concatenate(axis = -1)([x_10, bridge_5])
#    The shape of concatenate_5: 7 x 7 x 1024
    
    x = Conv2D(filters = 512, kernel_size = (1, 1), strides = 1, padding = 'same')(concatenate_5)
    x = BatchNormalization(axis = -1)(x)
    x = Activation('relu')(x)
#    The shape of x: 7 x 7 x 512
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation = 'relu', name = 'fc_1')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation = 'softmax', name = 'fc_2')(x)
    
    model = Model(inputs = inputs, outputs = x, name = 'new_net')
    
    plot_model(model, to_file = 'Newbase_model.png', show_shapes = True, show_layer_names = True)
    
    return model


    
    
