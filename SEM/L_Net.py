# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Thu Aug 22 09:50:38 2019

@author: Admin
"""
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import Input

from keras import regularizers

from keras import backend as K
from keras.models import Model
from keras.utils import plot_model

weight_decay = 0.01

def _block(inputs, filters, block):
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = -1
    
    base_name = 'block' + '_' + str(block)
    
    x = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', 
               kernel_regularizer = regularizers.l2(weight_decay), name = base_name + '_conv_1')(inputs)
    x = BatchNormalization(axis = bn_axis, name = base_name + '_bn_1')(x)
    x = Activation('relu', name = base_name + '_relu_1')(x)
    
    x = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', 
               kernel_regularizer = regularizers.l2(weight_decay), name = base_name + '_conv_2')(x)
    x = BatchNormalization(axis = bn_axis, name = base_name + '_bn_2')(x)
    x = Activation('relu', name = base_name + '_relu_2')(x)
    
    return x

def _block_ds(inputs, filters, block):
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = -1
    
    base_name = 'block' + '_' + str(block)
    
    x = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', 
               kernel_regularizer = regularizers.l2(weight_decay), name = base_name + '_conv_1')(inputs)
    x = BatchNormalization(axis = bn_axis, name = base_name + '_bn_1')(x)
    x = Activation('relu', name = base_name + '_relu_1')(x)
    
    x = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', 
               kernel_regularizer = regularizers.l2(weight_decay), name = base_name + '_conv_2')(x)
    x = BatchNormalization(axis = bn_axis, name = base_name + '_bn_2')(x)
    x = Activation('relu', name = base_name + '_relu_2')(x)
    
    x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid')(x)
    
    return x

def _bridge_block(inputs, filters, block):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    base_name = 'bridge_block' + '_' + str(block)
    
    x = Conv2D(filters = filters, kernel_size = (1, 1), strides = (1, 1), padding = 'same', 
               use_bias = False, kernel_regularizer = regularizers.l2(weight_decay), name = base_name + '_conv_1')(inputs)
    x = BatchNormalization(axis = channel_axis, name = base_name + '_bn_1')(x)
    x = Activation('relu', name = base_name + '_relu_1')(x)
    
    x = Conv2D(filters = filters, kernel_size = (1, 1), strides = (1, 1), padding = 'same', 
               use_bias = False, kernel_regularizer = regularizers.l2(weight_decay), name = base_name + '_conv_2')(x)
    x = BatchNormalization(axis = channel_axis, name = base_name + '_bn_2')(x)
    x = Activation('relu', name = base_name + '_relu_2')(x)
    
    return x

def New_Network(input_shape, classes):
    inputs = Input(shape = input_shape)
#    The input shape is 224 x 224 x 3
    
    x_1 = _block_ds(inputs = inputs, filters = 64, block = 1)
#    The shape of x_1: 112 x 112 x 64
    
    x_2 = _block_ds(inputs = x_1, filters = 128, block = 2)
#    The shape of x_2: 56 x 56 x 128
    
    x_3 = _block(inputs = x_2, filters = 256, block = 3)
#    The shape of x_3: 56 x 56 x 256
    
    maxpool_1 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid')(x_3)
#    The shape of maxpool_1: 28 x 28 x 256
    
    bridge_1 = Concatenate(axis = -1)([x_2, x_3])
#    The shape of bridge_1: 56 x 56 x 384
    
    bridge_2 = _bridge_block(inputs = bridge_1, filters = 256, block = 1)
#    The shape of bridge_2: 56 x 56 x 256
    
    bridge_3 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid')(bridge_2)
#    The shape of bridge_3: 28 x 28 x 256
    
    x_4 = _block(inputs = maxpool_1, filters = 512, block = 4)
#    The shape of x_4: 28 x 28 x 512
    
    bridge_4 = Concatenate(axis = -1)([bridge_3, x_4])
#    The shape of bridge_4: 28 x 28 x 768
    
    bridge_5 = _bridge_block(inputs = bridge_4, filters = 512, block = 2)
#    The shape of bridge_5: 28 x 28 x 512
    
    maxpool_2 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid')(bridge_5)
#    The shape of maxpool_2: 14 x 14 x 512
    
    output = GlobalAveragePooling2D()(maxpool_2)
    output = Dense(512, activation = 'relu', name = 'fc_1')(output)
    output = Dropout(0.5)(output)
    output = Dense(classes, activation = 'softmax', name = 'fc_2')(output)
    
    model = Model(inputs = inputs, outputs = output, name = 'SEM_Net')
    
    plot_model(model, to_file = 'model_L_Net.png',show_shapes = True, show_layer_names = True)
    
    return model
    
    
