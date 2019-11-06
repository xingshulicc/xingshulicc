# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Aug 28 09:24:42 2019

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
from keras.layers import Concatenate

from keras import backend as K
from keras.models import Model
from keras.utils import plot_model

from keras import regularizers
weight_decay = 0.005

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
    
    x = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', 
               name = base_name + '_conv_4')(x)
    x = BatchNormalization(axis = bn_axis, name = base_name + '_bn_4')(x)
    x = Activation('relu', name = base_name + '_relu_4')(x) 
    
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



def Bridge_VGG(input_shape, classes):
    inputs = Input(shape = input_shape)
    
    x_1 = _block1(inputs = inputs, filters = 64, block = 1)
#    The shape of x_1: 112 x 112 x 64
    
    x_2 = _block1(inputs = x_1, filters = 128, block = 2)
#    The shape of x_2: 56 x 56 x 128
    
    x_3 = _block2(inputs = x_2, filters = 256, block = 3)
#    The shape of x_3: 56 x 56 x 256
    
    bridge_1 = Concatenate(axis = -1)([x_2, x_3])
#    The shape of bridge_1: 56 x 56 x 384
    
    bridge_2 = _bridge_block(inputs = bridge_1, filters = 256, block = 1)
#    The shape of bridge_2: 56 x 56 x 256
    
    maxpool_1 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                             name = 'MaxPool_1')(x_3)
#    The shape of maxpool_1: 28 x 28 x 256
    
    bridge_pool_1 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                                 name = 'MaxPool_bridge_1')(bridge_2)
#    The shape of bridge_pool_1: 28 x 28 x 256
    
    x_4 = _block2(inputs = maxpool_1, filters = 512, block = 4)
#    The shape of x_4: 28 x 28 x 512
    
    bridge_3 = Concatenate(axis = -1)([bridge_pool_1, x_4])
#    The shape of bridge_3: 28 x 28 x 768
    
    bridge_4 = _bridge_block(inputs = bridge_3, filters = 512, block = 2)
#    The shape of bridge_4: 28 x 28 x 512
    
    maxpool_2 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                             name = 'MaxPool_2')(x_4)
#    The shape of maxpool_2: 14 x 14 x 512
    
    bridge_pool_2 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                                 name = 'MaxPool_bridge_2')(bridge_4)
#    The shape of bridge_pool_2: 14 x 14 x 512
    
    x_5 = _block2(inputs = maxpool_2, filters = 512, block = 5)
#    The shape of x_5: 14 x 14 x 512
    
    bridge_5 = Concatenate(axis = -1)([bridge_pool_2, x_5])
#    The shape of x_5: 14 x 14 x 1024
    
    bridge_6 = _bridge_block(inputs = bridge_5, filters = 512, block = 3)
#    The shape of x_6: 14 x 14 x 512
    
    bridge_pool_3 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                                 name = 'MaxPool_bridge_3')(bridge_6)
#    The shape of bridge_pool_3: 7 x 7 x 512
    
    output = GlobalAveragePooling2D()(bridge_pool_3)
    output = Dense(512, activation = 'relu', name = 'fc_1')(output)
    output = Dropout(0.5)(output)
    output = Dense(classes, activation = 'softmax', name = 'fc_2')(output)
    
    model = Model(inputs = inputs, outputs = output, name = 'bridge_vgg19')
    plot_model(model, to_file = 'model_bridge_vgg19.png',show_shapes = True, show_layer_names = True)
    
    return model






