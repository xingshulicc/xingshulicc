# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue Nov 19 19:10:09 2019

@author: Admin
"""
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D

from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Input

from keras import backend as K
from keras.models import Model

from keras import regularizers

from keras.utils import plot_model

weight_decay = 0.005

def _initial_conv_block(input):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Conv2D(filters = 32, 
               kernel_size = (7, 7), 
               strides = (2, 2), 
               padding = 'same', 
               kernel_regularizer = regularizers.l2(weight_decay), 
               name = 'init_conv')(input)
    x = BatchNormalization(axis = channel_axis, 
                           name = 'init_conv_bn')(x)
    x = Activation('relu', 
                   name = 'init_conv_relu')(x)
    
    return x

def pyramid_block(input, filters, block):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    base_name = 'block_' + str(block) + '_'
    
    x1 = Conv2D(filters = filters, 
                kernel_size = (1, 1), 
                strides = (1, 1), 
                padding = 'same', 
                use_bias = False, 
                kernel_regularizer = regularizers.l2(weight_decay), 
                name = base_name + 'conv_1')(input)
    x1 = BatchNormalization(axis = channel_axis, 
                            name = base_name + 'bn_1')(x1)
    x1 = Activation('relu', 
                    name = base_name + 'relu_1')(x1)
    
    x1 = Conv2D(filters = filters, 
                kernel_size = (1, 1), 
                strides = (1, 1), 
                padding = 'same', 
                use_bias = False, 
                kernel_regularizer = regularizers.l2(weight_decay), 
                name = base_name + 'conv_1_2')(x1)
    x1 = BatchNormalization(axis = channel_axis, 
                            name = base_name + 'bn_1_2')(x1)
    x1 = Activation('relu', 
                    name = base_name + 'relu_1_2')(x1)
    
    x1 = Add()([input, x1])
    
    x2 = Conv2D(filters = filters, 
                kernel_size = (3, 3), 
                strides = (1, 1), 
                padding = 'same', 
                kernel_regularizer = regularizers.l2(weight_decay), 
                name = base_name + 'conv_2')(x1)
    x2 = BatchNormalization(axis = channel_axis, 
                            name = base_name + 'bn_2')(x2)
    x2 = Activation('relu', 
                    name = base_name + 'relu_2')(x2)
    
    x2 = Add()([x1, x2])
    
    x3 = Conv2D(filters = filters, 
                kernel_size = (5, 5), 
                strides = (1, 1), 
                padding = 'same', 
                kernel_regularizer = regularizers.l2(weight_decay), 
                name = base_name + 'conv_3')(x2)
    x3 = BatchNormalization(axis = channel_axis, 
                            name = base_name + 'bn_3')(x3)
    x3 = Activation('relu', 
                    name = base_name + 'relu_3')(x3)
    
    x3 = Add()([x2, x3])
    
    return x3

def mean_max_pool(input, block):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    base_name = 'pooling_block_' + str(block) + '_'
    
    x_mean = AveragePooling2D(pool_size = (2, 2), 
                              strides = (2, 2), 
                              padding = 'same', 
                              name = base_name + 'avg')(input)
    x_max = MaxPooling2D(pool_size = (2, 2), 
                         strides = (2, 2), 
                         padding = 'same', 
                         name = base_name + 'max')(input)
    
    x = Concatenate(axis = channel_axis)([x_mean, x_max])
    
    return x


def end_block(input, filters):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    x1 = Conv2D(filters = filters, 
                kernel_size = (1, 1), 
                strides = (1, 1), 
                padding = 'same', 
                use_bias = False, 
                kernel_regularizer = regularizers.l2(weight_decay), 
                name = 'end_block_' + 'conv_1')(input)
    x1 = BatchNormalization(axis = channel_axis, 
                            name = 'end_block_' + 'bn_1')(x1)
    x1 = Activation('relu', 
                    name = 'end_block_' + 'relu_1')(x1)
    
    x1 = Conv2D(filters = filters, 
                kernel_size = (1, 1), 
                strides = (1, 1), 
                padding = 'same', 
                use_bias = False, 
                kernel_regularizer = regularizers.l2(weight_decay), 
                name = 'end_block_' + 'conv_1_2')(x1)
    x1 = BatchNormalization(axis = channel_axis, 
                            name = 'end_block_' + 'bn_1_2')(x1)
    x1 = Activation('relu', 
                    name = 'end_block_' + 'relu_1_2')(x1)
    
    x1 = Add()([input, x1])
    
    x = Conv2D(filters = filters, 
               kernel_size = (3, 3), 
               strides = (1, 1), 
               padding = 'same', 
               kernel_regularizer = regularizers.l2(weight_decay), 
               name = 'end_block_' + 'conv_2')(x1)
    x = BatchNormalization(axis = channel_axis, 
                           name = 'end_block_' + 'bn_2')(x)
    x = Activation('relu', 
                   name = 'end_block_' + 'relu_2')(x)
    
    
    x = Add()([x1, x])
    
    return x


def New_Net(input_shape, classes):
    inputs = Input(shape = input_shape)
    
    x_1 = _initial_conv_block(input = inputs)
#    The shape of x_1: 112 x 112 x 32
    
    x_2 = mean_max_pool(x_1, 1)
#    The shape of x_2: 56 x 56 x 64
    
    x_3 = pyramid_block(x_2, 64, 1)
#    The shape of x_3: 56 x 56 x 64
    
    x_4 = mean_max_pool(x_3, 2)
#    The shape of x_4: 28 x 28 x 128
    
    x_5 = pyramid_block(x_4, 128, 2)
#    The shape of x_5: 28 x 28 x 128
    
    x_6 = mean_max_pool(x_5, 3)
#    The shape of x_6: 14 x 14 x 256
    
    x_7 = pyramid_block(x_6, 256, 3)
#    The shape of x_7: 14 x 14 x 256
    
    x_8 = mean_max_pool(x_7, 4)
#    The shape of x_8: 7 x 7 x 512
    
    x_9 = end_block(x_8, 512)
#    The shape of x_9: 7 x 7 x 512
    
    output = GlobalAveragePooling2D()(x_9)
    output = Dense(512, activation = 'relu', name = 'fc_1')(output)
    output = Dropout(0.5)(output)
    output = Dense(classes, activation = 'softmax', name = 'fc_2')(output)
    
    model = Model(inputs = inputs, outputs = output, name = 'Pyramid_Network')
    plot_model(model, to_file = 'pyramid_net.png',show_shapes = True, show_layer_names = True)
    
    return model
    
    
    
    
    
    
