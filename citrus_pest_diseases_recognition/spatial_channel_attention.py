# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue Nov 12 13:46:36 2019

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

from keras.layers import DepthwiseConv2D
from keras.layers import ELU

from keras import backend as K
from keras.models import Model

from keras import regularizers

from keras.utils import plot_model

weight_decay = 0.005

def _initial_conv_block(input):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Conv2D(filters = 32, kernel_size = (7, 7), strides = (2, 2), padding = 'same', 
               kernel_regularizer = regularizers.l2(weight_decay), name = 'init_conv')(input)
    x = BatchNormalization(axis = channel_axis, name = 'init_conv_bn')(x)
    x = Activation('relu', name = 'init_conv_relu')(x)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same', 
                     name = 'init_MaxPool')(x)
    
    return x

def _spatial_attention(input, dilation_rate, kernel_size, block):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    x = DepthwiseConv2D(kernel_size = kernel_size, 
                        strides = 1, 
                        padding = 'same', 
                        dilation_rate = dilation_rate, 
                        use_bias = False, 
                        depthwise_regularizer = regularizers.l2(weight_decay), 
                        name = 'Spatial_conv' + '_' + str(block))(input)
    
    x = BatchNormalization(axis = channel_axis, name = 'Spatial_bn' + '_' + str(block))(x)
    x = Activation('relu', name = 'Spatial_relu' + '_' + str(block))(x)
    
    return x

def _feature_enhance(input, filters, block):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    x = Conv2D(filters = filters, 
               kernel_size = (1, 1), 
               strides = (1, 1), 
               padding = 'same', 
               use_bias = False, 
               kernel_regularizer = regularizers.l2(weight_decay), 
               name = 'Enhance_1_conv' + '_' + str(block))(input)
    x = BatchNormalization(axis = channel_axis, name = 'Enhance_1_bn' + '_' + str(block))(x)
    x = ELU(alpha = 1.0, name = 'Enhance_1_elu' + '_' + str(block))(x)
    
    x = Conv2D(filters = filters, 
               kernel_size = (1, 1), 
               strides = (1, 1), 
               padding = 'same', 
               use_bias = False, 
               kernel_regularizer = regularizers.l2(weight_decay), 
               name = 'Enhance_2_conv' + '_' + str(block))(x)
    x = BatchNormalization(axis = channel_axis, name = 'Enhance_2_bn' + '_' + str(block))(x)
    x = Activation('relu', name = 'Enhance_2_relu' + '_' + str(block))(x)
    
    return x

def _mean_max_pool(input):
    x_1 = AveragePooling2D(pool_size = (2, 2), 
                           strides = (2, 2), 
                           padding = 'same')(input)
    
    x_2 = MaxPooling2D(pool_size = (2, 2), 
                       strides = (2, 2), 
                       padding = 'same')(input)
    
    x = Add()([x_1, x_2])
    
    return x

def _building_block(input, dilation_rate, kernel_size, block, filters):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    x = _spatial_attention(input, dilation_rate, kernel_size, block)
    
    x = Conv2D(filters = filters, 
               kernel_size = (3, 3), 
               strides = (1, 1), 
               padding = 'same', 
               kernel_regularizer = regularizers.l2(weight_decay), 
               name = 'block_' + str(block) + '_regular_conv_1')(x)
    x = BatchNormalization(axis = channel_axis, name = 'block_' + str(block) + '_regular_bn_1')(x)
    x = Activation('relu', name = 'block_' + str(block) + '_regular_relu_1')(x)
    
    x = _feature_enhance(x, filters, block)
    
    return x

def new_model(input_shape, classes):
    inputs = Input(shape = input_shape)
    
    x1 = _initial_conv_block(input = inputs)
#    The shape of x1: 56 x 56 x 32
    
    x2 = _building_block(x1, 3, 5, 1, 64)
#    The shape of x2: 56 x 56 x 64
    
    pool_1 = _mean_max_pool(x2)
#    The shape of pool_1: 28 x 28 x 64
    
    concat_1 = Concatenate(axis = -1)([x1, x2])
#    The shape of concat_1: 56 x 56 x 96
    
    x3 = _building_block(concat_1, 3, 5, 2, 128)
#    The shape of x3: 56 x 56 x 128
    
    pool_2 = _mean_max_pool(x3)
#    The shape of pool_2: 28 x 28 x 128
    
    concat_2 = Concatenate(axis = -1)([pool_1, pool_2])
#    The shape of concat_2: 28 x 28 x 192
    
    x4 = _building_block(concat_2, 3, 3, 3, 256)
#    The shape of x4: 28 x 28 x 256
    
    pool_3 = _mean_max_pool(x4)
#    The shape of pool_3: 14 x 14 x 256
    
    concat_3 = Concatenate(axis = -1)([pool_2, x4])
#    The shape of concat_3: 28 x 28 x 384
    
    x5 = _building_block(concat_3, 3, 3, 4, 512)
#    The shape of x5: 28 x 28 x 512
    
    pool_4 = _mean_max_pool(x5)
#    The shape of pool_4: 14 x 14 x 512
    
    concat_4 = Concatenate(axis = -1)([pool_3, pool_4])
#    The shape of concat_4: 14 x 14 x 768
    
    x6 = _building_block(concat_4, 2, 3, 5, 512)
#    The shape of x6: 14 x 14 x 512
    
    pool_5 = _mean_max_pool(x6)
#    The shape of pool_5: 7 x 7 x 512
    
    bridge_1 = _feature_enhance(concat_3, 256, 6)
#    The shape of bridge_1: 28 x 28 x 256
    
    bridge_pool_1 = _mean_max_pool(bridge_1)
#    The shape of bridge_pool_1: 14 x 14 x 256
    
    bridge_concat_1 = Concatenate(axis = -1)([bridge_pool_1, pool_4])
#    The shape of bridge_concat_1: 14 x 14 x 768
    
    bridge_2 = _feature_enhance(bridge_concat_1, 512, 7)
#    The shape of bridge_2: 14 x 14 x 512
    
    bridge_pool_2 = _mean_max_pool(bridge_2)
#    The shape of bridge_pool_2: 7 x 7 x 512
    
    bridge_concat_2 = Concatenate(axis = -1)([bridge_pool_2, pool_5])
#    The shape of bridge_concat_2: 7 x 7 x 1024
    
    bridge_3 = _feature_enhance(bridge_concat_2, 512, 8)
#    The shape of bridge_3: 7 x 7 x 512
    
    output = GlobalAveragePooling2D()(bridge_3)
    output = Dense(512, activation = 'relu', name = 'fc_1')(output)
    output = Dropout(0.5)(output)
    output = Dense(classes, activation = 'softmax', name = 'fc_2')(output)
    
    model = Model(inputs = inputs, outputs = output, name = 'sca_WdenseNet')
    plot_model(model, to_file = 'model_sca_Wdensenet.png',show_shapes = True, show_layer_names = True)
    
    return model
    
    
