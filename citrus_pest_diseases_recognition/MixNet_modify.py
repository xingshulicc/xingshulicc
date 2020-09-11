# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Fri Sep 11 09:56:26 2020

@author: Admin
"""
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D

from keras.layers import Input
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Concatenate

from squeeze_excitation_layer import squeeze_excite_block

from keras import backend as K
from keras.models import Model
from keras.utils import plot_model

from keras import regularizers

weight_decay = 0.001

if K.image_data_format() == 'channels_first':
    bn_axis = 1
else:
    bn_axis = -1

def _initial_conv_block(input_tensor, filters, kernel_size, strides):
    base_name = 'initial_block_'
    x = Conv2D(filters = filters, 
               kernel_size = kernel_size, 
               strides = strides, 
               padding = 'same', 
               kernel_regularizer = regularizers.l2(weight_decay), 
               name = base_name + 'conv')(input_tensor)
    x = BatchNormalization(axis = bn_axis, name = base_name + 'bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(x)
    
    return x

def _mix_conv_1(input_tensor, filters, kernel_size, block):
    base_name = 'mix_block_' + str(block) + '_'
    k_1, k_2, k_3 = kernel_size
    
    br_1 = DepthwiseConv2D(kernel_size = k_1, 
                           strides = (1, 1), 
                           padding = 'same', 
                           depth_multiplier = 1, 
                           name = base_name + 'conv_1')(input_tensor)
    br_1 = BatchNormalization(axis = bn_axis, name = base_name + 'bn_1')(br_1)
    br_1 = Activation('relu')(br_1)
    
    br_2 = DepthwiseConv2D(kernel_size = k_2, 
                           strides = (1, 1), 
                           padding = 'same', 
                           depth_multiplier = 1, 
                           name = base_name + 'conv_2')(input_tensor)
    br_2 = BatchNormalization(axis = bn_axis, name = base_name + 'bn_2')(br_2)
    br_2 = Activation('relu')(br_2)
    
    br_3 = DepthwiseConv2D(kernel_size = k_3, 
                           strides = (1, 1), 
                           padding = 'same', 
                           depth_multiplier = 1, 
                           name = base_name + 'conv_3')(input_tensor)
    br_3 = BatchNormalization(axis = bn_axis, name = base_name + 'bn_3')(br_3)
    br_3 = Activation('relu')(br_3)
    
    c = Concatenate(axis = -1)([br_1, br_2, br_3])
    c = squeeze_excite_block(c, 2, block)
    
    mix = Conv2D(filters = filters, 
                 kernel_size = (1, 1), 
                 strides = (1, 1), 
                 name = base_name + 'conv_mix_1')(c)
    mix = BatchNormalization(axis = bn_axis, name = base_name + 'bn_mix_1')(mix)
    mix = Activation('relu')(mix)
    
    mix = Conv2D(filters = filters, 
                 kernel_size = (1, 1), 
                 strides = (1, 1), 
                 name = base_name + 'conv_mix_2')(mix)
    mix = BatchNormalization(axis = bn_axis, name = base_name + 'bn_mix_2')(mix)
    mix = Activation('relu')(mix)
    
    return mix

def _mix_conv_2(input_tensor, filters, kernel_size, block):
    base_name = 'mix_block_' + str(block) + '_'
    k_1, k_2 = kernel_size
    
    br_1 = DepthwiseConv2D(kernel_size = k_1, 
                           strides = (1, 1), 
                           padding = 'same', 
                           depth_multiplier = 1, 
                           name = base_name + 'conv_1')(input_tensor)
    br_1 = BatchNormalization(axis = bn_axis, name = base_name + 'bn_1')(br_1)
    br_1 = Activation('relu')(br_1)
    
    br_2 = DepthwiseConv2D(kernel_size = k_2, 
                           strides = (1, 1), 
                           padding = 'same', 
                           depth_multiplier = 1, 
                           name = base_name + 'conv_2')(input_tensor)
    br_2 = BatchNormalization(axis = bn_axis, name = base_name + 'bn_2')(br_2)
    br_2 = Activation('relu')(br_2)
    
    c = Concatenate(axis = -1)([br_1, br_2])
    c = squeeze_excite_block(c, 2, block)
    
    mix = Conv2D(filters = filters, 
                 kernel_size = (1, 1), 
                 strides = (1, 1), 
                 name = base_name + 'conv_mix_1')(c)
    mix = BatchNormalization(axis = bn_axis, name = base_name + 'bn_mix_1')(mix)
    mix = Activation('relu')(mix)
    
    mix = Conv2D(filters = filters, 
                 kernel_size = (1, 1), 
                 strides = (1, 1), 
                 name = base_name + 'conv_mix_2')(mix)
    mix = BatchNormalization(axis = bn_axis, name = base_name + 'bn_mix_2')(mix)
    mix = Activation('relu')(mix)
    
    return mix

def _mix_conv_3(input_tensor, filters, kernel_size, block):
    base_name = 'mix_block_' + str(block) + '_'
    
    br_1 = DepthwiseConv2D(kernel_size = kernel_size, 
                           strides = (1, 1), 
                           padding = 'same', 
                           depth_multiplier = 1, 
                           name = base_name + 'conv_1')(input_tensor)
    br_1 = BatchNormalization(axis = bn_axis, name = base_name + 'bn_1')(br_1)
    br_1 = Activation('relu')(br_1)
    
    se = squeeze_excite_block(br_1, 2, block)
    
    mix = Conv2D(filters = filters, 
                 kernel_size = (1, 1), 
                 strides = (1, 1), 
                 name = base_name + 'conv_mix_1')(se)
    mix = BatchNormalization(axis = bn_axis, name = base_name + 'bn_mix_1')(mix)
    mix = Activation('relu')(mix)
    
    mix = Conv2D(filters = filters, 
                 kernel_size = (1, 1), 
                 strides = (1, 1), 
                 name = base_name + 'conv_mix_2')(mix)
    mix = BatchNormalization(axis = bn_axis, name = base_name + 'bn_mix_2')(mix)
    mix = Activation('relu')(mix)
    
    return mix

def MMix_Net(input_shape, classes):
    inputs = Input(shape = input_shape)
    
    x_1 = _initial_conv_block(inputs, 32, (7, 7), (2, 2))
#    The shape of x_1: 56 x 56 x 32
    x_2 = _mix_conv_1(x_1, 64, [3, 5, 7], 1)
#    The shape of x_2: 56 x 56 x 64
    x_3 = _mix_conv_1(x_2, 128, [3, 5, 7], 2)
#    The shape of x_3: 56 x 56 x 128
    c_1 = Concatenate(axis = -1)([x_2, x_3])
#    The shape of c_1: 56 x 56 x 192
    p_1 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(c_1)
#    The shape of p_1: 28 x 28 x 192
    x_4 = _mix_conv_2(p_1, 192, [3, 5], 3)
#    The shape of x_4: 28 x 28 x 192
    x_5 = _mix_conv_2(x_4, 384, [3, 5], 4)
#    The shape of x_5: 28 x 28 x 384
    c_2 = Concatenate(axis = -1)([x_4, x_5])
#    The shape of c_2: 28 x 28 x 576
    p_2 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(c_2)
#    The shape of p_2: 14 x 14 x 576
    x_6 = _mix_conv_3(p_2, 576, 3, 5)
#    The shape of x_6: 14 x 14 x 576
    x_7 = _mix_conv_3(x_6, 576, 3, 6)
#    The shape of x_7: 14 x 14 x 576
    c_3 = Concatenate(axis = -1)([x_6, x_7])
#    The shape of c_3: 14 x 14 x 1152
    p_3 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(c_3)
#    The shape of p_3: 7 x 7 x 1152
    
    output = GlobalAveragePooling2D()(p_3)
    output = Dense(512, activation = 'relu', name = 'fc_1')(output)
    output = Dropout(rate = 0.5, name = 'dropout')(output)
    output = Dense(classes, activation = 'softmax', name = 'fc_2')(output)
    
    model = Model(inputs = inputs, outputs = output, name = 'Mix_NN')
    plot_model(model, to_file = 'MixNet.png',show_shapes = True, show_layer_names = True)
    
    return model





