# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Apr 24 10:35:26 2019

@author: xingshuli
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

from keras import regularizers

from keras.utils import plot_model

from squeeze_excitation_layer import squeeze_excite_block

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


def _MLP_block(inputs, filters, block):
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = -1
    
    base_name = 'block' + '_' + str(block)
    
    x = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', 
               kernel_regularizer = regularizers.l2(weight_decay), name = base_name + '_conv_1')(inputs)
    x = BatchNormalization(axis = bn_axis, name = base_name + '_bn_1')(x)
    x = Activation('relu', name = base_name + '_relu_1')(x)
    
    x = Conv2D(filters = filters, kernel_size = (1, 1), strides = (1, 1), padding = 'same', 
               use_bias = False, kernel_regularizer = regularizers.l2(weight_decay), name = base_name + '_conv_2')(x)
    x = BatchNormalization(axis = bn_axis, name = base_name + '_bn_2')(x)
    x = Activation('relu', name = base_name + '_relu_2')(x)
    
    x = Conv2D(filters = filters, kernel_size = (1, 1), strides = (1, 1), padding = 'same', 
               use_bias = False, kernel_regularizer = regularizers.l2(weight_decay), name = base_name + '_conv_3')(x)
    x = BatchNormalization(axis = bn_axis, name = base_name + '_bn_3')(x)
    x = Activation('relu', name = base_name + '_relu_3')(x)
    
    return x


def _end_block(inputs, filters):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    base_name = 'end_block'
    
    x = Conv2D(filters = filters, kernel_size = (1, 1), strides = (1, 1), padding = 'same', 
               use_bias = False, kernel_regularizer = regularizers.l2(weight_decay), name = base_name + '_conv_1')(inputs)
    x = BatchNormalization(axis = channel_axis, name = base_name + '_bn_1')(x)
    x = Activation('relu', name = base_name + '_relu_1')(x)
    
    x = Conv2D(filters = filters, kernel_size = (1, 1), strides = (1, 1), padding = 'same', 
               use_bias = False, kernel_regularizer = regularizers.l2(weight_decay), name = base_name + '_conv_2')(x)
    x = BatchNormalization(axis = channel_axis, name = base_name + '_bn_2')(x)
    x = Activation('relu', name = base_name + '_relu_2')(x)
    
    return x


def New_net(input_shape, classes):
    inputs = Input(shape = input_shape)
    
    x_1 = _initial_conv_block(input = inputs)
#    The shape of x_1: 56 x 56 x 32
    
    x_2 = _MLP_block(inputs = x_1, filters = 64, block = 1)
#    The shape of x_2: 56 x 56 x 64
    
    Pool_1 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                          name = 'MaxPool_1')(x_2)
#    The shape of Pool_1: 28 x 28 x 64
    
    x_3 = Concatenate(axis = -1)([x_1, x_2])
#    The shape of x_3: 56 x 56 x 96
    
    x_4 = _MLP_block(inputs = x_3, filters = 128, block = 2)
#    The shape of x_4: 56 x 56 x 128
    
    Pool_2 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                          name = 'MaxPool_2')(x_4)
#    The shape of Pool_2: 28 x 28 x 128

    x_5 = Concatenate(axis = -1)([Pool_1, Pool_2])
#    The shape of x_5: 28 x 28 x 192
    
    x_6 = _MLP_block(inputs = x_5, filters = 256, block = 3)
#    The shape of x_6: 28 x 28 x 256
    
    Pool_3 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                          name = 'MaxPool_3')(x_6)
#    The shape of Pool_3: 14 x 14 x 256
    
    x_7 = Concatenate(axis = -1)([Pool_2, x_6])
#    The shape of x_7: 28 x 28 x 384
    
    x_8 = _MLP_block(inputs = x_7, filters = 256, block = 4)
#    The shape of x_8: 28 x 28 x 256
    
    Pool_4 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                          name = 'MaxPool_4')(x_8)
#    The shape of Pool_4: 14 x 14 x 256
    
    SE_1 = squeeze_excite_block(Pool_4, ratio = 4)
#    The shape of SE_1: 14 x 14 x 256 
    
    x_9 = Concatenate(axis = -1)([Pool_3, Pool_4])
#    The shape of x_9: 14 x 14 x 512
    
    x_10 = _MLP_block(inputs = x_9, filters = 512, block = 5)
#    The shape of x_10: 14 x 14 x 512
    
    Pool_5 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                          name = 'MaxPool_5')(x_10)
#    The shape of Pool_5: 7 x 7 x 512
                          
    x_11 = Concatenate(axis = -1)([Pool_4, x_10])
#    The shape of x_11: 14 x 14 x 768
    
    x_12 = _MLP_block(inputs = x_11, filters = 512, block = 6)
#    The shape of x_12: 14 x 14 x 512
    
    Pool_6 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                          name = 'MaxPool_6')(x_12)
#    The shape of Pool_6: 7 x 7 x 512
    
    SE_2 = squeeze_excite_block(Pool_6, ratio = 4)
#    The shape of SE_2: 7 x 7 x 512 
    
    x_13 = Concatenate(axis = -1)([Pool_5, Pool_6])
#    The shape of x_13: 7 x 7 x 1024
    
    x_14 = _end_block(inputs = x_13, filters = 512)
#    The shape of x_14: 7 x 7 x 512
    
    Pool_7 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', 
                          name = 'MaxPool_7')(x_14)
#    The shape of Pool_7: 4 x 4 x 512
    
    SE_3 = squeeze_excite_block(Pool_7, ratio = 4)
#    The shape of SE_3: 4 x 4 x 512
    
    concate_1 = GlobalAveragePooling2D()(SE_1)
    concate_2 = GlobalAveragePooling2D()(SE_2)
    concate_3 = GlobalAveragePooling2D()(SE_3)
    x = Concatenate(axis = -1)([concate_1, concate_2, concate_3])
#    The shape of x: 1 x 1 x 1280
    
    output = Dense(1280, activation = 'relu', name = 'fc_1')(x)
    output = Dropout(0.5)(output)
    output = Dense(classes, activation = 'softmax', name = 'fc_2')(output)
    
    model = Model(inputs = inputs, outputs = output, name = 'multi_scale_net_v3')
    plot_model(model, to_file = 'model_Rnet_v3.png',show_shapes = True, show_layer_names = True)
    
    return model 


