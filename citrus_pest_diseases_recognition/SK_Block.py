# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Oct  7 09:36:23 2020

@author: Admin
"""
import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D

from keras.layers import BatchNormalization
from keras.layers import add
from keras.layers import multiply
from keras.layers import Reshape
from keras.layers import Lambda
from keras import backend as K

if K.image_data_format() == 'channels_first':
    bn_axis = 1
else:
    bn_axis = -1

def _select_kernel(inputs, kernels, filters, ratio, block):
    '''
    kernels = [3, 5]
    ratio = 4 or 8
    '''
    
    base_name = 'sk_block_' + str(block) + '_'
    inputs_shape = tf.shape(inputs)
    b, h, w = inputs_shape[0], inputs_shape[1], inputs_shape[2]
    k_1, k_2 = kernels
    k_length = len(kernels)
    
    x_1 = Conv2D(filters = filters, 
                 kernel_size = k_1, 
                 strides = (1, 1), 
                 padding = 'same', 
                 name = base_name + 'conv_1')(inputs)
    x_1 = BatchNormalization(axis = bn_axis, name = base_name + 'bn_1')(x_1)
    x_1 = Activation('relu')(x_1)
    
    x_2 = Conv2D(filters = filters, 
                 kernel_size = k_2, 
                 strides = (1, 1), 
                 padding = 'same', 
                 name = base_name + 'conv_2')(inputs)
    x_2 = BatchNormalization(axis = bn_axis, name = base_name + 'bn_2')(x_2)
    x_2 = Activation('relu')(x_2)
    
    x_stack = [x_1, x_2]
    x_stack = Lambda(lambda x: tf.stack(x, axis = -1), 
                     output_shape = (b, h, w, filters, k_length))(x_stack)
#    print(x_stack.get_shape())
    
    x_a = add([x_1, x_2])
    
    se_shape = (1, 1, filters)
    se = GlobalAveragePooling2D()(x_a)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, 
               activation = 'relu', 
               kernel_initializer = 'he_normal', 
               use_bias = False, 
               name = base_name + 'dense_1')(se)
    se = Dense(filters * k_length,  
               kernel_initializer = 'he_normal', 
               use_bias = False, 
               name = base_name + 'dense_2')(se)
    se_reshape = (1, 1, filters, k_length)    
    se = Reshape(se_reshape)(se)
    se = Activation('softmax')(se)
    
    x = multiply([x_stack, se])
#    print(x.get_shape())
    x = Lambda(lambda x: tf.reduce_sum(x, axis = -1), 
               output_shape = (b, h, w, filters))(x)
#    print(x.get_shape())
    
    return x

#The implementation of SK-Net (Figure 1 in the paper: https://arxiv.org/pdf/1903.06586.pdf)
