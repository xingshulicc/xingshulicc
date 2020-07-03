# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Mon Jun 15 11:41:08 2020

@author: Admin
"""
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import multiply
from keras.layers import Reshape
from keras.layers import Permute

from keras import backend as K

def squeeze_excite_block(input_tensor, ratio, block):
    '''
    the default value of ratio in the SENet paper is 16
    '''
    base_name = 'SE_block_' + str(block) + '_'
    
    inputs = input_tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = inputs._keras_shape[channel_axis]
    se_shape = (1, 1, filters)
    
    se = GlobalAveragePooling2D()(inputs)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, 
               activation = 'relu', 
               kernel_initializer = 'he_normal', 
               use_bias = False, 
               name = base_name + 'd1')(se)
    se = Dense(filters, 
               activation = 'sigmoid', 
               kernel_initializer = 'he_normal', 
               use_bias = False, 
               name = base_name + 'd2')(se)
    
    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)
    
    x = multiply([inputs, se])
    
    return x


#Note: the output dimension of GlobalAveragePooling2D is: (1, filters)
#So we need Reshape layer to convert it to (1, 1, filters)