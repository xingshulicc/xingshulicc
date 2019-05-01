# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Fri Apr 30 10:46:00 2019

@author: xingshuli
"""
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Add
from keras.layers import Concatenate
from keras.layers import Multiply
from keras.layers import Reshape
from keras.layers import Permute
from keras.layers import Lambda

from keras import backend as K


def channel_attention(input_feature, ratio):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    channel = input_feature._keras_shape[channel_axis]
    
    shared_layer_hidden = Dense(channel // ratio, 
                                activation = 'relu', 
                                use_bias = True, 
                                kernel_initializer = 'he_normal', 
                                bias_initializer = 'Zeros')
    shared_layer_output = Dense(channel, 
                                use_bias = True, 
                                kernel_initializer = 'he_normal', 
                                bias_initializer = 'Zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_hidden(avg_pool)
    avg_pool = shared_layer_output(avg_pool)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_hidden(max_pool)
    max_pool = shared_layer_output(max_pool)
    
    attention_feature = Add()([avg_pool, max_pool])
    attention_feature = Activation('sigmoid')(attention_feature)
    
    if K.image_data_format() == 'channels_first':
        attention_feature = Permute((3, 1, 2))(attention_feature)
    
    return Multiply()([input_feature, attention_feature])

def spatial_attention(input_feature):
    kernel_size = 7
    
    if K.image_data_format() == 'channels_first':
        inputs = Permute((2, 3, 1))(input_feature)
    else:
        inputs = input_feature
        
    avg_pool = Lambda(lambda x: K.mean(x, axis = 3, keepdims = True))(inputs)
    max_pool = Lambda(lambda x: K.max(x, axis = 3, keepdims = True))(inputs)
    concat = Concatenate(axis = 3)([avg_pool, max_pool])
    
    attention_feature = Conv2D(filters = 1, 
                               kernel_size = kernel_size, 
                               strides = 1, 
                               padding = 'same', 
                               activation = 'sigmoid', 
                               use_bias = False, 
                               kernel_initializer = 'he_normal')(concat)
    
    if K.image_data_format() == 'channels_first':
        attention_feature = Permute((3, 1, 2))(attention_feature)
        
    return Multiply()([input_feature, attention_feature])
    
def cbam(inputs, ratio):
    outputs = channel_attention(inputs, ratio)
    outputs = spatial_attention(outputs)
    
    return outputs
    
    
    