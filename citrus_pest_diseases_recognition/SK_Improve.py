# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Oct  7 15:48:20 2020

@author: Admin
"""
from keras.layers import Conv2D
from keras.layers import Activation

from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras import backend as K

from squeeze_excitation_layer import squeeze_excite_block

if K.image_data_format() == 'channels_first':
    bn_axis = 1
else:
    bn_axis = -1
    
def _select_kernel(inputs, kernels, filters, ratio, block):
    '''
    kernels = [3, 5]
    ratio: to keep the same computation with SK_Block, ratio should be doubled
    '''
    base_name = 'sk_block_' + str(block) + '_'
    k_1, k_2 = kernels
    
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
    
    x_c = Concatenate(axis = bn_axis)([x_1, x_2])
#    The shape of x_c: b, h, w, filters * 2
    
    x = squeeze_excite_block(x_c, ratio, block)
#    The shape of x: b, h, w, filters * 2
    x = Conv2D(filters = filters, 
               kernel_size = 1, 
               strides = (1, 1), 
               name = base_name + 'conv_3')(x)
#    The shape of se: b, h, w, filters
    x = BatchNormalization(axis = bn_axis, name = base_name + 'bn_3')(x)
    x = Activation('relu')(x)
    
    return x

#The 1 x 1 convolution is used for feature mixing and dimensionality reduction
    
