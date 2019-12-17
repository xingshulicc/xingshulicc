# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Thu Dec 12 14:25:53 2019
@author: Shuli Xing
"""
from keras import backend as K
from keras.layers import Conv2D
from keras import regularizers
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D

from keras.layers import Input

from keras.models import Model

from keras.utils import plot_model

weight_decay = 0.005
axis = 1 if K.image_data_format() == 'channels_first' else -1
expansion = 2

def convolution(x, filters, kernel_size, strides=(1, 1), bias=True):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
               use_bias=bias, kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(axis=axis)(x)
    x = Activation('relu')(x)
    return x


def pooling(x, pool_size=(2, 2), strides=(2, 2)):
    x = MaxPooling2D(pool_size=pool_size, strides=strides, padding='same')(x)
    return x


def stem(x):
    x = convolution(x, filters=32, kernel_size=(7, 7), strides=(2, 2))
    x = pooling(x, pool_size=(3, 3))
    return x


def bridge(x, filters):
    x = convolution(x, filters, kernel_size=(1, 1), bias=False)
    x = convolution(x, filters, kernel_size=(1, 1), bias=False)
    return x


def pyramid(x, filters):
    z1 = x
    z2 = convolution(z1, filters=filters // expansion, kernel_size=(3, 3))
    z3 = convolution(z2, filters=filters // expansion, kernel_size=(3, 3))   
    
    out = Concatenate()([z1, z2, z3])    
    out = bridge(out, filters)    
    return out


def network(input_shape, classes):
    inputs = Input(shape=input_shape)

    x1 = stem(inputs)
#    The shape of x1: 56 x 56 x 32
    
    x2 = pyramid(x1, filters=64)
#    The shape of x2: 56 x 56 x 64
    
    pool1 = pooling(x2)
#    The shape of pool1: 28 x 28 x 64

    x3 = Concatenate(axis=axis)([x1, x2])
#    The shape of x3: 56 x 56 x 96
    
    x4 = pyramid(x3, filters=128)
#    The shape of x4: 56 x 56 x 128
    
    pool2 = pooling(x4)
#    The shape of pool2: 28 x 28 x 128

    x5 = Concatenate(axis=axis)([pool1, pool2])
#    The shape of x5: 28 x 28 x 192
    
    x6 = pyramid(x5, filters=256)
#    The shape of x6: 28 x 28 x 256
    
    pool3 = pooling(x6)
#    The shape of pool3: 14 x 14 x 256

    x7 = Concatenate(axis=axis)([pool2, x6])
#    The shape of x7: 28 x 28 x 384
    
    x8 = pyramid(x7, filters=512)
#    The shape of x8: 28 x 28 x 512
    
    pool4 = pooling(x8)
#    The shape of pool4: 14 x 14 x 512

    x9 = Concatenate(axis=axis)([pool3, pool4])
#    The shape of x9: 14 x 14 x 768
    
    x10 = pyramid(x9, filters=512)
#    The shape of x10: 14 x 14 x 512
    
    pool5 = pooling(x10)
#    The shape of pool5: 7 x 7 x 512 

    bridge1 = bridge(x7, filters=256)
#    The shape of bridge1: 28 x 28 x 256
    
    pool_bridge1 = pooling(bridge1)
#    The shape of pool_bridge1: 14 x 14 x 256

    bridge2 = Concatenate(axis=axis)([pool_bridge1, pool4])
#    The shape of bridge2: 14 x 14 x 768
    
    bridge3 = bridge(bridge2, filters=512)
#    The shape of bridge3: 14 x 14 x 512
    
    pool_bridge2 = pooling(bridge3)
#    The shape of pool_bridge2: 7 x 7 x 512 

    bridge4 = Concatenate(axis=axis)([pool_bridge2, pool5])
#    The shape of bridge4: 7 x 7 x 1024 
    
    bridge5 = bridge(bridge4, filters=512)
#    The shape of bridge5: 7 x 7 x 512

    out = GlobalAveragePooling2D()(bridge5)
    out = Dense(512, activation='relu')(out)
    out = Dropout(0.5)(out)
    out = Dense(classes, activation='softmax')(out)
    
    model = Model(inputs, outputs=out, name='pyramid_bridge_net')
    plot_model(model, to_file = 'pyramid_bridge_model.png',show_shapes = True, show_layer_names = True)
    return model