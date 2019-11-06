# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Thu Sep 20 09:51:03 2018

@author: xingshuli
"""
import math

from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau


lr_base = 0.001
epochs = 500

def lr_scheduler(epoch, mode = 'step_decay'):
    if mode is 'progressive':
        if epoch > 0.9 * epochs:
            lr = lr_base * 1e-2
        elif epoch > 0.75 * epochs:
            lr = lr_base * 0.1
        elif epoch > 0.5 * epochs:
            lr = lr_base * 0.5
        else:
            lr = lr_base
        
    if mode is 'step_decay':
        drop = 0.1
        epochs_drop = epochs / 10
        lr = lr_base * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        
    if mode is 'Adam':
        lr = lr_base
    
    return lr
    
def choose(lr_monitorable = True):
    if lr_monitorable:
        lr_reduce = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.8, patience = 5, 
                                  mode = 'auto', min_lr = lr_base * 1e-7)
    else:
        lr_reduce = LearningRateScheduler(lr_scheduler)
        
    return lr_reduce





