# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue Aug  8 13:09:46 2017

@author: xingshuli
"""
import os

import keras
#from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import backend as K

from Baseline_Network import BaseNet
#from ResNet_8 import ResNet
#from Rnet_8 import RNet_8

from learning_rate import choose

#set hyper-parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # '1' or '0' GPU

batch_size = 32
num_classes = 100
epochs = 2


img_height, img_width = 32, 32

if K.image_dim_ordering() == 'th':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#the data shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
nb_train_samples = x_train.shape[0]
nb_validation_samples = x_test.shape[0]

#convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#load model
model = BaseNet(input_shape = input_shape, classes = num_classes)

optimizer = SGD(lr = 0.001, momentum = 0.9, nesterov = True) 
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
model.summary()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#prepare data augmentation configuration
train_datagen = ImageDataGenerator(width_shift_range = 0.1, 
                                   height_shift_range = 0.1, 
                                   rotation_range = 15, 
                                   horizontal_flip = True)
train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train, y_train, batch_size = batch_size)

#set learning rate schedule
lr_monitorable = True
lr_reduce = choose(lr_monitorable = lr_monitorable)

#set callbacks for model fit
callbacks = [lr_reduce]

#model fit
hist = model.fit_generator(train_generator, 
                           steps_per_epoch = nb_train_samples //batch_size, 
                           epochs = epochs, 
                           validation_data = (x_test, y_test), 
                           callbacks = callbacks)
                           
#print acc and stored into acc.txt
f = open('/home/xingshuli/Desktop/acc.txt','w')
f.write(str(hist.history['acc']))
f.close()
#print val_acc and stored into val_acc.txt
f = open('/home/xingshuli/Desktop/val_acc.txt','w')
f.write(str(hist.history['val_acc']))
f.close()
#print val_loss and stored into val_loss.txt   
f = open('/home/xingshuli/Desktop/val_loss.txt', 'w')
f.write(str(hist.history['val_loss']))
f.close()


Er_patience = 10
accur = []
with open('/home/xingshuli/Desktop/val_acc.txt','r') as f1:
    data1 = f1.readlines()
    for line in data1:
        odom = line.strip('[]\n').split(',')
        num_float = list(map(float, odom))
        accur.append(num_float)
f1.close()

y = sum(accur, [])
ave = sum(y[-Er_patience:]) / len(y[-Er_patience:])
print('Validation Accuracy = %.4f' % (ave))

#save model
save_dir = os.path.join(os.getcwd(), 'Cifar_100_compare')
model_name = 'keras_Basenet_model.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, model_name)
model.save(save_path)
print('the model has been saved at %s' %save_path)




