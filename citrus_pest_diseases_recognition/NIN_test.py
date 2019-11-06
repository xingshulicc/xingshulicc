# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
"""
Created on Fri May  4 13:43:46 2018

@author: xingshuli
"""
import os
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

#from NIN_16 import NIN16
#from model_vgg16 import VGG16
#from Rnet import New_net
from wide_Resnet import wide_resnet
from Bridge_VGG19 import Bridge_VGG
from learning_rate import choose

#pre-parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # '1' or '0' GPU

img_height, img_width = 224, 224

if K.image_dim_ordering() == 'th':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

batch_size = 16
epochs = 500

train_data_dir = os.path.join(os.getcwd(), 'image_Data/train')
validation_data_dir = os.path.join(os.getcwd(), 'image_Data/validation')

num_classes = 24
nb_train_samples = 10402
nb_validation_samples = 2159

#model = New_net(input_shape = input_shape, classes = num_classes)
#model = VGG16(input_shape = input_shape, classes = num_classes)
model = Bridge_VGG(input_shape = input_shape, classes = num_classes)

optimizer = SGD(lr = 0.001, momentum = 0.9, nesterov = True) 
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
model.summary()


train_datagen = ImageDataGenerator(rescale = 1. / 255, 
                                   rotation_range = 15, 
                                   width_shift_range = 0.2, 
                                   height_shift_range = 0.2, 
                                   horizontal_flip = True, 
                                   zoom_range = 0.2, 
                                   shear_range = 0.2)

test_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


#set learning rate schedule
lr_monitorable = True
lr_reduce = choose(lr_monitorable = lr_monitorable)

#set callbacks for model fit
callbacks = [lr_reduce]

#model fit
hist = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples //batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples //batch_size, 
    callbacks=callbacks)

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

#the reasonable accuracy of model should be calculated based on
#the value of patience in EarlyStopping: accur = accur[-patience + 1:]/patience
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
save_dir = os.path.join(os.getcwd(), 'Wide_ResNet_Model')
model_name = 'keras_trained_model.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, model_name)
model.save(save_path)
print('the model has been saved at %s' %save_path)


