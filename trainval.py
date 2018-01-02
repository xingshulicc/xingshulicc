# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Thu Dec 28 10:07:39 2017

@author: xingshuli
"""
'''
generate trainval.txt 

'''
import os

def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                file_name = file[0:-4]
                L.append(file_name)
                
    return L

label_folder = './Hyaloph_detection'
trainval_file = './trainval.txt'

txt_name = file_name(label_folder)

with open(trainval_file, 'w') as f:
    for i in txt_name:
        f.write('{}\n'.format(i))
f.close()



                