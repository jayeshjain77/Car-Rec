#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 22:27:14 2018

@author: Jayesh
"""

import numpy as np
from keras import layers,models
import os
from skimage import io, transform

imageSize = 100

def TransformData():        #load and transform data
    images = os.listdir("datafile")
    training_data = []
    training_labels = []
    for image in images:
        if image[-3:] == 'png':
            transformed_image = transform.resize(io.imread("datafile" +'/' + image), (imageSize, imageSize, io.imread("datafile" +'/' + image).shape[2]))
            training_data.append(transformed_image)
            label_file = image[:-4] + '.txt'
            with open("datafile" + '/' + label_file) as f:
                content = f.readlines()
                label = int(float(content[0]))
                l = [0, 0]
                l[label] = 1
                training_labels.append(l)
        elif (image[-3:] == 'jpg'):
            transformed_image = transform.resize(io.imread("datafile" +'/' + image), (imageSize, imageSize, io.imread("datafile" +'/' + image).shape[2]))
            training_data.append(transformed_image)
            label_file = image[:-4] + '.txt'
            with open("datafile" + '/' + label_file) as f:
                content = f.readlines()
                label = int(float(content[0]))
                l = [0, 0]
                l[label] = 1
                training_labels.append(l)
        elif (image[-4:] == 'jpeg'):
            transformed_image = transform.resize(io.imread("datafile" +'/' + image), (imageSize, imageSize, io.imread("datafile" +'/' + image).shape[2]))
            training_data.append(transformed_image)
            label_file = image[:-5] + '.txt'
            with open("datafile" + '/' + label_file) as f:
                content = f.readlines()
                label = int(float(content[0]))
                l = [0, 0]
                l[label] = 1
                training_labels.append(l)
    return np.array(training_data), np.array(training_labels)

def NeuralNet():
    model = models.Sequential()
    model.add(layers.Conv2D(8, 3, 3, border_mode='same',
                            input_shape=(imageSize, imageSize, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(16, 3, 3, border_mode='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    
    model.add(layers.Conv2D(32, 3, 3, border_mode='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model



def split_train_test(data, labels, f):
    test_size = int(len(data) * f)
    return data[:test_size],labels[:test_size],data[test_size:],labels[test_size:]


data, labels = TransformData()
training_data, training_labels, test_data, test_labels = split_train_test(
    data, labels, 0.9)
idx = np.random.permutation(training_data.shape[0])
model = NeuralNet()
model.fit(training_data[idx], training_labels[idx], nb_epoch=5)

predictions = np.argmax(model.predict(test_data), axis=1)
test_labels = np.argmax(test_labels, axis=1)

"""
for calculating accuracy
"""

a=[]
for i in range(len(test_labels-1)):
    if test_labels[i] == predictions[i]:
        a.append(1)

accuracy = len(a)/len(test_labels)
        

print ("Training data size: ", len(training_data))
print ("Test data size: ", len(test_data))
print ("accuracy: ", accuracy)