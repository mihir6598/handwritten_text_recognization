# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:19:37 2018

@author: mihir
"""

import os
from PIL import Image
from array import *
from random import shuffle
import numpy as np
from skimage.transform import resize
import keras
import cv2   
from keras.datasets import mnist
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense,Dropout, Flatten
from keras.utils import np_utils
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from skimage import io
from skimage.transform import resize
from PIL import Image
import PIL.ImageOps 
from sklearn import cross_validation


data_label = []
FileList = []
data_image = []
flag = 0
count = 0
area = 0
x1 = 0
y1 = 0
w1 = 0
h1 = 0
max = 0


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='Adam')
    return model

Names = [['./training-images','train']]

print(Names)
for name in Names:
    for dirname in os.listdir(name[0]):
        path = os.path.join(name[0],dirname)
        for filename in os.listdir(path):
            if filename.endswith(".png"):
                FileList.append(os.path.join(name[0],dirname,filename))
                
                
    shuffle(FileList)
    img = cv2.imread("train_42_00000.png", cv2.IMREAD_UNCHANGED)
    img3 = Image.open("train_42_00000.png")
    ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),127, 255, cv2.THRESH_BINARY)
    image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)
    area = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=1, lineType=8, shift=0)
        area = w*h
        if(area>max) : 
            x1 = x
            y1 = y
            w1 = w
            h1 = h
            
        
        #image = resize(image, (64, 64), Image.NEAREST)
        #img2.save("img2.png")
    img2 = img3.crop((x1, y1, x1+w1, y1+h1))
    img2.save("temp.png")
    image2 = io.imread("temp.png", as_grey=True)
    image = resize(image2, (28, 28), Image.NEAREST)
    array = np.array(image)
    array = 1-array
    data_image = np.reshape(array, (1,np.product(array.shape)))
    data_label.append("1")
    print(len(FileList))
    for filename in FileList:
        count = count+1
        label = (filename.split('\\')[1])
       
        label = ord(label) - 97
            
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        img3 = Image.open(filename)
        ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),127, 255, cv2.THRESH_BINARY)
        image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #print(contours)
        area = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=1, lineType=8, shift=0)
            area = w*h
            if(area>max) : 
                x1 = x
                y1 = y
                w1 = w
                h1 = h
                
            
            #image = resize(image, (64, 64), Image.NEAREST)
            #img2.save("img2.png")
        img2 = img3.crop((x1, y1, x1+w1, y1+h1))

        img2.save("temp.png")
        image2 = io.imread("temp.png", as_grey=True)
        image = resize(image2, (28, 28), Image.NEAREST)
        array = np.array(image)
        array = 1-array
        b = np.reshape(array, (1,np.product(array.shape)))
        
        data_image=np.concatenate((data_image, b), axis=0)
        data_label.append(label)

num_classes = len(np.unique(data_label))
data_label = keras.utils.to_categorical(data_label)
data_image = data_image.reshape(len(data_image),28,28,1)

X_train, X_test, Y_train, Y_test =   cross_validation.train_test_split(data_image,data_label, test_size=0.30, random_state=111)
input_shape = (28,28,1)
model = baseline_model()
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=200, verbose=2)
scores = model.evaluate(data_image, data_label, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

model.save('ayushi_mihir.h5')



                    
  
    
   
   
    

    
    
    
    
    
    
            
        
