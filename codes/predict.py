# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 09:45:58 2018

@author: mihir
"""

from keras.models import load_model
from skimage import io
from skimage.transform import resize
from PIL import Image
import numpy as np
import cv2
import tensorflowjs as tfjs


import random
import string
from claptcha import Claptcha

def randomString():
    rndLetters = (random.choice(string.ascii_uppercase) for _ in range(3))
    return "".join(rndLetters)

# Initialize Claptcha object with random text, FreeMono as font, of size
# 100x30px, using bicubic resampling filter and adding a bit of white noise
r_string = randomString()
c = Claptcha(r_string, "FreeMono.ttf", (1000,500),
             resample=Image.BICUBIC, noise=0.3)

text, _ = c.write('captcha1.png')
img = Image.open('captcha1.png')
img.show() 
print('Generated captcha : ',text)  # 'PZTBXB', string printed into captcha1.png


x1 = 0
y1 = 0
w1 = 0
z1 = 0
ready = 'a'
while(ready != 'r'):
    ready = input("type r when u are ready  : ")

    
answer = ''
data_image2 = []
img = cv2.imread('a.png', cv2.IMREAD_UNCHANGED)
img3 = Image.open("a.png")
array = np.array(img)
array1 = np.array(img3)
print(type(img))
print(type(img3))
ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),127, 255, cv2.THRESH_BINARY)
image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
main_ans = []
model = load_model('ayushi_mihir.h5')
tfjs.converters.save_keras_model(model,'mihir')
#print(contours)
x2 = []
y2 = []
w2 = []
h2 = []
count = 0
for c in contours:
    
    x, y, w, h = cv2.boundingRect(c)
    x2.append(x)
    y2.append(y)
    w2.append(w)
    h2.append(h)    
    
   
x2, y2,w2,h2 = zip(*sorted(zip(x2,y2,w2,h2)))
for i in range(len(x2)):
    x = x2[i]
    y = y2[i]
    w = w2[i]
    h = h2[i]
    
    if((x1<x and y1<y and z1>(x+w) and w1>(y+h)) or x == 0) : 
        #print("cant")
        x = 1
    else :
        x1 = x
        y1 = y
        z1 = (x+w)
        w1 = (y+h)
        
    
        img2 = img3.crop((x1, y1, z1, w1))
        img2.save("temp.png")
        
        
        image = io.imread("temp.png", as_grey=True)
           
        image = resize(image, (28, 28), Image.NEAREST)
        array = np.array(image)
        
        array = 1-array
        b = np.reshape(array, (1,np.product(array.shape)))
        b = b.reshape(1,28,28,1)
        
        ans = model.predict_classes(b)
        #model.summary()
        ans = chr(ans+65)
        answer = answer + ans
        print(ans)
    
          
        
       
print(answer)
print("Finished")

if(r_string == answer):
    print('You are wright')
else:
    print('You are wrong')
        
