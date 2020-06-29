# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:39:43 2020

@author: uni tech
"""

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import random
import cv2
import os
from sklearn.metrics import r2_score


model = load_model('fingerprint_recog.h5')


path = "C:/Users/uni tech/Desktop/spyderr/datasets/SOCOFing/Altered/Altered-Easy"
n=0
testing_data =  []


for img in os.listdir(path):
    if n<500:
        list_of_strings=[]
        img_path = os.path.join(path,img)
        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (80, 80))
        
        new_name=os.path.split(img_path)[-1]
        new_name2 = new_name[:-4]
   
        for x in new_name2:
            list_of_strings.append(x)
     
        
        if "M" in list_of_strings:
            testing_data.append([new_array, 0])
            
        elif "F" in list_of_strings:
            testing_data.append([new_array, 1])
            # X_test.append(new_array)
        n+=1
        
        
    else:
        break

random.shuffle(testing_data)


X_test=[]
y_test=[]

for features, labels in testing_data:
    X_test.append(features)
    y_test.append(labels)
    

X_test = np.array(X_test).reshape(-1, 80, 80, 1)
X_test = X_test / 255




result = model.predict(X_test)
classes = np.round(result)

# classes = np.argmax(result, axis = 1)
print(classes)

accuracy = r2_score(y_test, result)
print(accuracy)



    
