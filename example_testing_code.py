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

model = load_model('fingerprint_recog.h5')
X_test = []

path = "C:/Users/uni tech/Desktop/spyderr/datasets/SOCOFing/Altered/Altered-Easy"
n=0

for img in os.listdir(path):
    if n<500:
        img_path = os.path.join(path,img)
        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (80, 80))
        X_test.append(new_array)
        n+=1
    else:
        break



X_test = np.array(X_test).reshape(-1, 80, 80, 1)


X_test = X_test/255

random.shuffle(X_test)


result = model.predict(X_test)


classes = np.argmax(result, axis = 1)
print(classes)

    