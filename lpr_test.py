from lpr_model import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from numpy import load, save
import math, os
import cv2 

# load images and labels for the dataset to test on
images = load('data/AOLP_LE_plate_images.npy', allow_pickle=True)
labels = load('data/AOLP_LE_plate_labels_in_number.npy', allow_pickle=True)


len_images = len(images)
len_labels = len(labels)

print("len_images = ", len_images)
print("len_labels = ", len_labels)


#construct the lpr model
model = Model(
    inputs={'image': inputs},
    outputs={
        'char0': y0,
        'char1': y1,
        'char2': y2,
        'char3': y3,
        'char4': y4,
        'char5': y5
    },
)

#load model weights from model trained on testing data
model.load_weights("pretrained_model/train_AC_RP_model_weights_34.h5")

num_char_correct = 0
num_char_total = 0

#make predictions on test data
for i in range(len_images): 
    print(i)
    image = images[i]
    img2 = cv2.resize(image, (160,64))
    img_arr = np.array(img2)

    image = img_arr.reshape((1, 64, 160, 3))

    image = image / 255.0 

    string_pred1 = model(image ) #send current image through model

    #get predictions for each character in image
    y = []
    y.append(int(tf.math.argmax(string_pred1['char0'], 1)))
    y.append(int(tf.math.argmax(string_pred1['char1'], 1)))
    y.append(int(tf.math.argmax(string_pred1['char2'], 1)))
    y.append(int(tf.math.argmax(string_pred1['char3'], 1)))
    y.append(int(tf.math.argmax(string_pred1['char4'], 1)))
    y.append(int(tf.math.argmax(string_pred1['char5'], 1)))

    #count total number of characters and number of correct predictions
    str1 = []
    for j in range(6):
        num_char_total += 1
        if y[j] == labels[i][j]:
            num_char_correct += 1
 
accuracy = 100.* float( num_char_correct) / num_char_total
print("The accuracy of the pretrained model on AOLP LE dataset is = ", accuracy,"%")
