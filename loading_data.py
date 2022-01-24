import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

DATADIR = "C:/users/brade/desktop/classes/Comp Sci/TensorFlow/cats_and_dogs_dataset/PetImages"
CATEGORIES = ["Dog", "Cat"]

training_data = []
IMG_SIZE = 50

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in os.listdir(path):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

create_training_data()

random.shuffle(training_data)
x, y = [], []

for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x)
x = x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

np.save("x.npy", x)
np.save("y.npy", y)
