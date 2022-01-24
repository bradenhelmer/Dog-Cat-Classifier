import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
from keras.callbacks import TensorBoard
import time

NAME = "Cats-vs-Dogs-CNN-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

x = np.load("x.npy")
y = np.load("y.npy")

x = x/255.0

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape = x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

model.fit(x, y, batch_size=32, validation_split=0.1, epochs=20, callbacks=[tensorboard])
model.save("CNN.model")

