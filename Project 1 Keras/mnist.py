# Khoi Tuan Nguyen (UID 114906472) - HW1-Ex3 - PHYS476
# MNIST data: identify digit based on image (keras, CNN, deep NN, classification)

import sys, os, time
save_stdout = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow
import keras.utils
from keras import backend as K

# compliance with the grading code
sys.stderr = save_stdout
program_name = sys.argv[0]
image_file = sys.argv[1] 
label_file = sys.argv[2]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# params
batch_size = 128
epoch_num = 3
kernel_size = 3
pool_size = 2
clayer_size = 32
input_shape = (28, 28, 1)
num_classes = 10
start = time.time()

# load data
images = pd.read_csv(image_file, header=None)
labels = pd.read_csv(label_file, header=None)

# split train test
x_train, x_test, label_train, label_test = train_test_split(images, labels, test_size=0.1)
x_train = x_train.values.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.values.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# one hot encoding labels
y_train = keras.utils.to_categorical(label_train, num_classes)
y_test = keras.utils.to_categorical(label_test, num_classes)

# define the model
model = Sequential()
model.add(Conv2D(clayer_size, kernel_size=(kernel_size, kernel_size), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(kernel_size, kernel_size), activation='relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
model.fit(x_train, y_train, 
	batch_size=batch_size,
	epochs= epoch_num, verbose=0)
score = model.evaluate(x_test, y_test, verbose=0)
K.clear_session()
print(score[1]*100)
