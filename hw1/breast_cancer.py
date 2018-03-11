# Khoi Tuan Nguyen (UID 114906472) - HW1-Ex1 - PHYS476
# Predict malignancy using deep NN (keras, binary classification)

import sys, os
save_stdout = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import Sequential
from keras.layers import Dense
import keras.utils
import tensorflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend as K

# compliance with the grading code
sys.stderr = save_stdout
program_name = sys.argv[0]
filename = sys.argv[1] 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# load data
df = pd.read_csv(filename)
df.columns = ['id', 'clump', 'cell_size', 'cell_shape', 'adhesion', 'epithelial',
		'bare_nuclei', 'chromatin', 'nucleoli', 'mitoses', 'class']
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(axis=0, how='any', inplace=True)
# one hot encoding the class
labels = df['class'] / 2 - 1
ohe_labels = keras.utils.to_categorical(labels, num_classes = 2)
df.drop(['id', 'class'], axis=1, inplace= True)

# train test split
x_train, x_test, y_train, y_test = train_test_split(df, ohe_labels, test_size=0.1)

# model
model = Sequential()
model.add(Dense(12, input_dim=9, activation='relu'))
#model.add(Dense(18, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, batch_size=32,verbose=0)
scores = model.evaluate(x_test, y_test,verbose=0)

K.clear_session()
print (scores[1]*100)
