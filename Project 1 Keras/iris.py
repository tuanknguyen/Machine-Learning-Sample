# Khoi Tuan Nguyen (UID 114906472) - HW1-Ex2 - PHYS476
# Iris classification using deep NN (keras)
import sys, os
save_stdout = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow
from keras import backend as K
np.random.seed(7)

# compliance with the grading code
sys.stderr = save_stdout
program_name = sys.argv[0]
filename = sys.argv[1] 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# load data
df = pd.read_csv(filename)
df.columns = ['sepal_l', 'sepal_w', 'petal_l', 'petal_w', 'class']

# one hot encoding
ohe_labels = pd.get_dummies(df['class'])
del df['class']

# split train test data
x_train, x_test, y_train, y_test = train_test_split(df, ohe_labels, test_size=0.1)

# model
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=35, batch_size=10,verbose=0)

scores = model.evaluate(x_test, y_test, verbose=0)
K.clear_session()
print(scores[1]*100)
