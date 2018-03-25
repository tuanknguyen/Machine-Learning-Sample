from keras.models import Sequential
from keras.layers import Dense
import tensorflow
import pandas
import numpy as np
from sklearn.model_selection import train_test_split

data = pandas.read_csv('abalone.data', names=None)
#one hot encoding for sex columns
data['sex_m'] = np.where(data['sex']=='M', 1, 0)
data['sex_f'] = np.where(data['sex']=='F', 1, 0)
data['sex_i'] = np.where(data['sex']=='I', 1, 0)
del data['sex']
#print(data.head(n=5))

train, test = train_test_split(data, test_size=0.1)
y_train = train['rings'].values + 1.5
del train['rings']
x_train = train.values
y_test = test['rings'].values + 1.5
del test['rings']
x_test = test.values


model = Sequential()
model.add(Dense(12, input_dim=10, activation='relu'))
#model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=25, batch_size=10)
scores = model.evaluate(x_test, y_test)

print("\n%s: %.2f" %(model.metrics_names[1], scores[1]))
