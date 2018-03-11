import tensorflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv('pima-indians-diabetes.data')
# delete rows with missing data, remove columns with the most missing data
# remove columns with more than 10% missing data
nullCols = df.isnull().sum()
print (nullCols[nullCols > df.shape[0]*0.1])
df = df.dropna(how='any',axis=0) 

train, test = train_test_split(df, test_size=0.1)
y_train = train.iloc[:,-1]
x_train = train.iloc[:, :-1]

y_test = test.iloc[:,-1]
x_test = test.iloc[:, :-1]

clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=1, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.0001, verbose=False)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print('Accurary', accuracy_score (y_test, y_pred))
