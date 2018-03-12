# Khoi Tuan Nguyen (UID 114906472) - HW2-Ex2 - PHYS476
# Predict cases of heart diseas in process.hungrarian.data using Naive Bayes, decision tree and random forrest
# http://archive.ics.uci.edu/ml/datasets/Heart+Disease

import numpy as np
import pandas as pd
import sys, time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# compliance with the grading code
program_name = sys.argv[0]
filename = sys.argv[1]

# load data
df = pd.read_csv(filename, header=None)
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
		'oldpeak', 'slope', 'ca', 'thal', 'num']

# remove columns that has more than 10% missing data, then any row with missing data
df = df.apply(pd.to_numeric, errors='coerce')
nullCols = df.isnull().sum()
col_to_remove =  nullCols[nullCols > df.shape[0]*0.1].index
df.drop(col_to_remove, axis=1, inplace= True)
df.dropna(axis=0, how='any', inplace=True)
#print(col_to_remove)

labels = df['num']
del df['num']

# one hot encoding for the categorical columns
ohe_cols = ['sex', 'cp', 'exang', 'restecg']
df = pd.get_dummies(df, columns=ohe_cols, prefix=ohe_cols)
#print(df.head(n=10), df.shape)

gnb_acc = []
dt_acc = []
rf_acc = []
for i in range(0, 100):
	# split train test
	x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size= 0.1, random_state=i)
	
	# Naive Bayes model
	gnb = GaussianNB()
	gnb.fit(x_train, y_train)
	gnb_acc.append(gnb.score(x_test,y_test))

	# Decision Tree
	dt = tree.DecisionTreeClassifier(criterion='gini', max_depth=5, 
					min_samples_leaf=0.1, max_features=5)
	dt.fit(x_train, y_train)
	dt_acc.append(dt.score(x_test, y_test))
	
	# Random Forest
	rf = RandomForestClassifier(min_samples_leaf=0.05, max_depth=10, n_estimators=15)
	rf.fit(x_train, y_train)
	rf_acc.append(rf.score(x_test, y_test))

	#time.sleep(0.01)
print(sum(gnb_acc)/len(gnb_acc))
print(sum(dt_acc)/len(dt_acc))
print(sum(rf_acc)/len(rf_acc))
