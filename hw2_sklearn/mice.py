# Khoi Tuan Nguyen (UID 114906472) - HW2-Ex1 - PHYS476
# Mice protein classification (KNN, sklearn)
# http://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sys

# compliance with the grading code
program_name = sys.argv[0]
filename = sys.argv[1]

df = pd.read_excel(filename)
# delete rows with missing data, remove columns with the most missing data
# remove columns with more than 10% missing data
nullCols = df.isnull().sum()
df.drop(['MouseID', 'BAD_N', 'BCL2_N', 'H3AcK18_N', 'EGR1_N', 'H3MeK4_N'], axis=1, inplace=True)
df = df.dropna(how='any',axis=0) 
df.drop(['Genotype', 'Treatment', 'Behavior'], axis=1, inplace=True)
df.dropna(axis=0, how='any', inplace=True)
labels = df['class']
del df['class']

acc = []
for i in range (0,100):
	# split train and test data
	x_train, x_test, y_train, y_test = train_test_split(df, labels,  test_size=0.1)

	# create classifier model
	clf = KNeighborsClassifier(n_neighbors=4, weights='distance')

	# train then predict
	clf.fit(x_train, y_train)
	pred = clf.predict(x_test)
	acc.append(accuracy_score(y_test, pred))
print(sum(acc)/len(acc))	
