import tensorflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

df = pd.read_excel('Data_Cortex_Nuclear.xls')
# delete rows with missing data, remove columns with the most missing data
# remove columns with more than 10% missing data
nullCols = df.isnull().sum()
print (nullCols[nullCols > df.shape[0]*0.1])
df.drop(['MouseID','class', 'BAD_N', 'BCL2_N', 'H3AcK18_N', 'EGR1_N', 'H3MeK4_N'], axis=1, inplace=True)
df = df.dropna(how='any',axis=0) 

df.drop(['Genotype', 'Treatment', 'Behavior'], axis=1, inplace=True)
print(df.shape)

train, test = train_test_split(df, test_size=0.1)
kmeans = KMeans(n_clusters=8, random_state=0).fit(df)
kmeans.labels_
print(kmeans.cluster_centers_)
# unfinished bc cannot compare to find accuracy
