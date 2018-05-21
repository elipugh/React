import csv, itertools, random, math,sys
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

# TODO: Validation set

# TD-IDF as features using uni and bigram
transformer = TfidfVectorizer(ngram_range=(1,2))

# Read in only relevant columns. Drop all rows with a NaN value
print("processing data")
df = pd.read_csv(r"/Users/derekhuang/OneDrive/College/2017-2018/CS221/React/data/data1_compiled.csv",usecols=[1, 9, 10, 11, 12, 13, 14]).dropna(axis=0)

# Drop all rows where there are 0 likes 
df = df[df.num_likes != 0]

# Allocate 20% of data to test set 
data, test = train_test_split(df, test_size=0.2)

# extract status and reactions as s, v where status = the status and values = the labels
s_train = data.iloc[:,[0]].T.squeeze()
v_train = data.iloc[:,[2,3,4,5,6]].T.squeeze()

# normalize data by L1 norm
v_train = v_train.div(v_train.sum(axis=0), axis=1)

# Set NaN values to 0
v_train.fillna(value=0, inplace=True)
values = v_train.T

# Use TD-IDF as features 
sentences = s_train.tolist()
features = transformer.fit_transform(sentences)
print("data processed")

# Set softmax as the output function to mirror L1 norm
print("training...")
clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20), random_state=1)
clf.out_activation_ = 'softmax'
clf.fit(features, values) 
print("trained")

# TESTING
print("testing")
s_test = test.iloc[:,[0]].T.squeeze()
tfidf_test = transformer.transform(s_test.tolist())
v_test = test.iloc[:,[2,3,4,5,6]].T.squeeze()
v_test = v_test.div(v_test.sum(axis=0), axis=1)
v_test.fillna(value=0, inplace=True)
values_test = v_test.T
predictions = clf.predict(tfidf_test)
print("Mean squared error: %.2f" % mean_squared_error(values_test, predictions))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(values_test, predictions))
