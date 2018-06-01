import csv, itertools, random, math,sys
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import pickle

# try:
# 	clf = pickle.load(open("vector.pickel", "rb"))
# except (OSError, IOError) as e:

features = None
values = None
# 0 for TF-IDF, 1 for Word2Vec, 2 for GloVe
featurizer = 0
# TD-IDF as features using uni and bigram
transformer = TfidfVectorizer(ngram_range=(1,2))

# Read in only relevant columns. Drop all rows with a NaN value
print("processing data")
df = pd.read_csv(r"./data2_compiled.csv",usecols=[2, 4, 6, 7, 8, 11, 15]).dropna(axis=0)

# Drop all rows where there are 0 likes 
df = df[df.num_reactions != 0]

# Allocate 20% of data to test set 
data, test = train_test_split(df, test_size=0.3, shuffle=False)

# extract status and reactions as s, v where status = the status and values = the labels
s_train = data.iloc[:,[6]].T.squeeze()
v_train = data.iloc[:,[0,1,2,4,5]].T.squeeze()

# normalize data by L1 norm
v_train = v_train.div(v_train.sum(axis=0), axis=1)

# Set NaN values to 0
v_train.fillna(value=0, inplace=True)
values = v_train.T
print(values)

# Use TD-IDF as features 
sentences = s_train.tolist()
if featurizer == 0:
	features = transformer.fit_transform(sentences)
	print("data processed")
elif featurizer == 1:
	model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
	# Set softmax as the output function to mirror L1 norm
print("training...")
layers=(100)
# layers=(15)
clf = MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=layers, random_state=1, early_stopping=True)
clf.fit(features, values) 
# clf.n_layers_ = 3
# clf.out_activation_ = 'softmax'
print("trained")

# pickle.dump(clf, open("vector.pickel", "wb"))

# TESTING
print("testing")
s_test = test.iloc[:,[6]].T.squeeze()
tfidf_test = transformer.transform(s_test.tolist())
# tfidf_test = transformer.transform(["Security costs for Environmental Protection Agency Administrator Scott Pruitt tally up to nearly $3.5 million for the past year, according to figures the agency released Friday."])
v_test = test.iloc[:,[0,1,2,4,5]].T.squeeze()
v_test = v_test.div(v_test.sum(axis=0), axis=1)
v_test.fillna(value=0, inplace=True)
values_test = v_test.T
predictions = clf.predict(tfidf_test)
print(s_test)
print(values_test)
for elem in predictions:
	tot = 0
	for i in range(5):
		tot += max(0, elem[i])
	for i in range(5):
		elem[i] = max(0, elem[i]) / float(tot)

print(predictions)
print("Mean squared error: %.2f" % mean_squared_error(values_test, predictions))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(values_test, predictions))
