import csv, itertools, random, math,sys
import pandas as pd
import numpy as np
import gensim
import collections
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
import pickle
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = collections.defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

print("processing data")
df = pd.read_csv(r"./data2_compiled.csv",usecols=[2, 4, 6, 7, 8, 11, 21]).dropna(axis=0)

# Drop all rows where there are 0 likes 
df = df[df.num_reactions != 0]

# Allocate 20% of data to test set 
data, test = train_test_split(df, test_size=0.2, shuffle=False)

# Flag for which featurizer to be used, 0 for TF-IDF, 1 for Word2Vec, 2 for GloVe
featurizer = 2

# Set filename to load if present
fileName = None
if featurizer == 1:
    fileName = "word2vec.pickel"
elif featurizer == 0:
    fileName = "ngrams.pickel"
else:
    fileName = "unknown"

try:
    pipeline = pickle.load(open(fileName, "rb"))
    print("model present, loading...")
except (OSError, IOError, EOFError) as e:
    features = None

    # extract status and reactions as s, v where status = the status and values = the labels
    s_train = data.iloc[:,[6]].T.squeeze()
    v_train = data.iloc[:,[0,1,2,4,5]].T.squeeze()

    # normalize data by L1 norm
    v_train = v_train.div(v_train.sum(axis=0), axis=1)

    # Set NaN values to 0
    v_train.fillna(value=0, inplace=True)
    values = v_train.T
    print(values)


    pipeline = None

    # Set hidden layers
    layers=(300, 275, 250, 225, 200, 175, 150, 125, 100, 75, 50, 25)
    if featurizer ==0:
        layers=(100)
    # layers=(100)

    # Statuses 
    sentences = s_train.tolist()

    # Statuses as tokens
    tokens_input = None

    if featurizer == 0:
        print("1,2 gram")
        # Pipeline for TF-IDF into MLP
        pipeline = Pipeline([
            ("tfidf vectorizer", TfidfVectorizer(ngram_range=(1,2))),
            ("MLP", MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=layers, random_state=1, early_stopping=True))])
        tokens_input = sentences

    elif featurizer == 1:
        print("W2V")
        # Obtain corpus from all data, this is OK as it does not require label
        temp = df.iloc[:,[6]].T.squeeze().tolist()
        corpus = [temp[i].decode('utf-8').split() for i in range(len(temp))]
        # corpus = [temp[i].split() for i in range(len(temp))]

        # Obtain token for training data
        tokens_input = [sentences[i].decode('utf-8').split() for i in range(len(sentences))]
        # tokens_input = [sentences[i].split() for i in range(len(sentences))]

        # Construct Word2Vec neural net with a 200 neuron hidden layer 
        # model = gensim.models.FastText(corpus, size=200, sg=1, window=10, min_count=5, workers=2, hs=0, negative=10)
        model = gensim.models.Word2Vec(corpus, size=200, sg=1, window=10, min_count=5, workers=2, hs=0, negative=10)
        w2v = dict(zip(model.wv.index2word, model.wv.syn0))

        # Pipeline into MLP
        pipeline = Pipeline([
        ("tfidf vectorizer word2vec", TfidfEmbeddingVectorizer(w2v)),
        ("MLP", MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=layers, random_state=1, early_stopping=True))])
    elif featurizer == 2:
        print("GloVe")
        tokens_input = [sentences[i].decode('utf-8').split() for i in range(len(sentences))]
        glove_file = open('../random/glove.6B/glove.6B.300d.txt', 'rb')
        w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
                          for line in glove_file}

        pipeline = Pipeline([
        ("tfidf vectorizer glove", TfidfEmbeddingVectorizer(w2v)),
        ("MLP", MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=layers, random_state=1, early_stopping=True))])

    print("data processed")

    print("training...")

    pipeline.fit(tokens_input, values)

    print("trained")

    if featurizer == 0: # temp
        pickle.dump(pipeline, open(fileName, "wb"))

# TESTING
print("testing")

s_test = test.iloc[:,[6]].T.squeeze()
tokens = s_test.tolist()
s_test_words = [tokens[i].decode('utf-8').split() for i in range(len(tokens))]
if featurizer == 0:
    s_test_words = tokens

v_test = test.iloc[:,[0,1,2,4,5]].T.squeeze()
v_test = v_test.div(v_test.sum(axis=0), axis=1)
v_test.fillna(value=0, inplace=True)
values_test = v_test.T

predictions = pipeline.predict(s_test_words)
# predictions = pipeline.predict(s_test)
# print(s_test)
# print(values_test)

for elem in predictions:
    tot = 0
    for i in range(5):
        tot += max(0, elem[i])
    for i in range(5):
        elem[i] = max(0, elem[i]) / float(tot)

print(len(s_test_words))
print(len(predictions))

# Print out predictions
# for i in range(len(predictions)):
    # print(s_test_words[i])
    # print(predictions[i])


print("Mean squared error: %.5f" % mean_squared_error(values_test, predictions))
# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score(values_test, predictions))

# print predictions
# for value in values_test.itertuples():
#     print np.argmax(value[1:])

wrong = 0
right = 0
for prediction, value in zip(predictions, values_test.itertuples()):
    if np.argmax(prediction) == np.argmax(value[1:]):
        right += 1
    else:
        wrong += 1
print "Most popular reactions correct: ", float(right)/(wrong+right)
