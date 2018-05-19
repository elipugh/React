#!/usr/bin/python

import random
import collections
import math
import sys
import copy
import StringIO
import numpy as np
from scipy import stats
from util import *

#############################################################
#                                                           #
#                                                           #
#        this baseline program learns weights               #
#        for sentiment analysis of facebook posts           #
#        using unigram features and regression              #
#                                                           #
#                                                           #
#############################################################


############################################################
# unigram feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    arr = x.split()
    d={}
    for word in arr:
        if word in d:
            d[word]+= 1
        else:
            d[word]=1
    return d

############################################################
# stochastic gradient descent kinda

def gradient(weights, features, y):
    result={}
    x = predictReacts(weights, features) - y
    for k in features:
        result[k] = x
    return result

def geterror(examples, weights):
    total = 0.0
    for x,y in examples:
        yhat = predictReacts(x, weights)
        total += np.linalg.norm(y-yhat)
    return total / len(examples)

############################################################
# Run regression with pseudo SGD

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta, verbose):

    weights = {}  # feature => weight
    eta = 1
    trainExamples2 = []
    for x,y in trainExamples:
        trainExamples2.append((featureExtractor(x),y))
        norm = np.linalg.norm(y)
        if norm != 0:
            y /= norm
    trainExamples = trainExamples2

    for i in range(numIters):
        eta = 1 / math.sqrt(1.+i)
        if not verbose and i % 10 == 0 and i != 0:
            print i/20.0
        for x,y in trainExamples:
            grad = gradient(weights, x, y)
            for k in grad:
                if k in weights:
                    weights[k] -= grad[k]*eta
                else:
                    weights[k] = grad[k]*eta
        if verbose==1 and i % 20 == 0:
            print "epoch ", i, " avg error"
            print geterror(trainExamples,weights)
    return weights

############################################################
# character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        x = x.replace(" ", "")
        result = {}
        for i in range(0, len(x)-n+1):
            feature = x[i:i+n]
            if feature in result:
                result[feature] += 1
            else:
                result[feature] = 1
        return result
        # END_YOUR_CODE
    return extract

############################################################
# predict reactions for a given post

def predictReacts(weights, features):
    result = np.zeros(5)
    for k in features:
        if k in weights:
            result += features[k] * weights[k]
    for i in range(len(result)):
        if result[i] < 0:
            result[i] = 0
    norm = np.linalg.norm(result)
    if norm != 0:
        result /= norm
    return result

############################################################
#                        SCRIPT                            #
############################################################

filename = raw_input("CSV dataset: ")
examples = readExamples(filename)
#np.random.shuffle(examples)
trainsz = 8*len(examples)/10
trainExamples = examples[:trainsz]
testExamples = examples[trainsz:]
featureExtractor = extractWordFeatures
weights = learnPredictor(trainExamples, testExamples, featureExtractor, numIters=100, eta=1, verbose=1)
print " "
print "Today the president signed a wonderful bill", predictReacts(weights, extractWordFeatures("today the president signed a wonderful bill"))
print "New cancer cure discovered is incredible", predictReacts(weights, extractWordFeatures("new cancer cure discovered is incredible"))
print "this program sucks. it's the worst", predictReacts(weights, extractWordFeatures("this program sucks its the worst"))
print "I love my mom.", predictReacts(weights, extractWordFeatures("i love my mom"))
print " "
print "Try your own examples and hit ctrl-c when finished."
while 1>0:
    query = raw_input("What's on your mind? ")
    print " "
    print query, predictReacts(weights, extractWordFeatures(query.strip().lower()))
    print " "


