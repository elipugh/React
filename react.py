#!/usr/bin/python
# -*- coding: utf-8 -*-
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
# csv -> array of tuples of (status, reaction array)

def loadCsvData(fileName):
    examples = []
    # open a file
    with open(fileName) as f:
        reader = csv.DictReader(f)

        # loop over each row in the file
        for row in reader:

            # cast each value to a float
            examples.append((row['status_message'].strip().lower(), np.array([float(row['num_loves']), float(row['num_wows']),\
             float(row['num_hahas']), float(row['num_sads']), float(row['num_angrys'])])))
    print 'Read %d examples from %s' % (len(examples), fileName)
    return examples

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

############################################################
# go through training results and return error

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
        sum = np.sum(y)
        if sum != 0:
            y /= sum
    trainExamples = trainExamples2

    for i in range(numIters):
        eta = 1 / math.sqrt(1.+i)
        for x,y in trainExamples:
            grad = gradient(weights, x, y)
            for k in grad:
                if k in weights:
                    weights[k] -= grad[k]*eta
                else:
                    weights[k] = -grad[k]*eta

        if (((i+1) % 20) == 0):
            for x,y in weights.iteritems():
                norm = np.linalg.norm(y)
                if norm != 0:
                    y /= (norm/2)
            if verbose == 1:
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
        x = x.replace(" ", "")
        result = {}
        for i in range(0, len(x)-n+1):
            feature = x[i:i+n]
            if feature in result:
                result[feature] += 1
            else:
                result[feature] = 1
        return result
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
    sum = np.sum(result)
    if sum != 0.0:
        result /= sum
    return result

############################################################
# print predicted reactions with emojis
def printprediction(prediction):
    for i in range(len(prediction)):
        print int((100*prediction[i])+0.5), emojis[i]
    print " "

############################################################
#                        SCRIPT                            #
############################################################

print " "
filename = raw_input("CSV dataset: ")
examples = loadCsvData(filename)
#np.random.shuffle(examples)
trainsz = 8*len(examples)/10
trainExamples = examples[:trainsz]
testExamples = examples[trainsz:]
featureExtractor = extractWordFeatures
weights = learnPredictor(trainExamples, testExamples, featureExtractor, numIters=100, eta=1, verbose=1)
emojis = ["❤️", "😆", "😲", "😢", "😡"]
print " "
print " "
print "Today the president signed a wonderful bill"
printprediction(predictReacts(weights, extractWordFeatures("today the president signed a wonderful bill")))
print "New cancer cure discovered is incredible"
printprediction(predictReacts(weights, extractWordFeatures("new cancer cure discovered is incredible")))
print "this program sucks. it's the worst"
printprediction(predictReacts(weights, extractWordFeatures("this program sucks its the worst")))
print "I love my mom."
printprediction(predictReacts(weights, extractWordFeatures("i love my mom")))
print " "
print "Try your own examples and type \"q\" or hit ctrl-c when finished."
while 1>0:
    query = raw_input("What's on your mind? ")
    if query == "q":
        print " "
        break
    prediction = predictReacts(weights, extractWordFeatures(query.strip().lower()))
    printprediction(prediction)


