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
        for row in reader:
            x = row['status_message'].strip().lower()
            y = np.array([float(row['num_loves']), float(row['num_wows']),\
                float(row['num_hahas']), float(row['num_sads']), float(row['num_angrys'])])
            examples.append((x,y))
    print 'Read %d examples from %s' % (len(examples), fileName)
    return examples

############################################################
# simple baseline unigram feature extraction

def extractUnigramFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    arr = x.split()
    d={}
    for word in arr:
        if word[:4] == "http":
            continue
        if word in d:
            d[word]+= 1
        else:
            d[word]=1
    return d

############################################################
# simple baseline bigram feature extraction

def extractBigramFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    Example: "I am what I am" --> {(I,am): 2, (am,what): 1, (what,I): 1}
    """
    arr = x.split()
    d= {}
    for i in range(len(arr)-1):
        feature = (arr[i], arr[i+1])
        if arr[i][:4] == "http" or arr[i+1][:4] == "http":
            continue
        if feature in d:
            d[feature]+= 1
        else:
            d[feature]=1
    return d

############################################################
# features include both unigram and bigram

def extractCombogramFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    Example: "I am what I am" --> {(I,am): 2, (am,what): 1, (what,I): 1}
    """
    arr = x.split()
    d= extractUnigramFeatures(x)
    for i in range(len(arr)-1):
        feature = (arr[i], arr[i+1])
        if feature in d:
            d[feature]+= 1
        else:
            d[feature]=1
    return d

############################################################
# sliding character window feature extraction

def extractCharacterFeatures(x):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    n = 6
    x = x.replace(" ", "")
    result = {}
    for i in range(0, len(x)-n+1):
        feature = x[i:i+n]
        if feature in result:
            result[feature] += 1
        else:
            result[feature] = 1
    return result

############################################################
# gradient of squared loss is just (prediction - y)

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
# Run regression with SGD

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta, verbose):

    # boring stuff. converting tuples from (text,reacts) to (features,reacts)
    temp = []
    for x,y in trainExamples:
        temp.append((featureExtractor(x),y))
        sum = np.sum(y)
        if sum != 0:
            y /= sum
    trainExamples = temp
    temp = []
    for x,y in testExamples:
        temp.append((featureExtractor(x),y))
        sum = np.sum(y)
        if sum != 0:
            y /= sum
    testExamples = temp

    # let's start learning
    weights = {}  # feature => weight
    for i in range(numIters):

        if ((i % 20) == 0):                     # check in on how we're doing and
            #for x,y in weights.iteritems():    # normalize every once in a while
            #    norm = np.linalg.norm(y)       # to make sure the weights aren't
            #    if norm != 0:                  # just massive for some words
            #        y /= (norm/5)              # accuracy is better w/o this
            if verbose == 1:
                print " "
                print "epoch ", i, " out of ", numIters
                print "avg train error: ", geterror(trainExamples,weights)
                print "avg test error: ", geterror(testExamples,weights)
            else:
                print "epoch ", i, " out of ", numIters

        # The SGD step over the entire train set
        eta = 1.0 / math.sqrt(1.0+i)   # eta shrinks with time
        for x,y in trainExamples:
            grad = gradient(weights, x, y)
            for k in grad:
                if k in weights:
                    weights[k] -= grad[k]*eta
                else:
                    weights[k] = -grad[k]*eta

    print " "
    print "finished learning! getting accuracy..."
    print "final train error: ", geterror(trainExamples,weights)
    print "final test error: ", geterror(testExamples,weights)
    print "There are ", len(weights), " weights"

    return weights

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
    emojis = ["❤️ ", "😆", "😲", "😢", "😡"]
    for i in range(len(prediction)):
        print emojis[i], ": ", int((100*prediction[i])+0.5)
    print " "

############################################################
#                        SCRIPT                            #
############################################################

print " "
filename = raw_input("CSV dataset: ")
examples = loadCsvData(filename)
np.random.shuffle(examples)
trainsz = 8*len(examples)/10
trainExamples = examples[:trainsz]
testExamples = examples[trainsz:]
featureExtractor = extractCombogramFeatures
printerror = int(raw_input("Show error while training? (0 or 1): "))
iterations = int(raw_input("How much training? (50-200 recommended): "))

# now train the algorithm
weights = learnPredictor(trainExamples, testExamples, featureExtractor, numIters=iterations, eta=1, verbose=printerror)

# the fun part - seeing some example results
print " "
print "Today the president signed a wonderful bill"
printprediction(predictReacts(weights, extractCombogramFeatures("today the president signed a wonderful bill")))
print "New cancer cure discovered is incredible"
printprediction(predictReacts(weights, extractCombogramFeatures("new cancer cure discovered is incredible")))
print "this program is terrible. it's the worst"
printprediction(predictReacts(weights, extractCombogramFeatures("this program sucks its the worst")))
print "I love my mom. She is the best"
printprediction(predictReacts(weights, extractCombogramFeatures("i love my mom she is the best")))
print "Try your own examples and type \"q\" or hit ctrl-c when finished."
print " "
# allow the user to try some
while 1>0:
    query = raw_input("What's on your mind? ")
    if query == "q":
        print " "
        break
    prediction = predictReacts(weights, extractCombogramFeatures(query.strip().lower()))
    printprediction(prediction)
