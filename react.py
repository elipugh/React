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

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    arr = x.split()
    d={}
    for word in arr:
        if word in d:
            d[word]+= 1
        else:
            d[word]=1
    return d
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def sparsedp(d1, d2):
    result = 0.0
    for k in d1:
        if k in d2:
            result+= d1[k] * d2[k]
    return result

def gradient(weights, features, y):
    if sparsedp(features, weights) *y > 1:
        return {}
    else:
        result={}
        for k in features:
            result[k] = y * features[k]
        return result


def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    def predict(x):
        if sparsedp(weights, x) < 0:
            return -1
        return 1

    for i in xrange(numIters):
        for x,y in trainExamples:
            features = featureExtractor(x)
            grad = gradient(weights, features, y)
            for k in grad:
                if k in weights:
                    weights[k] += grad[k]*eta
                else:
                    weights[k] = grad[k]*eta

    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE
        phi = {}
        numwords = random.randint(1,300)
        for i in range(1, numwords):
            toadd = random.choice(weights.keys())
            if toadd in phi:
                phi[toadd] += 1
            else:
                phi[toadd] = 1
        y = 1
        if sparsedp(weights, phi) < 0:
            y = -1
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

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
#                        SCRIPT                            #
############################################################

filename = raw_Input("CSV dataset: ")
examples = readExamples('polarity.train')
np.random.shuffle(examples)
trainsz = 8*len(examples)/10
trainExamples = examples[:,trainsz]
testExamples = examples[trainsz,:]




