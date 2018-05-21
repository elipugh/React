# React
Stanford CS221 Final Project

We only consider non-like reactions (haha, love...). We normalized each reaction by the number of total non-like reactions and use this as our label. 

First featurizer: Used an n-gram model with TF-IDF to produce features. Playing around with some of the more important parameters, we obtained: 
  1 hidden layer with 15 neurons, with unigram and bigram: MSE of .04 and R^2 of .3
  2 hidden layers with 20 neurons each, with unigram and bigram: MSE of .05 and R^2 of .10
  3 hidden layers with 20 neurons each, with unigram and bigram: MSE of .06 and R^2 of 0
  3 hidden layers with 20 neurons each, with unigram through quad-gram: MSE of .05 and R^2 of .18
Experimenting with various metrics, we saw no noticeable gain above using 1 hidden layer with unigram and bigram features. 
