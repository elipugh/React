# React
Stanford CS221 Final Project

We only consider non-like reactions (haha, love...). We normalized each reaction by the number of total non-like reactions and use this as our label. 

First featurizer: Used an n-gram model with TF-IDF to produce features. MLP is using Adam optimization. Playing around with some of the more important parameters, we obtained: 
  1 hidden layer with 15 neurons, with unigram and bigram: MSE of .06 and R^2 of .20
  1 hidden layer with 100 neurons: MSE of .06 and R^2 of .20
  3 hidden layers with 100, 50, 10 neurons: MSE of .06 and R^2 of .22  
  
Experimenting with various metrics, we saw no significant gain  above using 1 hidden layer with unigram and bigram features. 

## Data
Retrieve glove dataset at: [http://nlp.stanford.edu/data/glove.6B.zip](here)
Unzip and copy into 'glove.6B' folder in this working directory
