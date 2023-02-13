#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:29:57 2023

@author: ghaithalseirawan
"""

# import Python modules we'll need
import json
import numpy as np
import pandas as pd
from random import shuffle, sample
from collections import Counter
from sklearn.naive_bayes import MultinomialNB



directory = "/Users/ghaithalseirawan/Desktop/The Folder/Universtiy/YEAR 3/Second semester/Computational Linguistics/2nd lecture - Naïve Bayes/data"

data = json.load(open(directory + '/LangID_tweets.json', 'r'))
#print(data[:10])

# let's split the dataset in half: we'll use one as training material, the other as test
half_split = round(len(data)*0.5)
training_set = data[:half_split]
test_set = data[half_split:]

# we still have a lot of data, so let's downsize a bit so we don't have to wait forever 
training_sample = sample(training_set, round(len(training_set)*0.025))
test_sample = sample(test_set, round(len(test_set)*0.025))


# draw 5 random tweets with the associated language
indices = np.random.randint(len(training_sample), size=5)
selected = [training_sample[i] for i in indices]
#for lang, tweet in selected:
#    print("{}: {}".format(lang, tweet))
    

# split the labels and the tweets
training_labels, training_tweets = [], []
for lang, tweet in training_sample:
    training_labels.append(lang)
    training_tweets.append(tweet)
    
test_labels, test_tweets = [], []
for lang, tweet in test_sample:
    test_labels.append(lang)
    test_tweets.append(tweet)



# we see some languages: are they all the languages we have? 
# Let's see (and transform strings into numbers which makes everything handier)

# Link both direction lang and id & id and lang
labels2ids = {lang: i for i, lang in enumerate(set(training_labels))}
ids2labels = {i: lang for i, lang in enumerate(set(training_labels))}

Ytrain = [labels2ids[lang] for lang in training_labels]
Ytest = [labels2ids[lang] for lang in test_labels]

#print(labels2ids)

# Time to represent tweets as feature vectors.
# This function extracts character n-grams of arbitrary length

def ngram_featurizer(s, n):
    
    """takes in a string and an integer defining the size of ngrams.
     Returns the ngrams of desired size in the input string"""
    
    s = '#'*(n-1) + s + '#'*(n-1)
    ngrams = [s[i:i+n] for i in range(len(s)-n+1)]
    
    return ngrams

# Character n-grams capture the internal structure of words so we can represent a sentence internal features 
# taking the distribution 

# This function encodes all tweets as frequency counts over n-grams
def encode_corpus(corpus, n, mapping=None):
    
    """
    Takes in a list of strings, an integer indicating the character ngrams' size,
    and a dictionary mapping ngrams to numerical indices. If no dictionary is passed,
    one is created inside the function.
    The function outputs a 2d NumPy array with as many rows as there are strings in 
    the input list, and the mapping from ngrams to indices, representing the columns 
    of the NumPy array.
    """
    
    if not mapping:
        all_ngrams = set()
        for instance in corpus:
            all_ngrams = all_ngrams.union(set(ngram_featurizer(instance, n)))
    
        mapping = {ngram: i for i, ngram in enumerate(all_ngrams)}
    
    X = np.zeros((len(corpus), len(mapping)))
    for i, instance in enumerate(corpus):
        for ngram in ngram_featurizer(instance, n):
            try:
                X[i, mapping[ngram]] += 1
            except KeyError:
                pass
    
    return X, mapping

Xtrain, mapping = encode_corpus(training_tweets, 2)

# Naïve Bayes classifier
NB = MultinomialNB(alpha=1, fit_prior=True)
NB.fit(Xtrain,Ytrain)

