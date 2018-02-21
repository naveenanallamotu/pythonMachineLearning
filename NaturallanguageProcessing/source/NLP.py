# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 12:16:18 2017

@author: NAVEENA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the data

text = pd.read_csv("Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)

# cleaning the data
import re 
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
nltk.download('stopwords') # stopword takes list 
# we need convert into different
from nltk.corpus import stopwords
corpus = []
for i in range(0,1000):
    review =re.sub("[^a-zA-Z]",' ' ,text['Review'][i]) # removing the puncation
    review =review.lower()#
    review = review.split() #sreview is list
    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review) #string
    corpus.append(review)
#bags of word model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y =text.iloc[:,1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = (cm[0][0] +cm[1][1])/(cm[0][0] + cm[0][1]+ cm[1][0]+ cm[1][1])
recall = (cm[1][1])/(cm[1][1]+cm[0][0])
precission = (cm[1][1])/(cm[1][1]+cm[0][0])
F1_Score = 2 * precission * recall / (precission + recall)



 