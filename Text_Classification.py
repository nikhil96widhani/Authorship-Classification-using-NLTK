#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 18:15:41 2019

@author: nikhilwidhani,
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk

#import 3 Gutenberg Books
from nltk.corpus import gutenberg
files_en = gutenberg.fileids()      # Get file ids
emma_en = gutenberg.open('austen-emma.txt').read()
sense_en = gutenberg.open('austen-sense.txt').read()
brown_en = gutenberg.open('chesterton-brown.txt').read()


#stopwords
from nltk.corpus import stopwords
stop_words= stopwords.words('english')

#Tokenization austen-emma.txt
from nltk import sent_tokenize, word_tokenize
emma_tokenize = word_tokenize(emma_en)

emma_tokenize = [w for w in emma_tokenize if w.isalpha()]
emma_tokenize = [w.lower() for w in emma_tokenize]
emma_tokenize = [w for w in emma_tokenize if not w in stop_words and len(w)>=3]

X1 = []
start = 0
end = 200*150 #no of document * no of words per document
for i in range(start, end,150):
    X1.append(emma_tokenize[i:i+150])
    
Y1 =[0]*200

#Tokenization austen-sense.txt
from nltk import sent_tokenize, word_tokenize
sense_tokenize = word_tokenize(sense_en)

sense_tokenize = [w for w in sense_tokenize if w.isalpha()]
sense_tokenize = [w.lower() for w in sense_tokenize]
sense_tokenize = [w for w in sense_tokenize if not w in stop_words and len(w)>=3]

X2 = []
start = 0
end = 200*150 #no of document * no of words per document
for i in range(start, end,150):
    X2.append(sense_tokenize[i:i+150])
    
Y2 =[1]*200

#Tokenization chesterton-brown.txt
from nltk import sent_tokenize, word_tokenize
brown_tokenize = word_tokenize(brown_en)

brown_tokenize = [w for w in brown_tokenize if w.isalpha()]
brown_tokenize = [w.lower() for w in brown_tokenize]
brown_tokenize = [w for w in brown_tokenize if not w in stop_words and len(w)>=3]

X3 = []
start = 0
end = 200*150 #no of document * no of words per document
for i in range(start, end,150):
    X3.append(brown_tokenize[i:i+150])
    
Y3 =[2]*200

#Append X1,X2,X3in X and Y1,Y2,Y3 in Y
X = [] 
Y = []
X = X1+X2+X3
Y = Y1+Y2+Y3
X = np.array(X)
Y = np.array(Y)

c = np.c_[X,Y]
np.random.shuffle(c)

X = c[:,0:150]
Y = c[:,-1:]

# =============================================================================
# def shuffle(X, Y):
#     permute = np.random.permutation(len(X))
#     X = X[permute]
#     Y = Y[permute]
#     return X, Y
# 
# X, Y = shuffle(X, Y)
# Y = np.reshape((Y), (-1,1))
# =============================================================================

#Converting into Data Frame
import pandas as pd
X = pd.DataFrame(X)
Y = pd.DataFrame(Y)

#Joining all columns to form a scentence for BOW
X=X.apply(" ".join, axis=1) #Joining all Columns to form Scentences

#BOW 
from sklearn.feature_extraction.text import CountVectorizer
matrixBoW = CountVectorizer(max_features=1000)
XBoW = matrixBoW.fit_transform(X).todense()
XBoW= pd.DataFrame(XBoW, columns= matrixBoW.get_feature_names())

# using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf= tf.fit_transform(X)

#Split train and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_tf, Y, test_size=0.3, random_state=0)


#Model Building and Evaluation (TF-IDF)
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from sklearn.metrics import accuracy_score
from sklearn import metrics

# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train.values.ravel())
y_pred= clf.predict(X_test)
print("Accuracy Naive Bayes:",metrics.accuracy_score(y_test, y_pred))

#SVC
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train.values.ravel()) 
y_pred = clf.predict(X_test)#Predict the response for test dataset
print("Accuracy SVC:",metrics.accuracy_score(y_test, y_pred))

#k neighbour BOW 
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=5)  
cl=classifier.fit(X_train, y_train.values.ravel())
y_pred=cl.predict(X_test)
print("Accuracy KNeighbours:",metrics.accuracy_score(y_test, y_pred))

#Decision Tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

clf = DecisionTreeClassifier() # Create Decision Tree classifer object
clf = clf.fit(X_train,y_train) # Train Decision Tree Classifer
y_pred = clf.predict(X_test) #Predict the response for test dataset
print("Accuracy Decision Tree:",metrics.accuracy_score(y_test, y_pred))



#TFIDF
# =============================================================================
# #Splitting into Train and Test data 
# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(XBoW, Y, test_size = 0.2, random_state=0)
# =============================================================================

# =============================================================================
# #Feature Scaling 
# from sklearn.preprocessing import StandardScaler
# sc_x = StandardScaler()
# x_train = sc_x.fit_transform(X_train)
# x_test =  sc_x.fit_transform(X_test)
# =============================================================================

# =============================================================================
# #K-fold But why?
# from sklearn.model_selection import KFold
# kfold = KFold(10, True, 1)
# # enumerate splits
# for train, test in kfold.split(X):
# 	print('train: %s, test: %s' % (X[train], X[test]))
# b=X[train]
# =============================================================================

# =============================================================================
# from nltk.tokenize import word_tokenize
# 
# emma_tokens = word_tokenize(emma_en)
# type(emma_tokens)
# print(emma_tokens[:22])
# 
# 
# import string
# #punctuation = str.maketrans("","", string.punctuation)
# #emma_filtered = [w.translate(punctuation) for w in emma_tokens]
# a1 = [w for w in emma_tokens if w.isalpha()]
# a2 = a1[:150]
# =============================================================================

