#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 18:15:41 2019

@author: nikhilwidhani, SimarpalSingh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
#scikit-learn metrics module for accuracy calculation
from sklearn import metrics
#Tokenization austen-emma.txt
from nltk import sent_tokenize, word_tokenize
#stopwords
from nltk.corpus import stopwords
#K Fold Cross Validation
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
#Import svm model
from sklearn.svm import SVC  

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#import 3 Gutenberg Books
from nltk.corpus import gutenberg
files_en = gutenberg.fileids()      # Get file ids
emma_en = gutenberg.open('austen-emma.txt').read()
sense_en = gutenberg.open('austen-sense.txt').read()
brown_en = gutenberg.open('chesterton-brown.txt').read()

def tokenize(data):
    tokenized_word = word_tokenize(data)
    tokenized_word = [w for w in tokenized_word if w.isalpha()]
    tokenized_word = [w.lower() for w in tokenized_word]
    stop_words= stopwords.words('english')
    tokenized_word = [w for w in tokenized_word if not w in stop_words and len(w)>=3]
    
    tokenized_word_list = []
    start = 0
    end = 200*150 #no of document * no of words per document
    for i in range(start, end, 150):
        tokenized_word_list.append(tokenized_word[i:i+150])
    return tokenized_word_list

emma_tokenized = tokenize(emma_en)
sense_tokenized = tokenize(sense_en)   
brown_tokenized = tokenize(brown_en)
    
Y1 =[0]*200
Y2 =[1]*200
Y3 =[2]*200

#Append emma_tokenized, sense_tokenized, brown_tokenized in X and Y1,Y2,Y3 in Y
X = [] 
Y = []
X = emma_tokenized + sense_tokenized + brown_tokenized
Y = Y1+Y2+Y3
X = np.array(X)
Y = np.array(Y)

C = np.c_[X, Y]
np.random.shuffle(C)

X = C[:, 0:150]
Y = C[:, -1:]

Y = np.reshape((Y), (-1,1))

#Converting into Data Frame
import pandas as pd
X = pd.DataFrame(X)
Y = pd.DataFrame(Y)

#Joining all columns to form a scentence for BOW
X=X.apply(" ".join, axis=1) #Joining all Columns to form Scentences

def check_accuracy(y_test, y_pred, modal, vectorizer): 
    print('Confusion Matrix :for ' + modal + ' using vectorizer ' + vectorizer)
    print(metrics.confusion_matrix(y_test, y_pred)) 
    print('Report : for ' + modal + ' using vectorizer ' + vectorizer)
    print(metrics.classification_report(y_test, y_pred))
    print('Accuracy: for ' + modal + ' using vectorizer ' + vectorizer)
    accuracy = np.asscalar(metrics.accuracy_score(y_test, y_pred))   
    print(accuracy)
    print('Mean square error: for ' + modal + ' using vectorizer ' + vectorizer)
    print(metrics.mean_squared_error(y_test, y_pred))
    print()
    return accuracy
    

def perform_analysis(vectorizer, vectorized_data):
    #Split train and test set 
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        vectorized_data, Y, test_size=0.25, shuffle=False, random_state=1234)
    
    svc_classifier = SVC(kernel='linear')  
    svc_classifier.fit(X_train, y_train.values.ravel()) 
    
    #Predict the response for test dataset
    svm_pred = svc_classifier.predict(X_test)    
    # Model Accuracy
    svm_accuracy = check_accuracy(y_test, svm_pred, 'SVM', vectorizer)
    
    #Import knearest neighbors Classifier model
    from sklearn.neighbors import KNeighborsClassifier    
    #Create KNN Classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=10)    
    #Train the model using the training sets
    knn_classifier.fit(X_train, y_train.values.ravel())    
    #Predict the response for test dataset
    knn_pred = knn_classifier.predict(X_test)
    # Model Accuracy
    knn_accuracy = check_accuracy(y_test, knn_pred, 'KNN', vectorizer)
    
    # Import Decision Tree Classifier
    from sklearn.tree import DecisionTreeClassifier 
    # Create Decision Tree classifer object
    dt_classifier = DecisionTreeClassifier(criterion="entropy", max_depth=3)    
    # Train Decision Tree Classifer
    dt_classifier.fit(X_train,y_train.values.ravel())    
    #Predict the response for test dataset
    dt_pred = dt_classifier.predict(X_test)    
    dt_accuracy = check_accuracy(y_test, dt_pred, 'Decision Tree', vectorizer)
    
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    model = LogisticRegression()
    results = model_selection.cross_val_score(model, vectorized_data, Y.values.ravel(), cv=kfold)
    kfold_accuracy = results.mean()
    print("Accuracy for K fold using " + vectorizer)
    print(kfold_accuracy)
     
    #Plot the graph
    objects = ('SVM', 'KNN', 'Decision Tree', 'kfold')
    y_pos = np.arange(len(objects))
    performance = [svm_accuracy, knn_accuracy, dt_accuracy, kfold_accuracy]     
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Percentage')
    plt.title('Accuracy of different modals using ' + vectorizer)     
    plt.show()
    return

#using BOW 
from sklearn.feature_extraction.text import CountVectorizer
matrixBoW = CountVectorizer(max_features=1000)
XBoW = matrixBoW.fit_transform(X).todense()
XBoW= pd.DataFrame(XBoW, columns= matrixBoW.get_feature_names())
perform_analysis('BOW', XBoW)

#using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf= tf.fit_transform(X)
perform_analysis('TF-IDF', text_tf)