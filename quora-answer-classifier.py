# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 08:40:39 2018

@author: Jack Geissinger
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

t0 = time.time()
# Importing the dataset
inputsize = sys.stdin.readline().split(' ')
N = int(inputsize[0])
M = int(inputsize[1])
X = numpy.zeros((N,M)) #zero array of N rows, M columns
y = numpy.zeros((N,1))
names = numpy.zeros((N,1))

for row in range(N):
    line = sys.stdin.readine().split(' ')
    name[row,1] = line[0]
    y[row,1] = line[1]
    for col in range(M):
        X[row,col] = float(line[col+2].split(':')[1])


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

grad_clf = GradientBoostingClassifier(max_depth=10, n_estimators=10, learning_rate=0.5)
grad_clf.fit(X_train, y_train)

y_pred_rf = grad_clf.predict(X_test)

v = accuracy_score(y_test, y_pred_rf)
print(v)

print("Time elapsed: ", time.time() - t0)