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
import random
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

t0 = time.time()
# Importing the dataset
inputsize = sys.stdin.readline().rstrip().split(' ')
N = int(inputsize[0])
M = int(inputsize[1])
X_train = np.zeros((N,M)) #zero array of N rows, M columns
y_train = np.zeros((N)).astype(int)

toBinary = {"+1" : 1, "-1" : 0}
toOutput = {"1" : "+1", "0" : "-1"}

for row in range(N):
    line = sys.stdin.readline().rstrip().split(' ')
    y_train[row] = toBinary[line[1]]
    for col in range(M):
        X_train[row,col] = float(line[col+2].split(':')[1])

N_test = int(sys.stdin.readline().rstrip())
names = np.chararray((N_test), itemsize=5)
X_test = np.zeros((N_test,M)) #zero array of N rows, M columns

for row in range(N_test):
    line = sys.stdin.readline().rstrip().split(' ')
    names[row] = line[0]
    for col in range(M):
        X_test[row,col] = float(line[col+1].split(':')[1])

# Splitting the dataset into the Training set and Test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random.randint(1, 100))
grad_clf = GradientBoostingClassifier(max_depth=1, n_estimators=300, learning_rate=0.3)
grad_clf.fit(X_train, y_train)

y_pred_rf = grad_clf.predict(X_test).astype(str)
y_pred_rf = np.array([toOutput[row] for row in y_pred_rf])

out = np.column_stack((names, y_pred_rf))
np.savetxt('answers.txt', out, delimiter=' ', fmt='%s')

print(time.time() - t0)
