# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:46:59 2020

@author: Gianl
"""


import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
import matplotlib.pyplot as plt
from BuildDataset import _Dataset
from sklearn import preprocessing
import matplotlib
from ConfusionMatrixCisco import confusion_matrix_Cisco, Report_Matrix


matplotlib.rcParams.update({'font.size': 30})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 15

X_train, y_train, X_test, y_test = _Dataset()
col_test = X_test.columns
col_train = X_train.columns
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns = col_train)
X_test = pd.DataFrame(X_test, columns = col_test)

prior = [1/7 ] * 7
clf = GaussianNB(priors = prior)
clf.fit(X_train, y_train["label"])

print(clf.score(X_test, y_test["label"]))
y_predict = clf.predict(X_test)

confusion_matrix_Cisco(y_test["label"], y_predict)
Report_Matrix(y_test["label"], y_predict)


clf_pf = GaussianNB(priors = prior)
clf_pf.partial_fit(X_train, y_train["label"], np.unique(y_train["label"]))
print(clf_pf.score(X_test, y_test["label"]))
y_predict_pf = clf_pf.predict(X_test)

confusion_matrix_Cisco(y_test["label"], y_predict_pf)
Report_Matrix(y_test["label"], y_predict_pf)


