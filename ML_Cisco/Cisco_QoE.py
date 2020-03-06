# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:27:49 2020

@author: Gianl
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.font_manager
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing
from BuildDataset import _Dataset
from ConfusionMatrixCisco import confusion_matrix_Cisco, Report_Matrix
import PCACisco
from sklearn.feature_selection import RFECV
from Learning_Curve_my import plot_learning_curve
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


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

new_name_col = ['interarr_std', 'interarr_mean', 'interarr_p25',
       'interarr_p50', 'interarr_p75', 'interarr_M_m_diff',
       'len_udp_std', 'len_udp_mean', 'num_packets', 'kbps', 'len_udp_p25',
       'len_udp_p50', 'len_udp_p75', 'len_udp_M_m_diff',
       'interlen_udp_mean', 'interlen_udp_p25', 'interlen_udp_p50',
       'interlen_udp_p75', 'interlen_udp_M_m_diff',
       'rtp_inter_time_std', 'rtp_inter_time_mean',
       'rtp_inter_time_n_zeros', 'rtp_interarr_M_m_diff',
       'inter_time_seq_std', 'inter_time_seq_mean',
       'inter_time_seq_p25', 'inter_time_seq_p50',
       'inter_time_seq_p75', 'inter_time_seq_M_m_diff']
#FEATURES OTTENUTE CON 0.03 DI TH 0.96X ACCURACY
best_features = ['interarr_p25', 'interarr_p50', 'interarr_p75', 'len_udp_std',
       'len_udp_mean', 'num_packets', 'kbps', 'len_udp_p75',
       'len_udp_M_m_diff', 'interlen_udp_M_m_diff', 'rtp_inter_time_mean']

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy




def generate_clf_from_search(grid_or_random, clf, parameters, scorer, X, y):
    if grid_or_random == "Grid":
        search_obj = GridSearchCV(clf, parameters, scoring=scorer)
    elif grid_or_random == "Random":
        search_obj = RandomizedSearchCV(clf, parameters, scoring=scorer, n_iter = 3000, random_state = 42, n_jobs = 47)
    fit_obj = search_obj.fit(X, y)
    best_clf = fit_obj.best_estimator_
    return best_clf
##########################################
##########################################
##########################################


#START

X_train, y_train, X_test, y_test = _Dataset()
col_test = new_name_col
col_train = new_name_col
X_train.columns = new_name_col
X_test.columns = new_name_col
X_train = X_train[best_features] #SONO 15
X_test =  X_test[best_features]

clf = RandomForestClassifier()
param_grid = {'bootstrap': [True, False],\
    'criterion': ['entropy', 'gini'],\
    'max_depth': [10, 20, 50, 100, 200, None],\
    'max_features': [3, 6, 9, None],\
    'min_samples_leaf': [50, 100, 200, 250, 300],\
    'min_samples_split': [10, 100, 300, 1000],\
    'n_estimators': [10, 50, 100, 150, 200],\
    'max_leaf_nodes' : [50, 100, 150, 200, 300]\
    }

best_clf = generate_clf_from_search("Random", clf, param_grid, None,X_train, y_train["label"])

import pickle

filename = "rf_param.pickle"
with open(filename, "wb") as f:
    pickle.dump(best_clf, f)

with open("RF_parameters.txt", "w") as f:
    f.write(str(best_clf.best_params_))
    best_grid = best_clf.best_estimator_
    grid_accuracy = evaluate(best_grid, X_test,  y_test["label"])
    f.write("\nAccuracy on test:\n")
    f.write(str(grid_accuracy))
