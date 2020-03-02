# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:11:10 2020

@author: Gianl
"""


#BEST RF


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


best_features = ['interarrival_p25', 'interarrival_p50', 'interarrival_p75',
        'len_udp_std', 'len_udp_mean', 'num_packets', 'kbps', 'len_udp_p25',
        'len_udp_p50', 'len_udp_p75', 'len_udp_max_min_diff',
        'interlength_udp_max_min_diff', 'rtp_inter_timestamp_num_zeros',
        'inter_time_sequence_std', 'inter_time_sequence_max_min_diff']


X_train, y_train, X_test, y_test = _Dataset()
X_train = X_train[best_features] #SONO 15
X_test =  X_test[best_features]


clf = RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                       criterion='entropy', max_depth=50, max_features=5,
                       max_leaf_nodes=300, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=50, min_samples_split=10,
                       min_weight_fraction_leaf=0.0, n_estimators=150,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
clf.fit(X_train, y_train["label"])
clf.score(X_test, y_test["label"])
y_pred = clf.predict(X_test)
confusion_matrix_Cisco(y_test["label"],y_pred)
Report_Matrix(y_test["label"],y_pred)