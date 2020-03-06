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
import matplotlib.font_manager
from BuildDataset import _Dataset
from ConfusionMatrixCisco import confusion_matrix_Cisco, Report_Matrix
import PCACisco
from Learning_Curve_my import plot_learning_curve
import pickle


matplotlib.rcParams.update({'font.size': 30})
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

file = open("rf_param.pickle", 'rb')

# dump information to that file
clf = pickle.load(file)
# close the file
file.close()

X_train, y_train, X_test, y_test = _Dataset()
col_test = new_name_col
col_train = new_name_col
X_train.columns = new_name_col
X_test.columns = new_name_col
X_train = X_train[best_features] #SONO 15
X_test =  X_test[best_features]

#clf.fit(X_train, y_train["label"])
print(clf.score(X_test, y_test["label"]))
y_pred = clf.predict(X_test)
confusion_matrix_Cisco(y_test["label"],y_pred, save = True)
Report_Matrix(y_test["label"],y_pred, save = True)

plot_learning_curve(clf, "Best clf", X_train, y_train["label"], X_test = X_test, y_test = y_test["label"])