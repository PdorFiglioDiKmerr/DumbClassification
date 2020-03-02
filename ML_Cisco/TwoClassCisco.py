# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 13:33:17 2020

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
from ConfusionMatrixCisco import confusion_matrix_Cisco
from PCACisco import PCA_Plot_3D,PCA_Plot_Variance


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

###### ANALISI MQ VS SS

X_train, y_train, X_test, y_test = _Dataset()
col_test = X_test.columns
col_train = X_train.columns
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns = col_train)
X_test = pd.DataFrame(X_test, columns = col_test)


def Plot_hist(df1, df2, name, title, label1, label2):

    plt.ioff()
    os.mkdir(name)
    for i in df_SS_test:
        plt.figure(figsize = (16,9))
        df1[i].hist(bins = 100, label = label1, alpha = 0.75)
        df2[i].hist(bins = 100, alpha = 0.5, label = label2)
        plt.title(title + "features: {}".format(i))
        plt.ylabel("Occurences")
        plt.xlabel("Support")
        plt.legend()
        plt.savefig(name+"/"+i+".png",dpi = 300)
    plt.close()

def build_class_df(df1, y_train, class1, class2):
    MQ_index = np.where (y_train["label"] == class1)
    SS_index = np.where (y_train["label"] == class2)
    df_MQ = df1.loc[MQ_index]
    df_SS = df1.loc[SS_index]
    y_MQ = y_train["label"].loc[MQ_index]
    y_SS = y_train["label"].loc[SS_index]
    return y_MQ, y_SS, df_MQ, df_SS

y_MQ, y_SS, df_MQ, df_SS = build_class_df(X_train, y_train, 0, 3)

df_train_2_class = pd.concat([df_MQ, df_SS], sort = False)
y_train_2_class = pd.concat([y_MQ, y_SS], sort = False)

y_MQ_test, y_SS_test, df_MQ_test, df_SS_test = build_class_df(X_test, y_test, 0, 3)

df_test_2_class = pd.concat([df_MQ_test, df_SS_test], sort = False)
y_test_2_class = pd.concat([y_MQ_test, y_SS_test], sort = False)

df_test_2_class.reset_index(drop = True, inplace = True)
df_train_2_class.reset_index(drop = True, inplace = True)
y_train_2_class.reset_index(drop = True, inplace = True)
y_test_2_class.reset_index(drop = True, inplace = True)

X_pcamaah, Pca = PCA_Plot_3D(df_train_2_class, y_train_2_class)
PCA_Plot_3D(df_test_2_class, y_test_2_class, Pca)
#outliers = np.where(X_pcamaah[:,0] > 15)[0]
df_train_2_class_no_out = df_train_2_class.drop(index = outliers, axis = 0)
y_train_2_class_no_out = y_train_2_class.drop(index = outliers, axis = 0)
X_pcamaah, Pca = PCA_Plot_3D(df_train_2_class_no_out, y_train_2_class_no_out)
X_pca_test, Pca = PCA_Plot_3D(df_test_2_class, y_test_2_class, Pca)
