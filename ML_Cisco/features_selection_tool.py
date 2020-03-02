# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:38:02 2020

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
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import f_classif


plt.ioff()

def RF_with_Graph(X_train, y_train, X_test = None, y_test = None):
    from sklearn import metrics
    clf_ = RandomForestClassifier(n_estimators = 50, max_depth = 20, criterion="entropy",\
                                  bootstrap= True, max_features = None, n_jobs = 4,\
                                  min_samples_leaf = 300, max_leaf_nodes = 80, \
                                  min_samples_split = 1000, random_state = 30)
    clf = RFECV(clf_, step=1, cv = StratifiedKFold(5), scoring = 'accuracy')
    clf = clf.fit(X_train, y_train)
    plt.ioff()
    # Plot number of features VS. cross-validation scores
    plt.ioff()
    plt.figure(figsize =(16,9))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(clf.grid_scores_) + 1), clf.grid_scores_)
    plt.grid()
    plt.savefig("Test.png", dpi = 300)
    plt.close()
    #clf.fit(X_train, y_train)
    # plt.title('Feature Importances')
    # plt.barh(X_train.columns, clf.feature_importances_, color='#0092CB', align='center')
    # plt.xlabel('Relative Importance')
    # plt.grid()
    # plt.show()
    # if X_test is not None and y_test is not None:
    #     print("Accuratezza Random Forest: %s" % clf.score(X_test, y_test))
    #     print("Accuratezza Random Forest Train: %s" % clf.score(X_train, y_train))
    return clf

def RandomF_(X,y,X_test,y_test, title, save = False):
    rm = RandomForestClassifier (n_estimators = 100)
    rm.fit(X,y)
    print(title + " accuracy: " +str(rm.score(X_test,y_test)))
    y_class_predict = rm.predict(X_test)
    #FEATURE IMPORTANCE
    plt.figure(figsize = [32,18])
    plt.title('Feature Importances ' + title) 
    plt.barh(X.columns, rm.feature_importances_, color='#0092CB', align='center')
    plt.xlabel('Relative Importance')
    plt.grid()
    if save:
        plt.savefig("fi_"+title+".png", dpi = 300)
    else:
        plt.show()
    #REPORT
    confusion_matrix_Cisco(y_test, y_class_predict, title, save = True)
    Report_Matrix(y_test, y_class_predict, title, save = True)

def Extra_ (X, y, save = False):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel 
    title = "tree_classifier"
    clf = ExtraTreesClassifier(n_estimators=100)
    clf = clf.fit(X, y)
    plt.figure(figsize = [32,18])
    plt.title('Feature Importances')
    plt.barh(X.columns, clf.feature_importances_, color='#0092CB', align='center')
    plt.xlabel('Relative Importance')
    plt.grid()
    if save:
        plt.savefig("extra_"+title+".png", dpi = 300)
    else:
        plt.show()
    return clf


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

from sklearn.naive_bayes import GaussianNB
X_train, y_train, X_test, y_test = _Dataset()
col_test = X_test.columns
col_train = X_train.columns
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns = col_train)
X_test = pd.DataFrame(X_test, columns = col_test)

# generate dataset
for i in range(10,20):
    fs = SelectKBest(score_func=mutual_info_classif, k=i)
    # apply feature selection
    X_selected = fs.fit_transform(X_train, y_train["label"])
    print(X_selected.shape)
    
    clf = GaussianNB()
    clf.fit(X_selected, y_train["label"])
    
    X_test_selected = fs.transform(X_test)
    print(clf.score(X_test_selected, y_test["label"]))
y_predict = clf.predict(X_test)


# estimator = RandomForestClassifier(n_estimators = 50, max_depth = 20, criterion="entropy",\
#                               bootstrap= True, max_features = None, n_jobs = 1,\
#                               min_samples_leaf = 300, max_leaf_nodes = 80, \
#                               min_samples_split = 1000, random_state = 30)

# title = "Random Forest Learning curves"
# clas = plot_learning_curve(estimator, title, X_train, y_train["label"], ylim=(0.5, 1.01),\
#                      n_jobs=4, X_test = X_test, y_test = y_test["label"], cv = 3)


# df_ = pd.concat ([X_train,y_train], axis = 1)
# df_ = df_.drop(["label2"], axis = 1)
# plt.figure(figsize=(12,10))
# cor = df_.corr()
# sns.heatmap(cor, cmap=plt.cm.Reds)
# plt.show()
# #Correlation with output variable
# cor_target = abs(cor["label"])
# #Selecting highly correlated features
# relevant_features = cor_target[cor_target>0.5]
# #GUARDO SE SONO CORRELATE ANCHE TRA DI LORO
# cor_rid = df_[relevant_features.index].corr()
# plt.figure(figsize=(12,10))
# sns.heatmap(cor_rid, cmap=plt.cm.Reds, annot = True)
# plt.show()


#STANDARDIZE FEATUREES
col_train = X_train.columns
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

X_train_s = pd.DataFrame(X_train_s, columns = col_train)
X_test_s = pd.DataFrame(X_test_s, columns = col_train)

#FEATURES SELECTION NO SCALE
clf_ = Extra_(X_train, y_train["label"], save = True)
X_train_select = X_train.loc[:, clf_.feature_importances_  > 0.03 ]
X_test_select = X_test.loc[:, clf_.feature_importances_  > 0.03]

#FEATURES SELECTION SCALE
X_train_select_s = X_train_s.loc[:, clf_.feature_importances_  > 0.03 ]
X_test_select_s = X_test_s.loc[:, clf_.feature_importances_  > 0.03]


#PLOT CORRELATION NEW FEATURES
# df_ = pd.concat ([X_train_select,y_train], axis = 1)
# df_ = df_.drop(["label2"], axis = 1)
# plt.figure(figsize=(12,10))
# cor = df_.corr()
# sns.heatmap(cor, cmap=plt.cm.Reds)
# plt.show()

RandomF_(X_train, y_train["label"], X_test, y_test["label"], "Complete Dataset", save = True) #complete dataset
RandomF_(X_train_s, y_train["label"], X_test_s, y_test["label"], "Complete Dataset standardaize", save = True) #complete dataset rescale
RandomF_(X_train_select, y_train["label"], X_test_select, y_test["label"], "Features selected", save = True) #features selection 
RandomF_(X_train_select_s, y_train["label"], X_test_select_s, y_test["label"], "Features selected standardize", save = True) #features selection scale

PCACisco.PCA_Plot_Variance(X_train_s)

from sklearn.decomposition import PCA

for i in [0.9, 0.95, 0.99]:
    
    pca = PCA(n_components=i)
    X_train_s_pca = pca.fit_transform(X_train_s)
    X_test_s_pca = pca.transform(X_test_s)
    n_comp = len(pca.singular_values_)
    X_train_s_pca = pd.DataFrame(X_train_s_pca)
    X_test_s_pca = pd.DataFrame(X_test_s_pca)
    RandomF_(X_train_s_pca, y_train["label"], X_test_s_pca, y_test["label"],\
             "PCA n_components " + str(n_comp), save = True) #complete dataset
    
    
