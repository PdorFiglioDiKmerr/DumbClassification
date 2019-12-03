#!/usr/bin/env python
# coding: utf-8

# In[5]:
from utilities import save_photo, plot_learning_curve, ecdf

def classifiers(data, seconds, save_dir):
    import pandas as pd
    import numpy as np
    from sklearn import svm
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    import seaborn as sns
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    
    dataset = pd.read_csv("/home/det_tesi/sgarofalo/OneClassClassifier analysis/train_dataset/dataset.csv")
    colors = ['r', 'b']
    classes_str = ["Non RTP", "RTP"]
    classes = [-1, 1] # -1 = Not RTP and 1 = RTP
    #dataset = dataset[dataset.label == 'RTP'] #ONECLASSONLY CLASSIFIER
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1:]
    y[y.label == 'RTP'] = 1
    y[y.label == 'Not RTP'] = -1
    y = np.array(y.label)
    X = StandardScaler().fit_transform(X)
    print("Classifing with " + str(seconds) +" window size..")
    print("Training set classes distribution: %.2f%% RTP" % (100*len(y[y == 1])/len(y)))

    # #############################################################################
    # Correlation Matrix
    # #############################################################################
    df_corr = pd.DataFrame(data=np.column_stack((X, y)), columns = dataset.columns)
    corr = df_corr.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure(figsize = (10,10))
    ax = sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                annot = True, cmap="YlGnBu", mask = mask)
    t = "Correlation matrix"
    save_photo(save_dir, t, str(seconds)+"s")
    
    # #############################################################################
    # Plot Data Using PCA
    # #############################################################################
    myModel = PCA(2)
    PC = myModel.fit_transform(X)
    # print ("%.2f%% variance ratio with 2 PC" % (100*sum(myModel.explained_variance_ratio_)))
    principalDf = pd.DataFrame(data = np.column_stack((PC[:, 0:2], y)), columns = ['principal component 1', 'principal component 2', 'label'])
    fig = plt.figure(figsize = (13,13))
    #plt.scatter(principalDf.iloc[:, 0], principalDf.iloc[:, 1])
    for target, color in zip(classes, colors):
        indicesToKeep = principalDf['label'] == target
        plt.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
                   , principalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color, s = 40)
    plt.xlabel('Principal Component 1', fontsize = 15)
    plt.ylabel('Principal Component 2', fontsize = 15, labelpad = -10)
    plt.legend(classes_str)
    plt.grid()
    t = "Data Plotting using PCA"
    plt.title(t, fontsize = 10)
    save_photo(save_dir, t, str(seconds)+"s")
    
    # #############################################################################
    # One-class SVM
    # #############################################################################
    #SVM = svm.OneClassSVM(gamma = 'scale')
    #SVM.fit(X)    
    model = svm.OneClassSVM()
    nu = [0.1, 0.3, 0.5, 0.7, 0.9]
    kernel = ['rbf', 'poly']
    gamma = [0.001, 0.01, 0.1, 1, 10]
    params = {'kernel': kernel, 'gamma': gamma, 'nu': nu}
    SVM = GridSearchCV(model, params, cv=10, n_jobs=-1, iid=True, scoring='recall')
    SVM.fit(X, y)
    print('best score: %f' % (SVM.best_score_))
    SVM = SVM.best_estimator_
    
    # #############################################################################
    # Test Set
    # #############################################################################
    print()
    dataset = pd.read_csv("/home/det_tesi/sgarofalo/OneClassClassifier analysis/test_dataset/dataset.csv")
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1:]
    y[y.label == 'RTP'] = 1
    y[y.label == 'Not RTP'] = -1
    y = np.array(y.label)
    X = StandardScaler().fit_transform(X)
    print("test_dataset shape: " + str(dataset.shape))
    from sklearn.metrics import accuracy_score
    SVM_accuracy = accuracy_score(y, SVM.predict(X))
    SVM_RTP_accuracy = accuracy_score(y[y == 1], SVM.predict(X[y == 1]))
    SVM_non_RTP_accuracy = accuracy_score(y[y == -1], SVM.predict(X[y == -1]))
    print("SVM accuracy => %.2f%%" % (100*SVM_accuracy))
    print("SVM accuracy on RTP => %.2f%%" % (100*SVM_RTP_accuracy))
    print("SVM accuracy on Non-RTP => %.2f%%" % (100*SVM_non_RTP_accuracy))
    print()
    
    t = 'Accuracy and Recall'
    plt.title(t, fontsize=16)
    plt.figure(figsize=(16, 9))
    hist_data = {"Accuracy": [SVM_accuracy], 
                 "RTP Accuracy": [SVM_RTP_accuracy],
                 "Not RTP Accuracy": [SVM_non_RTP_accuracy]}
    sns.barplot(data = pd.DataFrame(data=hist_data))
    plt.tight_layout()
    plt.grid()
    save_photo(save_dir, t, str(seconds)+"s")
    
    data[seconds] = {}
    data[seconds]["SVM"] = SVM_accuracy
    
    
import matplotlib.pyplot as plt
from pcap_manager import pcap_manager
import pandas as pd
import numpy as np
import os
data = {}
save_dir = "/home/det_tesi/sgarofalo/OneClassClassifier analysis"
train_dir = "/home/det_tesi/sgarofalo/OneClassClassifier analysis/train_dataset"
test_dir = "/home/det_tesi/sgarofalo/OneClassClassifier analysis/test_dataset"
for seconds in range(1, 4):
    seconds_samples = str(seconds) + "s"
    pm = pcap_manager(seconds_samples)
    print("Building datasets with seconds_samples = " + seconds_samples)
    pm.merge_pcap(train_dir)
    pm.merge_pcap(test_dir)
    classifiers(data, seconds, save_dir)
    print()
columns = ["SVM"]
df = pd.DataFrame(columns=columns)
for i in data:
    df = df.append({"SVM": data[i]["SVM"]}, ignore_index=True)
df.index +=  1 
plt.figure(figsize=(20, 16))
plt.plot(df)
plt.xlabel("Seconds")
plt.ylabel("Accuracy")
plt.legend(columns, fontsize = 16)
t = "OneClassClassifier analysis"
plt.title(t, fontsize=16)
plt.grid()
plt.tight_layout()
save_photo(save_dir, t)