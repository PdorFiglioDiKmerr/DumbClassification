import pandas as pd
import os
import matplotlib.pyplot as plt
from json import JSONDecoder, JSONDecodeError
import sys
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")
from utilities import save_photo, plot_learning_curve, ecdf

def classifiers(dataset_path, data, seconds, save_dir):
    warnings.filterwarnings("ignore")
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
    from sklearn.model_selection import ShuffleSplit
    
    dataset = pd.read_csv(dataset_path)
    colors = ['r', 'b']
    classes = [0, 1] # 0 = Not RTP and 1 = RTP
    columns = dataset.columns[:-1]
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1:]
    y[y.label == 'RTP'] = 1
    y[y.label == 'Not RTP'] = 0
    y = np.array(y.label)
    X = StandardScaler().fit_transform(X)
    print("dataset shape: " + str(dataset.shape))
    print("dataset classes distribution: %.2f%% RTP" % (100*len(y[y == 1])/len(y)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
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
    principalDf = pd.DataFrame(data = np.column_stack((PC[:, 0:2], y)), 
                               columns = ['principal component 1', 'principal component 2', 'label'])
    fig = plt.figure(figsize = (13,13))
    #plt.scatter(principalDf.iloc[:, 0], principalDf.iloc[:, 1])
    for target, color in zip(classes, colors):
        indicesToKeep = principalDf['label'] == target
        plt.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
                   , principalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color, s = 40)
    plt.xlabel('Principal Component 1', fontsize = 15)
    plt.ylabel('Principal Component 2', fontsize = 15, labelpad = -10)
    plt.legend(classes)
    plt.grid()
    t = "Data Plotting using PCA"
    plt.title(t, fontsize = 10)
    save_photo(save_dir, t, str(seconds)+"s")
    
    # #############################################################################
    # Feature characterization
    # #############################################################################
    labels = ['Not RTP', 'RTP']
    for feature in columns:
        for label in labels:
            if 'interarrival' in feature: color='r'
            elif 'len_udp' in feature: color = '#815EA4'
            elif "interlength" in feature: color = '#1E8449'
            elif "rtp_inter" in feature: color = 'c'
            elif "kbps" in feature: color = '#A82828'
            elif "num_packets" in feature: color = '#6C3483'

            plt.figure(figsize=(13,8))
            plt.grid()
            dataset[dataset.label == label][feature].hist(bins=50, density=True, color=color)
            t = feature + ' hist ' + label
            plt.title(t, fontsize=20)
            plt.tight_layout()
            save_photo(save_dir, t, str(seconds)+"s")

            plt.figure(figsize=(13,8))
            xplot, yplot = ecdf(dataset[dataset.label == label][feature])
            plt.plot(xplot, yplot, lw=3, color=color)
            plt.grid()
            t = feature + ' CDF ' + label
            plt.title(t, fontsize=20)
            plt.tight_layout()
            save_photo(save_dir, t, str(seconds)+"s")

    # #############################################################################
    # SVM
    # #############################################################################
    model = svm.SVC()
    C = [0.01, 0.1, 1, 10, 100, 1000]
    kernel = ['rbf']
    gamma = [0.001, 0.01, 0.1, 1, 10, 100]
    params = {'C': C, 'kernel': kernel, 'gamma': gamma}
    SVM = GridSearchCV(model, params, cv=10, n_jobs=-1, iid=True)
    SVM.fit(X_train, y_train)
    t = "SVM Learning Curve"
    plot_learning_curve(SVM.best_estimator_, t, X_train, y_train, ylim=(0.0, 1.10), cv=cv, n_jobs=-1)
    save_photo(save_dir, t, str(seconds)+"s")

    # #############################################################################
    # Random Forest
    # #############################################################################
    model = RandomForestClassifier()
    max_features = [3, 5, 7]
    n_estimators = [100, 200, 500, 1000, 2000]
    param_grid = {
        'max_features': max_features,
        'n_estimators': n_estimators
    }
    RF = GridSearchCV(model, param_grid, cv=10, n_jobs=-1)
    RF.fit(X_train, y_train)
    
    feature_imp = pd.Series(RF.best_estimator_.feature_importances_, index=columns)
    fig = plt.figure(figsize = (13,13))
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.legend()
    t = "Important Features"
    plt.title(t, fontsize = 10)
    save_photo(save_dir, t, str(seconds)+"s")
    t = "RF Learning Curve"
    plot_learning_curve(RF.best_estimator_, t, X_train, y_train, ylim=(0.0, 1.10), cv=cv, n_jobs=-1)
    save_photo(save_dir, t, str(seconds)+"s")
    
    # #############################################################################
    # KNN
    # #############################################################################
    model = KNeighborsClassifier()
    metric = ["manhattan", "euclidean", "chebyshev"]
    weights = ['uniform', 'distance']
    params = {"metric": metric, 'weights': weights, 'n_neighbors': range(1, 40)}
    KNN = GridSearchCV(model, params, cv=10, n_jobs=-1)
    KNN.fit(X_train, y_train)    
    t = "KNN Learning Curve"
    plot_learning_curve(KNN.best_estimator_, t, X_train, y_train, ylim=(0.0, 1.10), cv=cv, n_jobs=-1)
    save_photo(save_dir, t, str(seconds)+"s")

    # #############################################################################
    # Test Set composed by only Not RTP packets
    # #############################################################################
    X = X_test
    y = y_test
    data[seconds] = {}
    for clf, name in zip([SVM, RF, KNN], ['SVM', 'RF', 'KNN']):
        accuracy = clf.best_estimator_.score(X, y)
        RTP_accuracy = clf.best_estimator_.score(X[y == 1], y[y == 1])
        not_RTP_accuracy = clf.best_estimator_.score(X[y == 0], y[y == 0])
        
        t = name + ' accuracy and recall'
        plt.title(t, fontsize=16)
        plt.figure(figsize=(16, 9))
        hist_data = {"Accuracy": [accuracy], 
                     "RTP Accuracy": [RTP_accuracy],
                     "Not RTP Accuracy": [not_RTP_accuracy]}
        sns.barplot(data = pd.DataFrame(data=hist_data))
        plt.tight_layout()
        plt.grid()
        save_photo(save_dir, t, str(seconds)+"s")
        data[seconds][name] = accuracy

import matplotlib.pyplot as plt
from pcap_manager import pcap_manager
import pandas as pd
data = {}
dataset_path = "/home/det_tesi/sgarofalo/Window size analysis/unified_dataset/dataset.csv"
dataset_dir = dataset_path.rsplit('/', 1)[0]
save_dir = "/home/det_tesi/sgarofalo/Window size analysis"
for seconds in range(1, 11):
    seconds_samples = str(seconds) + "s"
    pm = pcap_manager(seconds_samples)
    print("Building datasets with seconds_samples = " + seconds_samples)
    pm.merge_pcap(dataset_dir)
    classifiers(dataset_path, data, seconds, save_dir)
    print()

columns = ["SVM", "RF", "KNN"]
df = pd.DataFrame(columns=columns)
for i in data:
    df = df.append({"SVM": data[i]["SVM"], "RF": data[i]["RF"], "KNN": data[i]["KNN"]}, ignore_index=True)
df.index +=  1 
plt.figure(figsize=(20, 16))
plt.plot(df)
plt.xlabel("Seconds")
plt.ylabel("Accuracy")
plt.legend(columns, fontsize = 16)
t = "Window size analysis"
plt.title(t, fontsize=16)
save_photo(save_dir, t)