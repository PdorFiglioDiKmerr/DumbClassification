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

def classifiers(data, n_PCA, save_dir):
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
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB
    
    dataset_path = "/home/det_tesi/sgarofalo/GridSearchGNB/train_dataset/dataset.csv"
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
    print("train_dataset shape: " + str(dataset.shape))
    print("train_dataset classes distribution: %.2f%% RTP" % (100*len(y[y == 1])/len(y)))
    
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
    save_photo(save_dir, t, str(n_PCA)+" PCA")
    
    # #############################################################################
    # Plot Data Using PCA
    # #############################################################################
    myModel = PCA(n_PCA)
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
    save_photo(save_dir, t, str(n_PCA)+" PCA")
    '''
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
            save_photo(save_dir, t, str(n_PCA)+" PCA")

            plt.figure(figsize=(13,8))
            xplot, yplot = ecdf(dataset[dataset.label == label][feature])
            plt.plot(xplot, yplot, lw=3, color=color)
            plt.grid()
            t = feature + ' CDF ' + label
            plt.title(t, fontsize=20)
            plt.tight_layout()
            save_photo(save_dir, t, str(n_PCA)+" PCA")
    
    # #############################################################################
    # SVM
    # #############################################################################
    X_train, X_test, y_train, y_test = train_test_split(PC, y, test_size=0.3, random_state=1)
    model = svm.SVC()
    C = [0.01, 0.1, 1, 10, 100, 1000]
    kernel = ['rbf']
    gamma = [0.001, 0.01, 0.1, 1, 10, 100]
    params = {'C': C, 'kernel': kernel, 'gamma': gamma}
    SVM = GridSearchCV(model, params, cv=5, n_jobs=-1, iid=True)
    SVM.fit(X_train, y_train)
    SVM = SVM.best_estimator_
    t = "SVM Learning Curve"
    plot_learning_curve(SVM, t, X, y, ylim=(0.0, 1.10), cv=cv, n_jobs=-1)
    save_photo(save_dir, t, str(n_PCA)+" PCA")
    '''
    # #############################################################################
    # Gaussian Naive Bayes
    # #############################################################################
    GNB = GaussianNB()
    GNB.fit(PC, y)
    '''
    # #############################################################################
    # MultiLayerPerceptron
    # #############################################################################
    classifier = "MLP"
    model = MLPClassifier()
    hidden_layer_sizes = [(50,50,50), (50,100,50), (100,)]
    max_iter = [200, 1000, 5000, 10000]
    activation = ['tanh', 'relu']
    alpha = [0.0001, 0.05]
    solver = ['sgd', 'adam']
    params = {'hidden_layer_sizes': hidden_layer_sizes, 'max_iter': max_iter, 'activation': activation, 'alpha': alpha}
    MLP = GridSearchCV(model, params, cv=5, n_jobs=-1, iid=True)
    MLP.fit(X_train, y_train)
    MLP = MLP.best_estimator_
    
    # #############################################################################
    # Random Forest
    # #############################################################################
    model = RandomForestClassifier()
    max_leaf = [5, 10]
    min_samples = [1, 3]
    min_samples_split = [2, 16, 32]
    max_depth = [None, 8, 32]
    max_features = [3, 5, 7]
    n_estimators = [200, 500, 1000, 2000]
    params = {
        #'max_features': max_features,
        'n_estimators': n_estimators
        #'max_depth': max_depth, 
		#'min_samples_split': min_samples_split, 
		#'max_leaf_nodes': max_leaf, 
		#'min_samples_leaf': min_samples
    }
    RF = GridSearchCV(model, params, cv=10, n_jobs=-1)
    RF.fit(X_train, y_train)
    RF = RF.best_estimator_
    
    feature_imp = pd.Series(RF.feature_importances_, index=columns)
    fig = plt.figure(figsize = (13,13))
    sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.legend()
    t = "Important Features"
    plt.title(t, fontsize = 10)
    save_photo(save_dir, t, str(seconds)+"s")
    
    t = "RF Learning Curve"
    plot_learning_curve(RF, t, X, y, ylim=(0.0, 1.10), cv=cv, n_jobs=-1)
    save_photo(save_dir, t, str(seconds)+"s")
    
    # #############################################################################
    # KNN
    # #############################################################################
    model = KNeighborsClassifier()
    metric = ["manhattan", "euclidean", "chebyshev"]
    weights = ['uniform', 'distance']
    params = {"metric": metric, 'weights': weights, 'n_neighbors': range(1, 20)}
    KNN = GridSearchCV(model, params, cv=10, n_jobs=-1)
    KNN.fit(X_train, y_train)
    KNN = KNN.best_estimator_
    
    t = "KNN Learning Curve"
    plot_learning_curve(KNN, t, X, y, ylim=(0.0, 1.10), cv=cv, n_jobs=-1)
    save_photo(save_dir, t, str(n_PCA)+" PCA")
    '''
    # #############################################################################
    # Test Set
    # #############################################################################
    dataset = pd.read_csv("/home/det_tesi/sgarofalo/GridSearchGNB/test_dataset/dataset.csv")
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1:]
    y[y.label == 'RTP'] = 1
    y[y.label == 'Not RTP'] = 0
    print("test_dataset shape: " + str(dataset.shape))
    print("test_dataset classes distribution: %.2f%% RTP" % (100*len(y[y == 1])/len(y)))
    y = np.array(y.label)    
    X = StandardScaler().fit_transform(X)
    
    myModel = PCA(n_PCA)
    PC = myModel.fit_transform(X)
    X = PC
    
    data[n_PCA] = {}
    accuracy = GNB.score(X, y)
    RTP_accuracy = GNB.score(X[y == 1], y[y == 1])
    not_RTP_accuracy = GNB.score(X[y == 0], y[y == 0])
    t = 'GNB accuracy and recall'
    plt.title(t, fontsize=16)
    plt.figure(figsize=(16, 9))
    hist_data = {"Accuracy": [accuracy],
                 "RTP Accuracy": [RTP_accuracy],
                 "Not RTP Accuracy": [not_RTP_accuracy]}
    sns.barplot(data = pd.DataFrame(data=hist_data))
    plt.tight_layout()
    plt.grid()
    save_photo(save_dir, t, str(n_PCA) + " PCA")
    data[n_PCA] = accuracy
    
    '''
    for clf, name in zip([SVM, RF, KNN, GNB], ['SVM', 'RF', 'KNN', 'GNB']):
        accuracy = clf.score(X, y)
        RTP_accuracy = clf.score(X[y == 1], y[y == 1])
        not_RTP_accuracy = clf.score(X[y == 0], y[y == 0])
        
        t = name + ' accuracy and recall'
        plt.title(t, fontsize=16)
        plt.figure(figsize=(16, 9))
        hist_data = {"Accuracy": [accuracy],
                     "RTP Accuracy": [RTP_accuracy],
                     "Not RTP Accuracy": [not_RTP_accuracy]}
        sns.barplot(data = pd.DataFrame(data=hist_data))
        plt.tight_layout()
        plt.grid()
        save_photo(save_dir, t, str(n_PCA)+" PCA")
        data[n_PCA][name] = accuracy
    '''
import matplotlib.pyplot as plt
from pcap_manager import pcap_manager
import pandas as pd
import numpy as np
from tabulate import tabulate
import sys
import pickle

plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12

data = {}
train_dir = "/home/det_tesi/sgarofalo/GridSearchGNB/train_dataset"
test_dir = "/home/det_tesi/sgarofalo/GridSearchGNB/test_dataset"
save_dir = "/home/det_tesi/sgarofalo/GridSearchGNB"
sys.stdout = open(save_dir + "/result.txt", "w+")
for seconds in range(1, 6):
    seconds_samples = str(seconds) + "s"
    save_dir_tmp = save_dir + "/" + seconds_samples
    if not os.path.exists(save_dir_tmp): os.makedirs(save_dir_tmp)
    pm = pcap_manager(seconds_samples)
    print("Building datasets with seconds_samples = " + seconds_samples)
    pm.merge_pcap(train_dir)
    pm.merge_pcap(test_dir)
    data[seconds] = {}
    for n_PCA in range(2, 15):
        print("Classifing with n_PCA = " + str(n_PCA))
        classifiers(data[seconds], n_PCA, save_dir_tmp)
        print()

        
#Save data dict
with open('./GridSearchGNB/data.pkl', 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

header = list(range(2, 15))
rows = []
for seconds in range(1, 6):
    new_row = [str(seconds) + "s"]
    for n_PCA in range(2, 15):
        new_row.append(data[seconds][n_PCA])
    rows.append(new_row)

print(tabulate(rows, headers=header, tablefmt='orgtbl'))


'''
columns = ["SVM", "RF", "KNN", "GNB"]
df = pd.DataFrame(columns=columns)
for i in data:
    df = df.append({"SVM": data[i]["SVM"], "RF": data[i]["RF"], 
                    "KNN": data[i]["KNN"], "GNB": data[i]["GNB"]}, ignore_index=True)
df.index += 1
plt.figure(figsize=(20, 16))
plt.plot(df)
plt.xlabel("Seconds")
plt.ylabel("Accuracy")
plt.legend(columns, fontsize = 16)
plt.grid()
plt.tight_layout()
t = "Window size analysis"
plt.title(t, fontsize=16)
save_photo(save_dir, t)
'''