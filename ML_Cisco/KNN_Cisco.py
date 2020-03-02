import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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


result = []
from tqdm import tqdm
for i in tqdm(range(3,10)):
    neigh = KNeighborsClassifier(n_neighbors=i, n_jobs = 4, p = 1)
    neigh.fit(X_train, y_train["label"])
    print("Neighbors: {}, Norm: True 1-0, p = 1, Train: Full, Test: Full, Feature: All, Score: {}\n"\
            .format(i,neigh.score(X_test,y_test["label"])))
    result.append(neigh.score(X_test,y_test["label"]))

y_predict = neigh.predict(X_test)
y_predict_rescale = [ i-1 if (i > 0) else i for i in y_predict] # class 2 diventa 1, 3-2 etcc
confusion_matrix_Cisco(y_test["label"], y_predict)
Report_Matrix(y_test["label"], y_predict)
