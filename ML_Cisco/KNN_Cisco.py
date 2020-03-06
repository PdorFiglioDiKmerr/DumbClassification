import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
from BuildDataset import _Dataset
from sklearn import preprocessing
import matplotlib
from ConfusionMatrixCisco import confusion_matrix_Cisco, Report_Matrix
import pickle


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

def generate_clf_from_search(grid_or_random, clf, parameters, scorer, X, y):
    if grid_or_random == "Grid":
        search_obj = GridSearchCV(clf, parameters, scoring=scorer, n_jobs = 47, random_state = 42)
    elif grid_or_random == "Random":
        search_obj = RandomizedSearchCV(clf, parameters, scoring=scorer, n_iter = 3000, random_state = 42, n_jobs = 47)
    fit_obj = search_obj.fit(X, y)
    best_clf = fit_obj.best_estimator_
    return best_clf

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

X_train, y_train, X_test, y_test = _Dataset()
col_test = new_name_col
col_train = new_name_col
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns = new_name_col)
X_test =  pd.DataFrame(X_test, columns = new_name_col)
X_train_rid = X_train[best_features]


param_grid = {n_neighbors : [i for i in range(3,41)],
              p : [1, 2]
              }
clf = KNeighborsClassifier()
best_clf = generate_clf_from_search("Grid", clf, param_grid, None,X_train, y_train["label"])
filename = "k_NN_c.pickle"
with open(filename, "wb") as f:
    pickle.dump(best_clf, f)


#DATASET STD RIDOTTO
param_grid = {n_neighbors : [i for i in range(3,41)],
              p : [1, 2]
              }
clf = KNeighborsClassifier()
best_clf = generate_clf_from_search("Grid", clf, param_grid, None,X_train_rid, y_train["label"])
filename = "k_NN_rid.pickle"
with open(filename, "wb") as f:
    pickle.dump(best_clf, f)



# confusion_matrix_Cisco(y_test["label"], y_predict)
# Report_Matrix(y_test["label"], y_predict)
