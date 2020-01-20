# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:44:30 2019

@author: Gianl
"""


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


matplotlib.rcParams.update({'font.size': 24})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12

def normalize (df , m = None , s = None):
    if (m is None and s is None):
        return (df-df.mean())/df.std() , df.mean(),df.std()
    else:
        return (df-m)/s

#%%
df = pd.read_csv(r'C:\Users\Gianl\Desktop\Captures_no_csv\dataset.csv', header=[0],)
y = df.iloc[:, -1:]
df = df.drop(["Unnamed: 0","timestamps","label"] , axis = 1)
X = df.dropna()
#%%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train,m_train,s_train = normalize(X_train)
X_test = normalize(X_test, m_train,s_train)

#%% heat map

matplotlib.rcParams.update({'font.size': 20})
plt.figure(0)
corr = X.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
mask[np.where (np.diag(mask) ),np.where (np.diag(mask) )] = False

cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap='magma', center=0, vmax=1, square=True, linewidths=1, cbar_kws={"shrink": 0.5}, xticklabels = range(1,20,1), \
            yticklabels = X_train.columns, annot = True)
#plt.xlabel("Features")
#plt.ylabel("Features")
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.title("Heatmap correlation matrix")
plt.show() # ta-da!

#%% PCA variance

pca2 = PCA()
pca2.fit(X_train)
plt.figure(1)
plt.plot (pca2.explained_variance_ratio_, color = "#0092CB" , linewidth = 6, marker = 'o')
#plt.annotate(label, (x,y),textcoords="offset points", xytext=(0,10), ha='center')
plt.grid()
plt.title("Variance Components")
plt.xlabel("Principal Component")
plt.ylabel("Variance")
plt.show()

#%%


pca = PCA(n_components=2)
#pca.fit(X)
X_train_new = pca.fit_transform(X_train)
X_test_new = pca.transform(X_test)
X_new_df =pd.DataFrame(data = X_train_new, columns = ["x","y"])
X_new_df['color']= np.where( y_train==1 , "#D1745C", "#EEBC98")
plt.figure(2)
sns.regplot(data=X_new_df, x="x", y="y", fit_reg=False, scatter_kws={'facecolors': X_new_df['color'], 'edgecolors': X_new_df['color']} )
kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train_new)
result = kmeans.labels_
centroid = pd.DataFrame( data=kmeans.cluster_centers_, columns = ["x", "y"])
sns.regplot(data=centroid, x="x", y="y", fit_reg=False, scatter_kws={'facecolors': "#000000", 'edgecolors': "#000000", 's':100}, label = "centroid")
clf = LinearSVC(penalty='l2', random_state=0, tol=1e-5, dual = False)
clf.fit(X_train_new, y_train)#result)
print ("Accuracy SVM PCA 2 componenti su test: %s"% clf.score(X_test_new,y_test))
# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 26)
yy = a * xx - (clf.intercept_[0]) / w[1]

margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin

plt.figure(2)
plt.plot(xx, yy,  linestyle = '-', color = '#012535', label = "svm" , linewidth = 4)
plt.plot(xx, yy_down,  linestyle = '--', color = '#760719', label = 'down_margin',linewidth = 2)
plt.plot(xx, yy_up,  linestyle = '--', color = '#FF7D45', label = 'up_margin',linewidth = 2)
plt.legend()
plt.grid()
plt.title("K-means SVM and PCA 2 components")


#%% PLOT CDF FEATURES


num = 4
for i in df:
    fig = plt.figure(num)
    plt.title(i)
    plt.grid()
    plt.ylabel("Density Function")
    sns.distplot(df[i],  color = 'darkblue', \
              hist_kws={'edgecolor':'black'},
              kde = True,
              norm_hist = True,
              )
    plt.savefig('temp.png', dpi=fig.dpi)
    num += 1


for i in df:
    plt.figure(num)
    plt.grid()
    plt.ylabel("Cumulative Function")
    plt.title(i)
    sns.distplot(df[i], \
                  hist_kws=dict(cumulative=True),
                  kde_kws=dict(cumulative=True))
    num +=1

#%% Tree Classifier


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


X_train = X_train.drop(["len_udp_p75","len_udp_mean","len_udp_p25","len_udp_p50","len_udp_std"] , axis = 1)
X_test = X_test.drop(["len_udp_p75","len_udp_mean","len_udp_p25","len_udp_p50","len_udp_std"], axis = 1)


X_train = X_train.drop(["rtp_inter_timestamp_std","rtp_inter_timestamp_mean", "rtp_inter_timestamp_num_zeros"] , axis = 1)
X_test = X_test.drop(["rtp_inter_timestamp_std","rtp_inter_timestamp_mean", "rtp_inter_timestamp_num_zeros"], axis = 1)

X_train = X_train.drop(["kbps"] , axis = 1)
X_test = X_test.drop(["kbps"], axis = 1)

clf_tree = DecisionTreeClassifier(criterion = 'entropy', random_state=0, max_depth = 5)
clf_tree.fit(X_train, y_train)
print("Accuratezza Decision Tree: %s" % clf_tree.score(X_test, y_test))

#names = list(df.columns.values)
names = list(X_train.columns.values)

export_graphviz(clf_tree, out_file=r"C:\Users\Gianl\Desktop\New folder\tree_no_udp_rtp_kbps.dot",  \
                filled=True, rounded=True,
                special_characters=True,
                feature_names=names)  

plt.figure(4)
plt.title('Feature Importances Tree')
plt.barh(X_train.columns, clf_tree.feature_importances_, color='#0092CB', align='center')
#name_split = [ name[i][0] +" \n " + name[i][1] for i in range(0,len(name))]
plt.xlabel('Relative Importance')
plt.grid()
plt.show()
#cross_val_score(clf, X_train, y_train, cv=10)

#%% Train Phase Random Forest

clf = RandomForestClassifier(n_estimators=11, random_state=0, criterion = 'entropy')
#X_train_no_udp = X_train.drop (["len_udp_p25","len_udp_p50","len_udp_p75","len_udp_std","len_udp_mean","len_udp_kbps"], axis = 1)
clf.fit(X_train, y_train)  

#%% Plot Phase Random Forest

plt.figure(3)
plt.title('Feature Importances')
plt.barh(X_train.columns, clf.feature_importances_, color='#0092CB', align='center')
#name_split = [ name[i][0] +" \n " + name[i][1] for i in range(0,len(name))]
plt.xlabel('Relative Importance')
plt.grid()
plt.show()

#%% Testing phase Random Forest

#X_test_no_udp = X_test.drop (["len_udp_p25","len_udp_p50","len_udp_p75","len_udp_std","len_udp_mean","len_udp_kbps"], axis = 1)
print("Accuratezza Random Forest: %s" % clf.score(X_test, y_test))

