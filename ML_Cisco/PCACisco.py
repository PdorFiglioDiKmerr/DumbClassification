# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 13:25:22 2020

@author: Gianl
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib

def PCA_Plot_Variance(X):

    pca2 = PCA()
    pca2.fit(X)
    plt.figure()
    plt.plot (pca2.explained_variance_ratio_, color = "#0092CB" , linewidth = 6, marker = 'o')
    plt.grid()
    plt.title("Variance Components")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance")
    plt.show()

def colors_for_pca(n_class):
    import matplotlib.patches as mpatches
    colors_list = ['#D1745C', '#EEBC98', '#9F28C1', '#28C186', '#282BC1',\
              '#00FFFF', '#A9A9A9', "#C756A9"]
    zero_patch = mpatches.Patch(color='#D1745C', label='0-Audio')
    one_patch = mpatches.Patch(color='#EEBC98', label='1-Video')
    two_patch = mpatches.Patch(color='#9F28C1', label='2-FEC Video')
    three_patch = mpatches.Patch(color='#28C186', label='3 Screen Sharing')
    four_patch = mpatches.Patch(color='#282BC1', label='4-FEC Audio')
    five_patch = mpatches.Patch(color='#00FFFF', label='5-VideoHQ')
    six_patch = mpatches.Patch(color='#A9A9A9', label='6-VideoLQ')
    seven_patch = mpatches.Patch(color="#C756A9", label='7-VideoMQ')
    patch_list = [zero_patch, one_patch, two_patch, three_patch, four_patch, five_patch, six_patch, seven_patch]
    patch_list_final = []
    colors_final = []
    for i in n_class:
        patch_list_final.append(patch_list[int(i)])
        colors_final.append(colors_list[int(i)])
    return patch_list_final,colors_final

def PCA_Plot_3D(X_train, y_train, pca = None):

    if pca is None:
        pca = PCA(n_components=3)
        X_train_new = pca.fit_transform(X_train)
    else:
        X_train_new = pca.transform(X_train)
    X_new_df =pd.DataFrame(data = X_train_new, columns = ["PCA0","PCA1","PCA2"])
    #Plot initialisation
    fig = plt.figure()
    patch_list, color_labels = colors_for_pca(list(set(y_train)))
    ax1 = fig.add_subplot(111,projection='3d')
    ax1.scatter(X_new_df['PCA0'], X_new_df['PCA1'], X_new_df['PCA2'], c = y_train, s=60, \
                cmap=matplotlib.colors.ListedColormap(color_labels))
    ax1.legend(handles=patch_list)
    # make simple, bare axis lines through space:
    xAxisLine = ((min(X_new_df['PCA0']), max(X_new_df['PCA0'])), (0, 0), (0,0))
    ax1.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
    yAxisLine = ((0, 0), (min(X_new_df['PCA1']), max(X_new_df['PCA1'])), (0,0))
    ax1.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
    zAxisLine = ((0, 0), (0,0), (min(X_new_df['PCA2']), max(X_new_df['PCA2'])))
    ax1.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')
    # label the axes
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    ax1.set_title("3D-PCA Plot")
    plt.show()
    return X_new_df, pca
