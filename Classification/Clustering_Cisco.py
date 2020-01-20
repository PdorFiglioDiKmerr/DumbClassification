# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:56:19 2019

@author: Gianl
"""

#%%
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.cm as cm
import matplotlib.font_manager

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

def normalize (df , m = None , s = None):
    if (m is None and s is None):
        return (df-df.mean())/df.std() , df.mean(),df.std()
    else:
        return (df-m)/s

def build_dataset(df):
    
    y_o = df.iloc[:,-2:]

    try:
        return df.drop(["Unnamed: 0","timestamps","label", "label2"] , axis = 1) , y_o
    except:
        return df.drop(["timestamps","label", "label2"] , axis = 1),  y_o
  
def heatmap_my (X):

    plt.figure()
    corr = X.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    mask[np.where (np.diag(mask) ),np.where (np.diag(mask) )] = False
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap='magma', center=0, vmax=1, square=False, linewidths=1, cbar_kws={"shrink": 0.5}, xticklabels =  X_train.columns, \
                yticklabels = X_train.columns, annot = True, fmt = '.0%')
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.title("Heatmap correlation matrix")
    plt.show() # ta-da!

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

def PCA_Plot_3D(X, y_train):
    
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches as mpatches
    pca = PCA(n_components=3)
    #pca.fit(X)
    X_train_new = pca.fit_transform(X_train)
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    #ax2 = fig.add_subplot( 1,2,2, projection='3d')
    # X_test_new = pca.transform(X_test)
    num = 1
    for lab in y_train:
        ax1 = fig.add_subplot(1,2,num, projection = '3d')
        X_new_df =pd.DataFrame(data = X_train_new, columns = ["PCA0","PCA1","PCA2"])
        X_new_df.loc[y_train[lab] == 0, 'color'] = "#D1745C"
        X_new_df.loc[y_train[lab] == 1, 'color'] = "#EEBC98"
        X_new_df.loc[y_train[lab] == 2, 'color'] = "#9F28C1"
        X_new_df.loc[y_train[lab] == 3, 'color'] = "#28C186"
        X_new_df.loc[y_train[lab] == 4, 'color'] = "#282BC1"
        X_new_df.loc[y_train[lab] == 5, 'color'] = "#00FFFF"
        X_new_df.loc[y_train[lab] == 6, 'color'] = "#A9A9A9"
        #Plot initialisation
        labell = ("0-Audio", "1-Video", "2-FEC Video", "3 Screen Sharing", "4-FEC Audio", "5-VideoHQ", "6-VideoLQ" )
       # fig = plt.figure()
        #ax = fig.add_subplot( 111,projection='3d')
        zero_patch = mpatches.Patch(color='#D1745C', label='0-Audio')
        one_patch = mpatches.Patch(color='#EEBC98', label='1-Video')     
        two_patch = mpatches.Patch(color='#9F28C1', label='2-FEC Video')                     
        three_patch = mpatches.Patch(color='#28C186', label='3 Screen Sharing')                     
        four_patch = mpatches.Patch(color='#282BC1', label='4-FEC Audio')                     
        five_patch = mpatches.Patch(color='#00FFFF', label='5-VideoHQ')                     
        six_patch = mpatches.Patch(color='#A9A9A9', label='6-VideoLQ')                     
        ax1.scatter(X_new_df['PCA0'], X_new_df['PCA1'], X_new_df['PCA2'], color = X_new_df["color"], s=60)
        ax1.legend(handles=[zero_patch, one_patch, two_patch, three_patch, four_patch, five_patch, six_patch])
    
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
        ax1.set_title("3D-PCA Plot "+ lab)
        num +=1
    plt.show()
    return X_train_new
    # kmeans = KMeans(n_clusters=4, random_state=0).fit(X_train_new)
    # result = kmeans.labels_
    # centroid = pd.DataFrame( data=kmeans.cluster_centers_, columns = ["x", "y","z"])
    # ax.scatter(centroid["x"], centroid["y"], centroid["z"], color="black", label = "centroid", s = 60)

def silhoutte_analysis(X_train, range_n_clusters):
    silhoutte_list = []
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig = plt.figure(figsize=plt.figaspect(0.5))
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1 = fig.add_subplot(1,2,1)
        #fig.set_size_inches(18, 7)
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X_train) + (n_clusters + 1) * 10])
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X_train)
        print("Rand Score vs Label2: %f" % adjusted_rand_score(y_train["label2"],cluster_labels))
        print("Rand Score vs Label: %f" % adjusted_rand_score(y_train["label"],cluster_labels))

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X_train, cluster_labels)
        silhoutte_list.append(silhouette_avg)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
    
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X_train, cluster_labels)
    
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
         # Labeling the clusters
    
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
            
            
    plt.figure()
    plt.plot (range_n_clusters, silhoutte_list, color = "#0092CB" , linewidth = 6, marker = 'o', markersize = 8)
    plt.grid()
    plt.title("Average Silhouette Score vs # Cluster")
    plt.xlabel("# Cluster")
    plt.ylabel("Average Silhouette Score")
    plt.show()
#%% PREPARE TRAIN
        
df = pd.read_csv(r'C:\Users\Gianl\Desktop\dataset.csv', header=[0],)
df, y_train = build_dataset(df)
X_train = df.dropna()

only_audio_cisco = df[y_train["label2"] == 0]
only_audio_my = df[y_train["label"] == 0]
X_train,m_train,s_train = normalize(X_train)
#X_train = X_train.drop(["interarrival_p25", "interarrival_p50", "len_udp_p25", "len_udp_p50",\
#                       "interlength_udp_p25","interlength_udp_p50", "interarrival_p75", "rtp_inter_timestamp_mean","rtp_inter_timestamp_num_zeros", "kbps",
#                      "len_udp_p75", "inter_time_sequence_p25", "inter_time_sequence_p50", "inter_time_sequence_p75"] , axis = 1)
heatmap_my(X_train)

#%% PCA Part and K-means

PCA_Plot_Variance(X_train)
X_PCA = PCA_Plot_3D(X_train, y_train)

X_train = X_PCA

range_n_clusters = [6]
silhoutte_analysis(X_train,range_n_clusters)
silhoutte_list = []
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig = plt.figure(figsize=plt.figaspect(0.5))
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot( 1,2,2, projection='3d')
    #fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X_train) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X_train)
    
    print(adjusted_rand_score(y_train["label2"],cluster_labels))
    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X_train, cluster_labels)
    silhoutte_list.append(silhouette_avg)

    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X_train, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
      # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    #ax2.scatter(centers[:, 0], centers[:, 1], centers[:,2], marker='o')
    # ax2.scatter(centers[:, 0], centers[:, 1], centers[:,2], marker='o', \
    #             c="white", alpha=1, s=200, edgecolor='k')
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], c[2], marker='o', \
                  c="white", alpha=1, s=200, edgecolor='k')
        ax2.scatter(c[0], c[1],c[2], marker='$%d$' % i, alpha=1, \
                    s=150, edgecolor='k')
        
    ax2.scatter(X_train[:, 0], X_train[:, 1], X_train[:,2], marker='.', s=100, lw=0, alpha=0.5,
                c=colors, edgecolor='k')
     
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                  fontsize=14, fontweight='bold')

#     plt.show()
    
# plt.figure()
# plt.plot (range_n_clusters, silhoutte_list, color = "#0092CB" , linewidth = 6, marker = 'o', markersize = 8)
# plt.grid()
# plt.title("Average Silhouette Score vs # Cluster")
# plt.xlabel("# Cluster")
# plt.ylabel("Average Silhouette Score")
# plt.show()
# #%% KmeanShift
    


# clustering = MeanShift()
# etichette = clustering.fit_predict(X_train)
# print("MeanShift: Rand Score vs Label2: %f" % adjusted_rand_score(y_train["label2"],etichette))
# print("MeanShift: Rand Score vs Label: %f" % adjusted_rand_score(y_train["label"],etichette))
# # #%% DBSCAN

# for i in np.arange(0.5,8,0.1):
    
#     clustering_DB = DBSCAN(eps=i, min_samples=80)
#     etichette_DB = clustering_DB.fit_predict(X_train)
#     print ("Eps = %f" % i)
#     print("DB-scan: Rand Score vs Label2: %f" % adjusted_rand_score(y_train["label2"],etichette_DB))
#     print("DB-scan: Rand Score vs Label: %f" % adjusted_rand_score(y_train["label"],etichette_DB))


# pca = PCA(n_components=5)
#     #pca.fit(X)
# X_train = pca.fit_transform(X_train)
# range_n_clusters = [2,3,4,5,6,7,8,9,10]


# #
# df2 = df.copy()
# df2 = df2.drop(["Unnamed: 0","timestamps","interarrival_p25", "interarrival_p50", "len_udp_p25", "len_udp_p50",\
#                         "interlength_udp_p25","interlength_udp_p50", "interarrival_p75", "rtp_inter_timestamp_mean","rtp_inter_timestamp_num_zeros", "kbps",
#                         "len_udp_p75","label", "label2"] , axis = 1)
# df2 = df2.dropna()
# cols = df2.columns


# def hide_current_axis(*args, **kwds):
#     plt.gca().set_visible(False)
# pp = sns.pairplot(df2[cols],size=1.8, aspect=1.8,
#                   plot_kws=dict(edgecolor="k", linewidth=0.5),
#                   diag_kind="kde", diag_kws=dict(shade=True))
# pp.map_upper(hide_current_axis)
# # fig = pp.fig 
# # fig.subplots_adjust(top=0.93, wspace=0.3)
# # t = fig.suptitle('Attributes Pairwise Plots', fontsize=14)

# silhoutte_analysis(X_train,range_n_clusters)
    
    
