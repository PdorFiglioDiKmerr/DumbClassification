# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 13:21:21 2020

@author: Gianl
"""
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def confusion_matrix_Cisco(y_test, y_predict, title ='Confusion Matrix' , save = False):
    labelll = ["audio", "Fec_V", "Screen", "Fec_A"\
              , "720p", "180p", "360p"]
    cm_ = confusion_matrix(y_test,y_predict)#, labels = range(0,7,1))
    cm_
    plt.figure(figsize = (16,16))
    cm_df = pd.DataFrame(cm_, columns = labelll, index = labelll)
    cm_df["All"] = cm_df.sum( axis = 1)
    plt.yticks(np.arange(8)+0.5,labelll, va="center")
    sns.heatmap(cm_df, annot = True, cbar=False, cmap = 'Blues', fmt = 'd').set_ylim(len(cm_), -0.5)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    if save:
        plt.savefig("cm_"+title+".png", dpi = 300)
    else:
        plt.show()
  
def Report_Matrix(y_test, y_class_predict, title ='Report Matrix' , save = False):
    from sklearn.metrics import precision_recall_fscore_support
    to =precision_recall_fscore_support(y_test, y_class_predict)
    label_ = ["audio", "Fec_V", "Screen", "Fec_A"\
                  , "720p", "180p", "360p"]
    to_df = pd.DataFrame(index = label_)
    to_df["Precision"] = to[0]
    to_df["Recall"] = to[1]
    to_df["F1"] = to[2]
    to_df["Support"] = to[3]
    plt.figure(figsize = (16,16))
    to_df = pd.DataFrame(to_df, index = label_)
    plt.yticks(np.arange(8)+0.5,label_, va="center")
    sns.heatmap(to_df, annot = True, cbar=False, cmap = 'Blues',fmt = 'g').set_ylim(len(label_), -0.5)
    plt.ylabel('Class')
    plt.xlabel('Score')
    plt.title(title)
    if save:
        plt.savefig("rep_"+title+".png", dpi = 300)
    else:
        plt.show()