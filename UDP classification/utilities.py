import os
import sys
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

def save_photo(save_dir, t, sub_dir=None):
    separator = '/' if sys.platform.startswith("linux") else r'\\'
    save_dir += separator + 'plots' + separator
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    if sub_dir != None: sub_dir = save_dir + sub_dir + separator
    dpi = 100
    if sub_dir == None:
        plt.savefig(save_dir + t +'.png', dpi = dpi)
    else:
        if not os.path.exists(sub_dir): os.makedirs(sub_dir)
        plt.savefig(sub_dir + t +'.png', dpi = dpi)      
    plt.close()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    
    
    plt.figure(figsize=(8, 6), dpi=80)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="cornflowerblue")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
             label="Cross-validation score")

    plt.legend(loc="best")


def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x, y)