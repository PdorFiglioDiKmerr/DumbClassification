# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 23:38:09 2020

@author: Gianl
"""
import numpy as np
import pandas as pd


def build_dataset(df):

    y_o = df.iloc[:,-2:]

    try:
        return df.drop(["label", "label2"] , axis = 1) , y_o
    except:
        return df.drop(["label", "label2"] , axis = 1),  y_o

def _Dataset():
    dir_Train = r".\Dataset\Train"
    dir_Test = r".\Dataset\Test"
    #TRAIN
    df_HQ = pd.read_csv (dir_Train + r"\Train_HQ\dataset_HQ.csv", index_col = [0])
    df_MQ = pd.read_csv (dir_Train + r"\Train_MQ\dataset_MQ.csv", index_col = [0])
    df_LQ = pd.read_csv (dir_Train + r"\Train_LQ\dataset_LQ.csv", index_col = [0])
    df_SS = pd.read_csv (dir_Train + r"\Train_SS\dataset_SS.csv", index_col = [0])
    #TEST
    df_HQ_test = pd.read_csv (dir_Test + r"\Test_HQ\dataset_HQ_test.csv", index_col = [0])
    df_MQ_test = pd.read_csv (dir_Test + r"\Test_MQ\dataset_MQ_test.csv", index_col = [0])
    df_LQ_test = pd.read_csv (dir_Test + r"\Test_LQ\dataset_LQ_test.csv", index_col = [0])
    df_SS_test = pd.read_csv (dir_Test + r"\Test_SS\dataset_SS_test.csv", index_col = False)
    df_AUD_test = pd.read_csv (dir_Test + r"\Test_AUD\dataset_AUD_test.csv", index_col = False)

    df_train = pd.concat([df_HQ, df_MQ, df_LQ, df_SS], sort = False)
    shuffled_df = df_train.sample(frac=1,random_state=4)
    # Put all the fraud class in a separate dataset.
    fraud_df = shuffled_df.loc[shuffled_df['label'] != 0]
    #Randomly select 492 observations from the non-fraud (majority class)
    non_fraud_df = shuffled_df.loc[shuffled_df['label'] == 0].sample(n=20000,random_state=42)

    # Concatenate both dataframes again
    normalized_df = pd.concat([fraud_df, non_fraud_df])
    normalized_df.reset_index(drop = True, inplace = True)
    df_train, y_train = build_dataset(normalized_df) #df_train
    X_train = df_train.dropna()

    df_test = pd.concat([df_HQ_test, df_MQ_test, df_LQ_test, df_SS_test], sort = False)
    
    shuffled_df = df_test.sample(frac=1,random_state=4)
    # Put all the fraud class in a separate dataset.
    fraud_df = shuffled_df.loc[shuffled_df['label'] != 0]
    fraud_df = fraud_df.loc[fraud_df['label'] != 4]
    #Randomly select 492 observations from the non-fraud (majority class)
    #non_fraud_df = shuffled_df.loc[shuffled_df['label'] == 0].sample(n=2000,random_state=42)
    # Concatenate both dataframes again
    normalized_df = pd.concat([fraud_df, df_AUD_test])
    normalized_df.reset_index(drop = True, inplace = True)
    df_test, y_test = build_dataset(normalized_df)
    X_test = df_test.dropna()

    return X_train, y_train, X_test, y_test
