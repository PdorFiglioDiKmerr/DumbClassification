# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 13:28:08 2020

@author: Gianl
"""

import keras
import matplotlib.pyplot as plt
from ConfusionMatrixCisco import confusion_matrix_Cisco
from sklearn.preprocessing import label_binarize
from BuildDataset import _Dataset
from sklearn import preprocessing
import pandas as pd
import numpy as np
from ConfusionMatrixCisco import confusion_matrix_Cisco, Report_Matrix


def neural_network_cisco(X, y, X_test, y_test):

    from keras.models import Sequential
    from keras.layers import Dense, Dropout
   # from sklearn.preprocessing import LabelEncoder

    y_test_rescale = [ int(i)-1 if (int(i) > 0) else int(i) for i in y_test] # class 2 diventa 1, 3-2 etcc
    y_train_rescale = [ int(i)-1 if (int(i) > 0) else int(i) for i in y]
    dummy_y_train = label_binarize(y_train_rescale, classes = [0,1,2,3,4,5,6])
    dummy_y_test = label_binarize(y_test_rescale, classes = [0,1,2,3,4,5,6])
    # Build the model.
    model = Sequential([
      Dense(27, activation='sigmoid', kernel_initializer='random_normal',input_dim=29),
      Dropout(0.2),
      Dense(20, activation='sigmoid', kernel_initializer='random_normal',input_dim=29),
      Dropout(0.15),
      Dense(7,  activation='softmax', kernel_initializer='random_normal'),
    ])
    # Compile the model.
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    # Train the model.
    model.fit(X, dummy_y_train, epochs=40, batch_size = 100)
    print(model.evaluate(X, dummy_y_train))
    #Test
    print("Test: {}".format(model.evaluate(X_test, dummy_y_test)))
    y_predict = model.predict(X_test)
    y_pred = y_predict.argmax(axis=-1)
    confusion_matrix_Cisco(y_test_rescale, y_pred)
    return y_pred


X_train, y_train, X_test, y_test = _Dataset()
col_test = X_test.columns
col_train = X_train.columns
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns = col_train)
X_test = pd.DataFrame(X_test, columns = col_test)
y_test_rescale = [ i-1 if (i > 0) else i for i in y_test["label"]] # class 2 diventa 1, 3-2 etcc#confusion_matrix_Cisco(y_test["label"], y_class_predict)

y_pred = neural_network_cisco(X_train, y_train["label"], X_test, y_test["label"])
confusion_matrix_Cisco(y_test_rescale, y_pred)
Report_Matrix(y_test_rescale, y_pred)