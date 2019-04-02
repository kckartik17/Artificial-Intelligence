#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:37:45 2019

@author: kartikchauhan
"""

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1] 
y = dataset.iloc[:, -1] 


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)


pred = np.array([2020])
pred = pred.reshape(-1,1)
y_pred = regressor.predict(pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=regressor,X=X_train,y=y_train)
accuracies.mean()
accuracies.std()

from sklearn.metrics import mean_squared_error

# The mean squared error
print("Mean squared error: {}".format(mean_squared_error(y_test, y_pred)))

