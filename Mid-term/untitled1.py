#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:44:43 2019

@author: kartikchauhan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('Dataset.csv')
X = dataset.iloc[:,-1]
y = dataset.iloc[:,-1]



from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy="most_frequent")
imputer = imputer.fit(X.iloc[:, 1:])
X.iloc[:, 1:] = imputer.transform(X.iloc[:, 1:])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.iloc[:,2] = labelencoder_X.fit_transform(X.iloc[:,2])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Make dummy variables
onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

array1 = np.array([X_train])
array1 = array1.reshape(-1,1)

array2 = np.array([y_train])
array2 = array2.reshape(-1,1)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=regressor, X=array1 ,y=array2, cv=7)
accuracies.mean()
accuracies.std()



    