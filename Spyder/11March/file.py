# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer = imputer.fit(X.iloc[:,1:])
X.iloc[:,1:] = imputer.transform(X.iloc[:,1:])

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X.iloc[:,0] = labelencoder_X.fit_transform(X.iloc[:,0])

onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:,3:] = sc_X.fit_transform(X_train[:,3:])
X_test[:,3:] = sc_X.transform(X_test[:,3:])