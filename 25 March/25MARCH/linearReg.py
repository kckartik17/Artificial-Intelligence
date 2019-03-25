
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1] 
y = dataset.iloc[:, -1] 

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X.iloc[:, 1:])
X.iloc[:, 1:] = imputer.transform(X.iloc[:, 1:])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.iloc[:, 0] = labelencoder_X.fit_transform(X.iloc[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3.0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:, 3:] = sc_X.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc_X.transform(X_test[:, 3:])


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


import seaborn as sns
sns.set(color_codes=True)

dataframe_training = pd.DataFrame()
dataframe_training['YearsExperience'] = X_train['YearsExperience']
dataframe_training['Salary'] = y_train
ax = sns.regplot(x="YearsExperience", y="Salary", data= dataframe_training)

dataframe_test = pd.DataFrame()
dataframe_test['YearsExperience'] = X_test['YearsExperience']
dataframe_test['Salary'] = y_test
ax = sns.regplot(x="YearsExperience", y="Salary", data= dataframe_training)

print('Coefficients: \n', regressor.coef_)

from sklearn.metrics import mean_squared_error, r2_score
print("Mean squared error: {}".format(mean_squared_error(y_test, y_pred)))
print("Variance score: {}".format(r2_score(y_test, y_pred)))


import statsmodels.api as sm

x = sm.add_constant(X)
results = sm.OLS(endog = y, exog=x).fit()
results.summary()