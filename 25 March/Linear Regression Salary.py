# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1] #Take all the columns except last one
y = dataset.iloc[:, -1] #Take the last column as the result

# Taking care of missing data
'''from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X.iloc[:, 1:])
X.iloc[:, 1:] = imputer.transform(X.iloc[:, 1:])'''

'''# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.iloc[:, 0] = labelencoder_X.fit_transform(X.iloc[:, 0])

#Make dummy variables
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3.0)

# Library will do this automatically
'''# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:, 3:] = sc_X.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc_X.transform(X_test[:, 3:])

#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train.reshape(-1, 1))'''

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


# Visualising the Training set results
import seaborn as sns
sns.set(color_codes=True)

dataframe_training = pd.DataFrame()
dataframe_training['YearsExperience'] = X_train['YearsExperience']
dataframe_training['Salary'] = y_train
ax = sns.regplot(x="YearsExperience", y="Salary", data= dataframe_training)

# Visualising the Test set results
dataframe_test = pd.DataFrame()
dataframe_test['YearsExperience'] = X_test['YearsExperience']
dataframe_test['Salary'] = y_test
ax = sns.regplot(x="YearsExperience", y="Salary", data= dataframe_training)

# The coefficients
print('Coefficients: \n', regressor.coef_)

from sklearn.metrics import mean_squared_error, r2_score
# The mean squared error
print("Mean squared error: {}".format(mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print("Variance score: {}".format(r2_score(y_test, y_pred)))


import statsmodels.api as sm

# Add a constant. Essentially, we are adding a new column (equal in lenght to x), which consists only of 1s
x = sm.add_constant(X)
# Fit the model, according to the OLS (ordinary least squares) method with a dependent variable y and an idependent x
results = sm.OLS(endog = y, exog=x).fit()
# Print a nice summary of the regression. That's one of the strong points of statsmodels -> the summaries
results.summary()