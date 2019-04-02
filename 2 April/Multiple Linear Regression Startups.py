# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import scipy.stats as stats
import statsmodels.api as sm

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1] #Take all the columns except last one
y = dataset.iloc[:, -1] #Take the last column as the result


sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')



'''No multicolinearity - also check for condition number
We observe it when two or more variables have a high coorelation.
If a can be represented using b, there is no pint using both
c and d have a correlation of 90% (imprefect multicolinearity). if c can be almost
represented using d there is no point using both
FIX : a) Drop one of the two variables. b) Transform them into one variable by taking
mean. c) Keep them both but use caution. 
Test : before creating the model find correlation between each pairs.
'''
dataset.corr()






'''# Taking care of missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X.iloc[:, 1:])
X.iloc[:, 1:] = imputer.transform(X.iloc[:, 1:])'''






# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.iloc[:, 3] = labelencoder_X.fit_transform(X.iloc[:, 3])

#Make dummy variables
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]







'''# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)'''






# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)






'''# Library will do this automatically
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:, 3:] = sc_X.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc_X.transform(X_test[:, 3:])

#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train.reshape(-1, 1))'''






# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


# K-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=regressor, X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()

''' Check for linearity assumption again (dataset should be large)
This assumption implies that there should be a linear relationship between the
response variable and the predictors.
The data can be explained by the linear equation y = c + m1x1 + m2x2 + ... + mkxk 

TEST : Checking the linearity assumption may require plotting of predictor 
versus response variables. This should be fairly easy for simple linear
regression but in multiple linear regression with large number of predictor
variables, we can use standardised residual plot against each one of the
predictor variables. The ideal plot of residuals with each of the predictor
should be a random scatter because we assume that the residuals are 
uncorrelated with the predictor variables. Any noticeable pattern in such
plots indicates violation of linear relationship assumption. Use R2 to check
the strength of correlations.

Using a pairsplot check that no two scatterplots look like they are folloing a 
curve and need a polynomial regression

FIX : use a polynomial regression, or use logs/exponential to straighten the data 
'''

def r_squared(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r^2 = {:.3f}".format(r**2),
                xy=(.1, .9), xycoords=ax.transAxes)

# Check for Linearity visually
linearity_assumption_plot_1 = sns.pairplot(pd.DataFrame(X), kind="reg")
linearity_assumption_plot_1.map_lower(r_squared)

error_residual = pd.DataFrame(y_test-y_pred)
error_residual.reset_index(inplace = True)
linearity_test_df = pd.DataFrame(X_test)
linearity_test_df['Residual'] = error_residual['Profit']
linearity_test_df.columns = 'S1 S2 RnD Admin Marketing Residuals'.split()

linearity_assumption_plot_2 = sns.pairplot(linearity_test_df.iloc[:, 2:], kind="reg")
linearity_assumption_plot_2.map_lower(r_squared)






''' Check for No Endogenity of regressors : Residuals should not be highly
coorelated with other predictors

The errors (difference between the observed and predicted values) is coorelated 
with our independent values

This is a problem referred to as omitted variable bias - when we forget to
include a relevant variable
 
y is explained (somewhat correlated by xs)
y is explained (somewhat correlated by omitted xs)
Chances are that the omitted variable is also coorelated with an independent x,
however we forgot to include that as a regressor
 
Everything that we dont explain with our model goes into the error - so the 
error becomes coorelated with everything else

TEST : Create a datafrme of difference of observed and predicted and other 
variables and check coorelation

FIX : hard to fix, think what you missed
Where did we get the sample from? Can we get a better sample? 
See if you can include back some variables you dropped due to low p-value.
Domain expertise will help.
'''
endogenity_check = linearity_test_df.corr() # Check only the reciduals row with other data




''' 4) Homoscedasticity
Assumes the error terms have equal variance. 
An example of a dataset where errors have a different variance looks like a 
cone around the regression line.This means that with smaller values we get a better
prediction than with larger values. 

FIX : Check for omitted vairable bias. Look for outliers and omit them. Use
natural log transformation. We can apply Ln on dependent and independent variables.
Ln(y) = b0 + b1 * Ln(x)

Levene tests the null hypothesis that all samples come from 
populations with equal variances. It returns the test statistic ('W') and the 
probabilyt ('p').

The variance criterion holds true when p > a (where a is the probability 
threshold usually set to 0.05)

Three variations of Levene’s test are possible. The possibilities and their 
recommended usages are:
‘median’ : Recommended for skewed (non-normal) distributions>
‘mean’ : Recommended for symmetric, moderate-tailed distributions.
‘trimmed’ : Recommended for heavy-tailed distributions.
'''
#stats.levene(error_residual['Profit'], X_test[:, 2], X_test[:, 3], X_test[:, 4])
#stats.levene(error_residual['Profit'], X_test[:, 0], X_test[:, 1], X_test[:, 2], X_test[:, 3], X_test[:, 4])

residual_test = np.column_stack([y_test,y_pred])
residual_test = pd.DataFrame(residual_test)
residual_test.columns='Y_test predictions'.split()
sns.jointplot(x='Y_test', y='predictions', data=residual_test, kind='reg')

stats.levene(residual_test['Y_test'], residual_test['predictions'])

import statsmodels.stats as st
# Small p-value (pval below) shows that there is violation of homoscedasticity.
#_, pval, __, f_pval = st.diagnostic.het_breuschpagan(error_residual['Profit'],  X_test[:, 2:])
_, pval, __, f_pval = st.diagnostic.het_breuschpagan(error_residual['Profit'],  X_test)


'''3) Normality
We assume error terms are normally distributed with mean of zero.
This is not requred to make the model but to make the inferences.
The pvalues and t values we get in the reports work because we assume normality
of the error terms.
 
FIX : Central Limit Theorm : For large samples, CLT applies to error terms too
So if smaple size is large, we can assume normality

If the error means are not expected to be zero the line is not the best fitting
line.
 
Fix : Adding an intercept solves this. In real life, its hard to voilate this.

Do a hypothesis test in which the null hypothesis is that the errors have a 
normal distribution. Failure to reject this null hypothesis is a good result. 
It means that it is reasonable to assume that the errors have a normal 
distribution.

p > 0.05 confirm to the normality criterion
'''
# Shapiro-Wilk normality test
stats.shapiro(error_residual['Profit'])







# The coefficients
print('Coefficients: \n', regressor.coef_)

from sklearn.metrics import mean_squared_error, r2_score

# The mean squared error
print("Mean squared error: {}".format(mean_squared_error(y_test, y_pred)))

# Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
print("Variance score: {}".format(r2_score(y_test, y_pred)))






# Building the optimal model using Backward Elimination
# Add a constant. Essentially, we are adding a new column (equal in lenght to x), which consists only of 1s
x = sm.add_constant(X)

x = pd.DataFrame(x, columns = 'Const S1 S2 R&D Administration Marketing'.split())

X_opt = x.loc[:, ['Const', 'S1', 'S2', 'R&D', 'Administration', 'Marketing']]

# Fit the model, according to the OLS (ordinary least squares)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

# Print a nice summary of the regression. That's one of the strong points of statsmodels -> the summaries
# Check for multicolinearity with large condition number

''' No Autocorrelation (No serial correlation) check for Durbin-Watson number
This cannot be relaxed.
Errors are assumed to be uncorrelated - randomly spread around the regression line
Its unlikey to find this in data taken in one moment of time (cross sectional data)
but its common to find this time series data
Test : Durbin-Watson - generally value falls between 0 and 4
2 -> No autocorrelation
<1 and >3 -> Cause for alarm
Fix : There is no fix! Dont use a linear regression. Use time series models
Autoregressive models
moving average models
autoregressive moving average model
autoregressive integrated moving average model
'''

regressor_OLS.summary()

X_opt = x.loc[:, ['Const', 'S2', 'R&D', 'Administration', 'Marketing']]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = x.loc[:, ['Const', 'R&D', 'Administration', 'Marketing']]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#This is the best - see adjusted R2
# Run all the tests on the final model to check if all auumptions hold true
X_opt = x.loc[:, ['Const', 'R&D', 'Marketing']]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = x.loc[:, ['Const', 'R&D']]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()