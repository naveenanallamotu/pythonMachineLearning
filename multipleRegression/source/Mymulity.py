# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 16:42:45 2017

@author: NAVEENA
"""

#multple regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,0:4].values
Y= dataset.iloc[:,4].values
 #need to encode the data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder =OneHotEncoder(categorical_features = [3] )
X =onehotencoder.fit_transform(X).toarray()
#avoiding the dummy variable trap by excluding the one dummy variable
X = X[:,1:]


from sklearn.cross_validation import train_test_split
X_train, X_test = train_test_split(X, test_size = 0.2 , random_state=0)
Y_train, Y_test = train_test_split(Y, test_size = 0.2, random_state=0)
# fit the data to multiple regression
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
Regressor.fit(X_train,Y_train)
#predecting the Y_test results
Y_pred = Regressor.predict(X_test)
#backward elimination together
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis =1 )
X_OPT = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_OPT ).fit()
regressor_OLS.summary()
X_OPT = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_OPT ).fit()
regressor_OLS.summary()
X_OPT = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_OPT ).fit()
regressor_OLS.summary()
X_OPT = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_OPT ).fit()
regressor_OLS.summary()
X_OPT = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_OPT ).fit()
regressor_OLS.summary()