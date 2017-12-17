# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:58:39 2017

@author: NAVEENA
"""

#linear regression 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the data se
dataset = pd.read_csv('Salary_Data.csv')
X_inde = dataset.iloc[:,:-1]
Y_depe = dataset.iloc[:,1]
#spliting into the data set
from sklearn.cross_validation import train_test_split
train_X,test_X = train_test_split(X_inde,test_size=1/3,random_state =0)
train_Y, test_Y = train_test_split(Y_depe,test_size=1/3,random_state =0)
#simple linearregression no need of feature scaling
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train_X,train_Y)
#predicting the new observartion
y_pred = regressor.predict(test_X)
y_pred_train = regressor.predict(train_X)
plt.scatter(train_X,train_Y, color = 'red')
plt.plot(train_X, y_pred_train, color ='blue')
plt.title('Salary vs Expreince')
plt.xlabel('year of experince')
plt.ylabel('Salary')
plt.show()

plt.scatter(test_X,test_Y, color = 'red')
plt.plot(train_X, y_pred_train , color ='blue')
plt.title('Salary vs Expreince(testing set)')
plt.xlabel('year of experince')
plt.ylabel('Salary')
plt.show

