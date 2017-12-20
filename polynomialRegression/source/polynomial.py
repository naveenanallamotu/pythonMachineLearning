# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 09:34:53 2017

@author: NAVEENA
"""

#polynomial regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 #importing the dataset
data = pd.read_csv("Position_Salaries.csv")
X_inde = data.iloc[:,1:2].values # always making it matrices 
Y_depe = data.iloc[:,2].values
# fitting the linear regression model
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X_inde,Y_depe)
#converting the data according to the polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 4)
X_polynomial = poly.fit_transform(X_inde)
#now ing running the linear regression
linear_regressor2 = LinearRegression()
linear_regressor2.fit(X_polynomial,Y_depe)
#visualizing the linear regression results

plt.scatter(X_inde,Y_depe,color='green')
plt.plot(X_inde, linear_regressor.predict( X_inde),color = 'red')
plt.title('estimating the salary(linearRgeression)')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()

#visualing the polynomial regression
X_grid =np.arange(min(X_inde),max(X_inde),0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_inde,Y_depe,color='green')
plt.plot(X_grid, linear_regressor2.predict(poly.fit_transform(X_grid)),color = 'red')
plt.title('esitmating the salary(polynomiaRegrssion)')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()
#perdecting the data
perdictwithpoly = linear_regressor2.predict(poly.fit_transform(6.5))
perdictwithlinear = linear_regressor.predict(6.5)