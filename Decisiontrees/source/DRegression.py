# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:59:27 2017

@author: NAVEENA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 #importing the dataset
data = pd.read_csv("Position_Salaries.csv")
X_inde = data.iloc[:,1:2].values # always making it matrices 
Y_depe = data.iloc[:,2].values
#buikding our model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_inde,Y_depe)
y_pred = regressor.predict(6.5)
#visualing the  regression
plt.scatter(X_inde,Y_depe,color='green')
plt.plot(X_inde, regressor.predict(X_inde),color = 'red')
plt.title('esitmating the salary(regression)')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()

X_grid =np.arange(min(X_inde),max(X_inde),0.01) # here we get vector
X_grid = X_grid.reshape((len(X_grid), 1))      # reshaping into the matrices
plt.scatter(X_inde,Y_depe,color='green')
plt.plot(X_grid, regressor.predict(X_grid),color = 'red') # because only takes martrices
plt.title('esitmating the salary(polynomiaRegrssion)')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()

