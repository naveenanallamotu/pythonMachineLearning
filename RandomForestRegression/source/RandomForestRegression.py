# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:19:08 2017

@author: NAVEENA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X_inde = dataset.iloc[:, 1:2].values
y_depe = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators =100 , random_state =0)
regressor.fit(X_inde,y_depe)
# Predicting a new result
y_pred = regressor.predict(6.5)



# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X_inde), max(X_inde), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_inde, y_depe, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('estimation of salary (RandomForestRegression Model)')
plt.xlabel(' level')
plt.ylabel('Salary')
plt.show()