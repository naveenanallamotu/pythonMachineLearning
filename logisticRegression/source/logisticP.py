# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:03:22 2017

@author: NAVEENA
"""
#preprocessing the data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datafile = pd.read_csv("Social_Network_Ads.csv")
#making the matrix out of the them 
Y_depe = datafile.iloc[:,4].values
X_inde = datafile.iloc[:,[2,3]].values
#spliting the data
from sklearn.cross_validation import train_test_split
X_train,X_test = train_test_split(X_inde,test_size = 0.25, random_state =0)
Y_train,Y_test = train_test_split(Y_depe,test_size = 0.25, random_state =0)
#feature scaling
from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)

#linear classifer
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state =0)
log_reg.fit(X_train,Y_train)

Y_pred =log_reg.predict(X_test)
#confusion Matrix
from sklearn.metrics import confusion_matrix
con_mat = confusion_matrix(Y_test,Y_pred)
#visualizing the classification 
from matplotlib.colors import ListedColormap
X_point, Y_point = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_point[:, 0].min() - 1, stop = X_point[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_point[:, 1].min() - 1, stop = X_point[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, log_reg.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'white')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_point)):
    plt.scatter(X_point[Y_point == j, 0], X_point[Y_point == j, 1],
                c = ListedColormap(('red', 'white'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
X_point, Y_point = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_point[:, 0].min() - 1, stop = X_point[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_point[:, 1].min() - 1, stop = X_point[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, log_reg.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'white')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_point)):
    plt.scatter(X_point[Y_point == j, 0], X_point[Y_point == j, 1],
                c = ListedColormap(('red', 'white'))(i), label = j)
plt.title('Logistic Regression (test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


