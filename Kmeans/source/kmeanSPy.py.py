# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 10:48:20 2017

@author: NAVEENA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Mall_Customers.csv")
X_inde = data.iloc[:,3:5].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans_classifier = KMeans(n_clusters = i , init = 'k-means++',max_iter = 300, n_init = 10, random_state =0)
    kmeans_classifier.fit(X_inde)
    wcss.append(kmeans_classifier.inertia_)
plt.plot(range(1,11),wcss)
plt.title("elbow graph")
plt.xlabel("nof clusters")
plt.ylabel("wcs")
plt.show()
kmeans_classifier = KMeans(n_clusters = 5, init = 'k-means++',max_iter = 300, n_init = 10, random_state =0)
predictions = kmeans_classifier.fit_predict(X_inde)

#visualing the clusters two dimensions clustering
#ploting the cluster
plt.scatter(X_inde[predictions == 0,0],X_inde[predictions == 0,1], s=100, c= 'red', label ='1stcluster')
plt.scatter(X_inde[predictions == 1,0],X_inde[predictions == 1,1], s=100, c= 'blue', label ='2ndcluster')
plt.scatter(X_inde[predictions == 2,0],X_inde[predictions == 2,1], s=100, c= 'green', label ='3rdcluster')
plt.scatter(X_inde[predictions == 3,0],X_inde[predictions == 3,1], s=100, c= 'purple', label ='4thcluster')
plt.scatter(X_inde[predictions == 4,0],X_inde[predictions == 4,1], s=100, c= 'orange', label ='5thcluster')
plt.scatter(kmeans_classifier.cluster_centers_[:,0],kmeans_classifier.cluster_centers_[:,1], s=300, c= 'yellow', label ='centriod')
plt.title("cluster")
plt.xlabel("annual income")
plt.ylabel("spend score")
plt.legend()
plt.show() 
 