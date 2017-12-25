# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:30:16 2017

@author: NAVEENA
"""
#hierarchical clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")
X_inde = data.iloc[:,3:5].values
#here we need to build the dendogram to have th optimal cluster
import scipy.cluster.hierarchy as sch
dendrogram =sch.dendrogram(sch.linkage(X_inde, method = 'ward'))
plt.title("dendrogram")
plt.xlabel("cluster")
plt.ylabel("distances")
plt.show()

#fitting the hc
from sklearn.cluster import AgglomerativeClustering
Agc =AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean',linkage ='ward')
y_hc = Agc.fit_predict(X_inde)

#visulazing the code
plt.scatter(X_inde[y_hc == 0, 0], X_inde[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_inde[y_hc == 1, 0], X_inde[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_inde[y_hc == 2, 0], X_inde[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X_inde[y_hc == 3, 0], X_inde[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X_inde[y_hc == 4, 0], X_inde[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()