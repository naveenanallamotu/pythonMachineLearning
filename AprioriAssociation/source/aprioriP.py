# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 12:37:03 2017

@author: NAVEENA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
# data is a dataframe but not the case apriori list of list is expected by apriori
transcation = []
for i in range(0,7501):
    transcation.append([str(data.values[i,j])for j in range(0,20)])
from apyori import apriori
rules = apriori(transcation, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
visuals = list(rules)