
#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#importing the data file
dataset = pd.read_csv('Data.csv')
#converting them into matrices
X = dataset.iloc[:,:-1].values
Y= dataset.iloc[:,3].values

#takingcare of the missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values ='NaN', strategy = 'mean', axis =0) 
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
print(X[:,1:3])
#categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x = LabelEncoder() # creating the object
X[:,0]= labelencoder_x.fit_transform(X[:,0])
# here we are using other technique to convert cateogerical value
oneHotEncoder_x = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder_x.fit_transform(X).toarray()
labelencoder_y = LabelEncoder() # creating the object
Y= labelencoder_y.fit_transform(Y)
#traing and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, Y_tarin, Y_test = train_test_split(X,Y, test_size = 0.2,random_state = 0)
#feature scaling
from sklearn.preprocessing  import StandardScaler
x_scaler = StandardScaler()
x_train = x_scaler.fit_transform(x_train)
x_test =x_scaler.transform(x_test)

