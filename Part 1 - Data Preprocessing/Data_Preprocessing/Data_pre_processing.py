# -*- coding: utf-8 -*-

#Importing the basic required libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#Reading the dataset
dataset=pd.read_csv("Data.csv")
X=dataset.iloc[:,:-1].values #storing the independent variables in X
Y=dataset.iloc[:,3].values #storing the dependent variables in Y

##Preprocessing the raw data

#Handling missing data 
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0) #we will fill missing values represented by nan by taking the mean and applying this strategy on columns - 0
imputer=imputer.fit(X[:,1:3]) #specifying the columsn with missing data
X[:,1:3]=imputer.transform(X[:,1:3]) #transform used to replace the missing values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

#splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#feature scaling - to normalize the variables in the same range

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

sc_Y = StandardScaler()
Y_train = sc_Y.fit_transform(Y_train)"""