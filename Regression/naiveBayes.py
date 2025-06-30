# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 21:19:15 2025

@author: sevva
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

veriler = pd.read_csv('veriler.csv')

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:5].values


"""
x = veriler.iloc[5:,1:4].values
y = veriler.iloc[5:,4:5].values
"""

x_train, x_test, y_train, y_test, = train_test_split(x, y, test_size = 0.33, random_state = 0)


sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


from sklearn.linear_model import LogisticRegression
logR = LogisticRegression(random_state=0)
logR.fit(X_train,y_train)
yPred = logR.predict(X_test)
print(yPred)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, yPred)
print("\nLOGISTIC")
print(cm)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5,metric = "minkowski")
knn.fit(X_train,y_train)
yPred = knn.predict(X_test)
cm = confusion_matrix(y_test, yPred)
print("\nKNN")
print(cm)

from sklearn.svm import SVC
svc = SVC(kernel="rbf")
svc.fit(X_train,y_train)
yPred = svc.predict(X_test)
cm = confusion_matrix(y_test, yPred)
print("\nSVC")
print(cm)

from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()
gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)
cm = confusion_matrix(y_test, yPred)
print("\nGNB")
print(cm)

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy')


dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)

y_pred=gnb.predict(X_test)
cm = confusion_matrix(y_test, yPred)
print("\nDTC")
print(cm)

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(X_train,y_train)

y_pred=rfc.predict(X_test)
cm = confusion_matrix(y_test, yPred)
print("\nRFC")
print(cm)







