# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 21:58:55 2025

@author: sevva
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

#2.1 Veri Yukleme
veriler=pd.read_csv('veriler.csv')

x=veriler.iloc[:,1:4].values #bagimsiz degiskenler
y=veriler.iloc[:,4].values #bagimli degisken
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)

#Buradan itibaren siniflandirma algoritmaları başlar
#1. Logistic Regression
from sklearn.linear_model import LogisticRegression
logr=LogisticRegression(random_state=0)
logr.fit(X_train,y_train) #egitim

y_pred=logr.predict(X_test) #tahmin
print(y_pred)
print(y_test)

#confusion matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)

# 2. KNN

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
print(cm)

# 3. SVC(SVM classifier)
from sklearn.svm import SVC
svc=SVC(kernel='poly')
svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
print('SVC')
print(cm)

# 4. Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
print('GNB')
print(cm)

# 5. Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy')

dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
print('DTC')
print(cm)

# 6. Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(X_train,y_train)

y_pred=rfc.predict(X_test)
y_proba=rfc.predict_proba(X_test)
cm=confusion_matrix(y_test, y_pred)
print('RFC')
print(cm)
print(y_test)
print(y_proba[:,0])


# 7. ROC,TPR,FPR degerleri
from sklearn import metrics
fpr,tpr,thold=metrics.roc_curve(y_test, y_proba[:,0],pos_label='e')
print(fpr)
print(tpr)







