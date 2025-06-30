# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 19:59:50 2025

@author: sevva
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv('musteriler.csv')

X=veriler.iloc[:,2:4].values #musteri yas ve hacim alındı

from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=3,init='k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)
sonuclar=[]

for i in range (1,10):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_) #wcss değerleri
    
plt.plot(range(1,10),sonuclar)
plt.show()

#HC
from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=3,metric='euclidean',linkage='ward')
Y_tahmin=ac.fit_predict(X)
print(Y_tahmin)

plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100,c='red')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100,c='green')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100,c='blue')
plt.show()










