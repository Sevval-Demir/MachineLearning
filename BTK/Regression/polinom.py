# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:44:49 2025

@author: sevva
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Veri yükleme
veri = pd.read_csv('maaslar.csv')

#data frame dilimleme (slice)
X = veri.iloc[:, 1:2].values  # Bağımsız değişken
Y = veri.iloc[:, 2:].values   # Bağımlı değişken

# Lineer Regresyon
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# Lineer Regresyon grafiği
plt.scatter(X, Y, color='orange')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title("Lineer Regresyon")
plt.xlabel("Deneyim (Yıl)")
plt.ylabel("Maaş")
plt.show()

# Polinom Regresyon
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)  # Dereceyi artırarak daha iyi eğri elde edilebilir
X_poly = poly_reg.fit_transform(X)  # X değerlerini polinomik hale getir

# Polinom Modeli Eğitme
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y) 

# Polinom Regresyon grafiği
plt.scatter(X, Y, color='orange')
plt.plot(X, lin_reg2.predict(X_poly), color='red') 
plt.title("Polinom Regresyon")
plt.xlabel("Deneyim (Yıl)")
plt.ylabel("Maaş")
plt.show()

#Tahminler

#print(lin_reg.predict([[11]]))
#print(lin_reg.predict([[6.6]]))

#print(lin_reg2.predict(poly_reg.fit_transform([6.6])))
#print(lin_reg2.predict(poly_reg.fit_transform([11])))











