# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:37:40 2025

@author: sevva
"""

class insan:
    boy=180
    def kosmak(self,b):
        return b+10
ali=insan()
print(ali.boy)
print(ali.kosmak(90))

#Eksik Veriler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veri=pd.read_csv('eksikveriler.csv')
print(veri)

from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy='mean')

Yas=veri.iloc[:,1:4].values
print(Yas)
imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print(Yas)