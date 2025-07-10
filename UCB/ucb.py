# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 20:34:05 2025

@author: sevva
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv('Ads_CTR_Optimisation.csv')

#Random Selection

'''
import random

N=10000
d=10
toplam=0
secilenler=[]
for n in range(0,N):
    ad=random.randrange(d)
    secilenler.append(ad)
    odul=veriler.values[n,ad] # verilerdeki n.satır = 1 ise odul 1
    toplam=toplam+odul
    
plt.hist(secilenler)
plt.show()
'''

#UCB

import math

N=10000 # 10000 tüklama
d=10 #toplam 10 ilan var
#Ri(n)
oduller=[0]*d #ilk basta butun ilanlarin odulu 0
toplam=0 #toplam odul
#Ni(n)
tiklamalar=[0]*d #o ana kadar ki tiklamalar
secilenler=[]
for n in range(0,N):
    ad=0 #secilen ilan
    max_ucb=0  
    
    for i in range(0,d):
        if(tiklamalar[i]>0):
            ortalama=oduller[i]/tiklamalar[i]
            delta=math.sqrt(3/2*math.log(n)/tiklamalar[i])
            ucb=ortalama+delta
        else:
            ucb=N*10
        if max_ucb<ucb: #maxtan buyuk ucb cikti
            max_ucb=ucb
            ad=i
         
    secilenler.append(ad)
    tiklamalar[ad]=tiklamalar[ad]+1
    odul=veriler.values[n,ad]
    oduller[ad]=oduller[ad]+odul
    toplam=toplam+odul

print("Toplam Odul: ")
print(toplam)
    
plt.show()
    
plt.hist(secilenler)
plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    


