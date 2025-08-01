# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:06:15 2025

@author: sevva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri yukleme
veriler = pd.read_csv('veriler.csv')
print(veriler)

#veri on isleme
boy=veriler[['boy']]
print(boy)

boykilo=veriler[['boy','kilo']]
print(boykilo)


