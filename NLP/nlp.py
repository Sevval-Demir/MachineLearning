# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 21:52:36 2025

@author: sevva
"""

import numpy as np
import pandas as pd

yorumlar=pd.read_csv('Restaurant_Reviews.csv',on_bad_lines='skip')

import re

yorum=re.sub('[^a-zA-Z]',' ',yorumlar['Review'][6])
yorum=yorum.lower()
yorum=yorum.split()

import nltk
durma=nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

