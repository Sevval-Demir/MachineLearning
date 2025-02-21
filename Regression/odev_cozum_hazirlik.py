#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('odev_tenis.csv')
#pd.read_csv("veriler.csv")
print(veriler)

from sklearn import preprocessing
#encoder:  Kategorik -> Numeric
veriler2=veriler.apply(preprocessing.LabelEncoder().fit_transform)

c=veriler2.iloc[:,:1]

ohe=OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print(c)

havadurumu=pd.DataFrame(data=c,index=range(14),columns=['o','r','s'])
sonveriler=pd.concat([havadurumu,veriler.iloc[:,1:3]],axis=1)
sonveriler=pd.concat([veriler2.iloc[:,-2:],sonveriler],axis=1)

x_train,x_test,y_train,y_test=train_test_split(sonveriler.iloc[:,:-1], sonveriler.iloc[:,-1:],test_size=0.33,random_state=0)

regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)


X=np.append(arr=np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1],axis=1)
X_list=sonveriler.iloc[:,[0,1,2,3,4,5]].values
X_list=np.array(X_list,dtype=float)

model=sm.OLS(sonveriler.iloc[:,-1:], X_list).fit()
print(model.summary())

#backward ile model oluşturma

sonveriler=sonveriler.iloc[:,1:]
x_train=x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:] 

regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

X=np.append(arr=np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1],axis=1)


X_l = sonveriler.iloc[:,[1,2,3,4]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())

sonveriler=sonveriler.iloc[:,1:]

X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())

x_train=x_train.iloc[:,1:]
x_test=x_test.iloc[:,1:]

regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)



