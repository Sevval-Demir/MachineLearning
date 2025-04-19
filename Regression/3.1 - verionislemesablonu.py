
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)







#verilerin egitim ve test icin bolunmesi

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

# boy kolonunu aradan cekiyoruz
boy = s2.iloc[:,3:4]
print(boy)
sol = s2.iloc[:,:2]
sag = s2.iloc[:,4:]
veri = pd.concat([sol,sag],axis = 1)
x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)


# p-value hesaplama, çoklu doğrusal regresyon
#denkelemindeki sabit değeri ekleme 
X = np.append(arr = np.ones((22,1)).astype(int), values=veri, axis=1)

#bütün degerler dahil edilerek modelin basari testi yapiliyor
X_l = veri.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())
#backward ile model oluşumu
