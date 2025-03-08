import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Veri yükleme
veri = pd.read_csv('maaslar.csv')

# Data Frame dilimleme (slice)
X = veri.iloc[:, 1:2].values  # Bağımsız değişken
Y = veri.iloc[:, 2:].values   # Bağımlı değişken

# Lineer Regresyon
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# Lineer Regresyon Grafiği
plt.scatter(X, Y, color='orange')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title("Lineer Regresyon")
plt.xlabel("Deneyim (Yıl)")
plt.ylabel("Maaş")
plt.show()

# Polinom Regresyon
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)

# Polinom Modeli Eğitme
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)

# Polinom Regresyon Grafiği
plt.scatter(X, Y, color='orange')
plt.plot(X, lin_reg2.predict(X_poly), color='red')
plt.title("Polinom Regresyon")
plt.xlabel("Deneyim (Yıl)")
plt.ylabel("Maaş")
plt.show()

# Tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.transform(np.array([[6.6]]))))  # Düzeltilmiş
print(lin_reg2.predict(poly_reg.transform(np.array([[11]]))))  # Düzeltilmiş

# Verileri Ölçekleme (SVR İçin)
sc1 = StandardScaler()
X_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
Y_olcekli = sc2.fit_transform(Y).flatten()

# SVR Modeli
svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_olcekli, Y_olcekli)

# SVR Grafiği
plt.scatter(X_olcekli, Y_olcekli, color='red')
plt.plot(X_olcekli, svr_reg.predict(X_olcekli), color='orange')
plt.title("Support Vector Regression (SVR)")
plt.show()

# SVR Tahminler
print(svr_reg.predict(sc1.transform(np.array([[11]]))))  # Düzeltilmiş
print(svr_reg.predict(sc1.transform(np.array([[6.6]]))))  # Düzeltilmiş

# Decision Tree Regressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X, Y)


X_grid = np.arange(min(X), max(X), 0.01)  
X_grid = X_grid.reshape(-1, 1)

plt.scatter(X, Y, color='red')
plt.plot(X_grid, r_dt.predict(X_grid), color='blue', linestyle='dashed')
plt.title("Decision Tree Regression")
plt.xlabel("Deneyim (Yıl)")
plt.ylabel("Maaş")
plt.show()

# Decision Tree Tahminler
print(r_dt.predict(np.array([[11]])))  # Düzeltilmiş
print(r_dt.predict(np.array([[6.6]])))  # Düzeltilmiş
