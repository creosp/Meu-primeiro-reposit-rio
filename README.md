import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Dados
df = pd.read_csv("MKT.csv")

# Exibir o dataset
print("Visualizando as primeiras linhas do dataset:")
print(df.head())

# Análise descritiva
print("\nResumo estatístico do dataset:")
print(df.describe())

# Verificar valores ausentes
print("\nValores ausentes por coluna:")
print(df.isnull().sum())

# Verificar dados duplicados
print("\nNúmero de linhas duplicadas:", df.duplicated().sum())

# Análise exploratória
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlação entre as variáveis")
plt.show()

# Histogramas das variáveis
plt.figure(figsize=(12, 6))
df.hist(bins=30, figsize=(12, 6), edgecolor='black')
plt.suptitle("Distribuição das Variáveis", fontsize=16)
plt.show()

# Separação das variáveis
X = df[['youtube', 'facebook', 'newspaper']]
y = df['sales']

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predições
y_pred = modelo.predict(X_test)

# Avaliação do modelo
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\nAvaliação do Modelo:")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Exibir coeficientes do modelo
print("\nCoeficientes do Modelo:")
print("Youtube:", modelo.coef_[0])
print("Facebook:", modelo.coef_[1])
print("Newspaper:", modelo.coef_[2])
print("Intercepto:", modelo.intercept_)
