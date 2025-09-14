import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os

# Mostra o diretório atual e os arquivos na pasta datasets
print("Diretório atual:", os.getcwd())
print("Arquivos na pasta datasets:", os.listdir("../datasets"))

# Leitura do CSV
try:
    df = pd.read_csv("../datasets/phishing.csv", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("../datasets/phishing.csv", encoding="ISO-8859-1")

# Mostra as colunas originais
print("Colunas disponíveis:", df.columns.tolist())

# Verifica colunas não numéricas
colunas_nao_numericas = df.select_dtypes(include=['object']).columns.tolist()
print("Colunas não numéricas:", colunas_nao_numericas)

# Remove coluna 'Unnamed: 0' se existir
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Aplica get_dummies nas colunas não numéricas
df_dummies = pd.get_dummies(df, columns=colunas_nao_numericas)

# Mostra as últimas colunas
print("Últimas colunas:", df_dummies.columns[-5:].tolist())

# Define variável alvo
if 'Email Type_Phishing Email' in df_dummies.columns:
    y = df_dummies['Email Type_Phishing Email']
    X = df_dummies.drop(columns=['Email Type_Phishing Email'])
else:
    raise ValueError("Coluna alvo 'Email Type_Phishing Email' não encontrada.")

# Treina modelo Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y)

# Calcula importâncias
importancias = pd.Series(modelo.feature_importances_, index=X.columns)
importancias = importancias.sort_values(ascending=False)

# Exibe top 10
print("Top 10 variáveis mais importantes:")
print(importancias.head(10))

# Gráfico
importancias.head(10).plot(kind='barh')
plt.title("Top 10 Importâncias - Phishing Dataset")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
