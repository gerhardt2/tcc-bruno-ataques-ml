import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Ler o dataset
df = pd.read_csv("datasets/ransomware.csv")

# 2. Exibir colunas
print("Colunas disponíveis:", df.columns.tolist())

# 3. Remover colunas não úteis
df = df.drop(columns=['FileName', 'md5Hash'])

# 4. Identificar colunas não numéricas
colunas_nao_numericas = df.select_dtypes(include=['object']).columns.tolist()
print("Colunas não numéricas:", colunas_nao_numericas)

# 5. Transformar variáveis categóricas em dummies
df_dummies = pd.get_dummies(df, columns=colunas_nao_numericas)

# 6. Separar X e y
X = df_dummies.drop(columns=['Benign'])  # <- substituindo label por Benign
y = df_dummies['Benign']

# 7. Escalar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Treinar modelo
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 9. Importância das variáveis
importancias = modelo.feature_importances_
variaveis = X.columns

df_importancias = pd.DataFrame({'Variável': variaveis, 'Importância': importancias})
df_importancias = df_importancias.sort_values(by='Importância', ascending=False).head(10)

# 10. Plotar gráfico
plt.figure(figsize=(10,6))
plt.barh(df_importancias['Variável'], df_importancias['Importância'])
plt.xlabel("Importância")
plt.title("Top 10 Variáveis - Random Forest (Ransomware)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
