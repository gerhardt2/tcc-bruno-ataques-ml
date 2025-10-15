import random

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle

SEEDS = np.arange(1, 20) * 1000

# Função para avaliar modelos com validação cruzada
def avaliar_modelo(nome_modelo, modelo, X, y):
    print(f'\nModelo: {nome_modelo}')
    accs, precs, recs, f1s = [], [], [], []



    for seed in SEEDS:  # Você pode expandir para 20 seeds se quiser mais precisão
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            accs.append(accuracy_score(y_test, y_pred))
            precs.append(precision_score(y_test, y_pred, zero_division=0))
            recs.append(recall_score(y_test, y_pred, zero_division=0))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))

    print(f"Acurácia média: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Precisão média: {np.mean(precs):.4f}")
    print(f"Revocação média: {np.mean(recs):.4f}")
    print(f"F1-score médio: {np.mean(f1s):.4f}")

    return {
        'modelo': nome_modelo,
        'acuracia_media': np.mean(accs),
        'desvio_padrao': np.std(accs),
        # 'desvio_padrao': np.std(accs), # DESVIO PADRÃO DE TODAS AS MEDIDAS
        'precisao_media': np.mean(precs),
        'revocacao_media': np.mean(recs),
        'f1_score_media': np.mean(f1s)
    }

# Função principal para testar cada dataset
def testar_dataset(caminho, nome_dataset):
    print(f"\n=== Avaliando Dataset: {nome_dataset} ===")
    df = pd.read_csv(caminho)

    if 'target' not in df.columns:
        raise ValueError("Coluna 'target' não encontrada no dataset!")

    # Separar atributos e rótulo
    X = df.drop(columns=['target'])
    X = X.select_dtypes(include=[np.number])  # Apenas colunas numéricas
    y = df['target']

    # Escalonar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    acuracias_random_search = {}
    for seed in SEEDS:
        random.seed(seed)
        espaco_busca_n_estimators = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        espaco_busca_max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        espaco_busca_kernels = ["rbf", "linear"] # ver kerneçls
        espaco_busca_c = np.arange(0, 1, 0.02)

        for i in range(100):
            # Modelos
            #salvar em variaveis a melhor opcao
            #pega a acuracia e salvar em uma lista
            modelos = {
                'Random Forest': RandomForestClassifier(n_estimators=random.choice(espaco_busca_n_estimators), max_depth=espaco_busca_max_depth, random_state=42),
                'SVM': SVC(C=1.0, kernel='rbf', gamma='scale', random_state=seed),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=10, random_state=seed)
            }

    resultados = []
    for nome_modelo, modelo in modelos.items():
        resultado = avaliar_modelo(nome_modelo, modelo, X_scaled, y.values)
        resultado['dataset'] = nome_dataset
        resultados.append(resultado)
        acuracias_random_search[(nome_modelo,)] = resultado # colocar depois da , uma forma dele salvar o valor de cada parametro

    return resultados

# Execução para todos os conjuntos de dados
todos_resultados = []
todos_resultados += testar_dataset("datasets/DDoS.csv", "DDoS")
todos_resultados += testar_dataset("datasets/phishing.csv", "Phishing")
todos_resultados += testar_dataset("datasets/ransomware.csv", "Ransomware")

# Resultado final em DataFrame
df_resultados = pd.DataFrame(todos_resultados)
print("\n=== Resultados Finais Comparativos ===")
print(df_resultados)
