#%% md
# # Importações
#%%
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC

SEEDS = np.arange(1, 11) * 1000

print(SEEDS)

df = pd.read_csv("../datasets/phishing_transformado.csv", sep=";", decimal=".")

X = df.drop(columns="Email Type_Phishing Email")
Y = df["Email Type_Phishing Email"]

df.head()

#%% md
# # Parâmetros dos Modelos
#%% md
# ## GBM
# 
#%%
# ======== ESPAÇOS PARA AMOSTRAGEM ALEATÓRIA (GBM) ========
n_estimators_choices = list(range(50, 501, 10))  # 50..500
max_depth_choices = list(range(2, 101, 2))  # 2..100
min_samples_split_choices = list(range(2, 21))  # 2..20
min_samples_leaf_choices = list(range(1, 21))  # 1..20
# (Opcional) Você pode incluir também learning_rate/subsample:
# learning_rate_choices = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
# subsample_choices     = [0.6, 0.8, 1.0]

# ================== LOOP PRINCIPAL ==================
rows = []

for seed in SEEDS:
    random.seed(int(seed))
    np.random.seed(seed)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for i in range(10):  # 25 testes por seed
        params_rf = {
            "n_estimators": random.choice(n_estimators_choices),
            "max_depth": random.choice(max_depth_choices),
            "min_samples_split": random.choice(min_samples_split_choices),
            "min_samples_leaf": random.choice(min_samples_leaf_choices),
            "random_state": seed,
        }
        gbm = GradientBoostingClassifier(**params_rf)

        # Cross Validation 5-fold
        scores_rf = cross_val_score(
            gbm, X, Y,
            cv=skf,
            scoring="accuracy",
        )

        rows.append({
            "seed": seed,
            "iter": i + 1,
            "clf__n_estimators": params_rf["n_estimators"],
            "clf__max_depth": params_rf["max_depth"],
            "clf__min_samples_split": params_rf["min_samples_split"],
            "clf__min_samples_leaf": params_rf["min_samples_leaf"],
            "mean_accuracy": float(np.mean(scores_rf)),
            "std_accuracy": float(np.std(scores_rf)),
        })

        # ================== SALVAR RESULTADOS ==================
        pd.DataFrame(rows).to_csv("../resultados/parametros_gbm_phishing.csv", index=False, decimal=".", sep=";")

best_gbm = pd.DataFrame(rows).loc[df["mean_accuracy"].idxmax()]
print("Melhores Parâmetros GBM", best_gbm)


#%% md
# ## RF
#%%
# ======== ESPAÇOS PARA AMOSTRAGEM ALEATÓRIA ========
n_estimators_choices = list(range(50, 501, 10))  # 50..500 passo 10
max_depth_choices = list(range(2, 101, 2))  # 2..100
min_samples_split_choices = list(range(2, 21))  # 2..20
min_samples_leaf_choices = list(range(1, 21))  # 1..20

# ======== LOOP PRINCIPAL ========
rows = []

for seed in SEEDS:
    random.seed(int(seed))
    np.random.seed(seed)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for i in range(10):  # 25 testes por seed
        params_rf = {
            "n_estimators": random.choice(n_estimators_choices),
            "max_depth": random.choice(max_depth_choices),
            "min_samples_split": random.choice(min_samples_split_choices),
            "min_samples_leaf": random.choice(min_samples_leaf_choices),
            "random_state": seed,
        }

        rf = RandomForestClassifier(**params_rf)

        # Cross Validation 5-fold
        scores_rf = cross_val_score(
            rf, X, Y,
            cv=skf,
            scoring="accuracy",
        )

        rows.append({
            "seed": seed,
            "iter": i + 1,
            "clf__n_estimators": params_rf["n_estimators"],
            "clf__max_depth": params_rf["max_depth"],
            "clf__min_samples_split": params_rf["min_samples_split"],
            "clf__min_samples_leaf": params_rf["min_samples_leaf"],

            "mean_accuracy": float(np.mean(scores_rf)),
            "std_accuracy": float(np.std(scores_rf)),
        })

        # ======== SALVAR RESULTADOS ========
        pd.DataFrame(rows).to_csv("../resultados/parametros_rf_phishing.csv", index=False, decimal=".", sep=";")

best_rf = pd.DataFrame(rows).loc[df["mean_accuracy"].idxmax()]
print("Melhores Parâmetros RF", best_rf)

#%% md
# ## SVM
#%%
# ======== ESPAÇOS PARA AMOSTRAGEM ALEATÓRIA (SVM) ========
kernel_choices = ["linear", "rbf", "poly", "sigmoid"]
C_choices = list(range(50, 501, 10))
gamma_choices = list(np.arange(0, 1, 0.05))

# ================== LOOP PRINCIPAL ==================
rows = []

for seed in SEEDS:
    random.seed(int(seed))
    np.random.seed(seed)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for i in range(10):  # 10 testes por seed
        kernel = random.choice(kernel_choices)
        C = random.choice(C_choices)
        gamma = random.choice(gamma_choices)

        svm = SVC(kernel=kernel, gamma=gamma, C=C, random_state=seed)
        print(kernel, C, gamma)
        # Cross Validation 5-fold
        scores_rf = cross_val_score(
            svm, X, Y,
            cv=skf,
            scoring="accuracy",
        )

        # monta linha de saída
        row = {
            "seed": seed,
            "iter": i + 1,
            "clf__kernel": kernel,
            "clf__C": C,
            "clf__gamma": gamma,
            "mean_accuracy": float(np.mean(scores_rf)),
            "std_accuracy": float(np.std(scores_rf)),
        }

        # ================== SALVAR RESULTADOS ==================
        pd.DataFrame(rows).to_csv("../resultados/parametros_svm_phishing.csv", index=False, decimal=".", sep=";")

best_svm = pd.DataFrame(rows).loc[df["mean_accuracy"].idxmax()]
print("Melhores Parâmetros SVM", best_svm)
#%% md
# # Execução dos Testes
# ## Validação Cruzada
# 
#%%
# ================== PARÂMETROS ==================
params_gbm = {
    "n_estimators": int(best_gbm["clf__n_estimators"]),
    "max_depth": int(best_gbm["clf__max_depth"]),
    "min_samples_split": int(best_gbm["clf__min_samples_split"]),
    "min_samples_leaf": int(best_gbm["clf__min_samples_leaf"]),
}

params_rf = {
    "n_estimators": int(best_rf["clf__n_estimators"]),
    "max_depth": int(best_rf["clf__max_depth"]),
    "min_samples_split": int(best_rf["clf__min_samples_split"]),
    "min_samples_leaf": int(best_rf["clf__min_samples_leaf"]),
}


params_svm = {
    "kernel": str(best_svm["clf__kernel"]),
    "C": float(best_svm["clf__C"]),
    "gamma": best_svm["clf__gamma"],  # pode ser "scale"/"auto" ou numérico
}
if "clf__degree" in best_svm:
    params_svm["degree"] = int(best_svm["clf__degree"])
if "clf__coef0" in best_svm:
    params_svm["coef0"] = float(best_svm["clf__coef0"])

# ================== RESULTADOS ==================
results_gbm = []
results_rf = []
results_svm = []

for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # ===== RF =====
    model_rf = RandomForestClassifier(**params_rf, random_state=seed)
    scores_rf = cross_val_score(model_rf, X, Y, cv=skf, scoring="accuracy")
    results_rf.append({
        "seed": seed,
        "mean_accuracy": scores_rf.mean(),
        "std_accuracy": scores_rf.std(),
        "fold_1": scores_rf[0],
        "fold_2": scores_rf[1],
        "fold_3": scores_rf[2],
        "fold_4": scores_rf[3],
        "fold_5": scores_rf[4],
    })

    # ===== GBM =====
    model_gbm = GradientBoostingClassifier(**params_gbm, random_state=seed)
    scores_gbm = cross_val_score(model_gbm, X, Y, cv=skf, scoring="accuracy")
    results_gbm.append({
        "seed": seed,
        "mean_accuracy": scores_gbm.mean(),
        "std_accuracy": scores_gbm.std(),
        "fold_1": scores_gbm[0],
        "fold_2": scores_gbm[1],
        "fold_3": scores_gbm[2],
        "fold_4": scores_gbm[3],
        "fold_5": scores_gbm[4],
    })

    # ===== SVM =====
    model_svm = SVC(**params_svm, random_state=seed)
    scores_svm = cross_val_score(model_svm, X, Y, cv=skf, scoring="accuracy")
    results_svm.append({
        "seed": seed,
        "mean_accuracy": scores_svm.mean(),
        "std_accuracy": scores_svm.std(),
        "fold_1": scores_svm[0],
        "fold_2": scores_svm[1],
        "fold_3": scores_svm[2],
        "fold_4": scores_svm[3],
        "fold_5": scores_svm[4],
    })


    # salvar novo csv
    pd.DataFrame(results_gbm).to_csv("../resultados/resultados_validacao_cruzada_gbm_phishing.csv", index=False, decimal=".",
                                    sep=";")
    pd.DataFrame(results_rf).to_csv("../resultados/resultados_validacao_cruzada_rf_phishing.csv", index=False, decimal=".",
                                    sep=";")
    pd.DataFrame(results_svm).to_csv("../resultados/resultados_validacao_cruzada_svm_phishing.csv", index=False, decimal=".",
                                    sep=";")

#%% md
# ## Holdout
# 
#%%
results_gbm = []
results_rf = []
results_svm = []

for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)

    # Hold-Out
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=0.4,
        shuffle=True,
        random_state=seed,
    )

    # ===== RF =====
    model_rf = RandomForestClassifier(
        **params_rf,
        random_state=seed
    )
    model_rf.fit(X_train, y_train)
    acc_rf = model_rf.score(X_test, y_test)

    results_rf.append({
        "seed": seed,
        "accuracy": acc_rf
    })

    # ===== GBM =====
    model_gbm = GradientBoostingClassifier(
        **params_gbm,
        random_state=seed
    )
    model_gbm.fit(X_train, y_train)
    acc_gbm = model_gbm.score(X_test, y_test)

    results_gbm.append({
        "seed": seed,
        "accuracy": acc_gbm
    })

    # ===== SVM =====
    model_svm = SVC(
        **params_svm,
        random_state=seed
    )
    model_svm.fit(X_train, y_train)
    acc_svm = model_svm.score(X_test, y_test)

    results_svm.append({
        "seed": seed,
        "accuracy": acc_svm
    })

    # salvar novo csv
    pd.DataFrame(results_gbm).to_csv("../resultados/resultados_holdout_gbm_phishing.csv", index=False, sep=";", decimal=".")
    pd.DataFrame(results_rf).to_csv("../resultados/resultados_holdout_rf_phishing.csv", index=False, sep=";", decimal=".")
    pd.DataFrame(results_svm).to_csv("../resultados/resultados_holdout_svm_phishing.csv", index=False, sep=";", decimal=".")

