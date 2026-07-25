"""
WP1 — Ablación multi-task vs single-task, PROXY con sklearn (Reviewer 1 #25).

La ablación que pide Reviewer 1 (5 redes de una sola salida vs. la red
multi-output) requiere TensorFlow/Keras, que no se pudo ejecutar en esta
sesión (ver 06_dnn_multitask_singletask.py y METODOLOGIA-V3.md). Como
evidencia parcial mientras eso se corre en un entorno donde TF sí funcione,
este script usa una analogía disponible en sklearn (SIN TensorFlow) que
captura la misma pregunta de fondo -- "¿comparte información entre targets
ayuda o perjudica?" -- aunque con un modelo distinto (Random Forest, no una
red neuronal con tronco compartido):

  - "single-task proxy": un RandomForestRegressor INDEPENDIENTE por target
    (5 modelos, cada uno solo ve su propio target -- es lo que ya hace
    05_baselines_grouped_temporal.py target-por-target).
  - "multi-task proxy": UN SOLO RandomForestRegressor con y multi-columna
    (scikit-learn permite total naturalmente `y` de forma (n, 5) en
    RandomForestRegressor/DecisionTreeRegressor: cada árbol elige sus splits
    para reducir el error conjunto de los 5 targets simultáneamente,
    compartiendo la estructura del árbol entre targets -- la analogía más
    cercana a un "tronco compartido" que ofrece sklearn sin redes
    neuronales).

Esto NO reemplaza la ablación real con la arquitectura Keras del paper
(tronco compartido + cabezas por target) -- se reporta como evidencia
complementaria, ejecutada de verdad, mientras la ablación con DNN queda
pendiente de un entorno con TensorFlow funcional.

Protocolo: mismo split agrupado (componente estudiante+curso) + temporal
(train <=2023 / holdout 2024) + features limpias (18, sin fuga) que
05_baselines_grouped_temporal.py, para que sea directamente comparable.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
SEED = 42
N_FOLDS = 5

df = pd.read_csv(os.path.join(OUT_DIR, "clean_dataset.csv"))
with open(os.path.join(OUT_DIR, "feature_lists.json")) as f:
    feat_lists = json.load(f)
CLEAN_FEATURES = feat_lists["clean_features"]
TARGETS = feat_lists["targets"]

train_pool = df[df["split_bucket"] == "train_le2023"].reset_index(drop=True)
holdout_2024 = df[df["split_bucket"] == "holdout_2024"].reset_index(drop=True)


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def metrics_per_target(y_true_df, y_pred_mat):
    rows = []
    for i, t in enumerate(TARGETS):
        yt = y_true_df[t].to_numpy()
        yp = y_pred_mat[:, i]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rows.append({"target": t, "R2": r2, "MAE": float(np.mean(np.abs(yt - yp))), "RMSE": _rmse(yt, yp)})
    return pd.DataFrame(rows)


def preprocess(X_train_raw, X_val_raw):
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw[CLEAN_FEATURES]))
    X_val = scaler.transform(imputer.transform(X_val_raw[CLEAN_FEATURES]))
    return X_train, X_val


def fit_single_task(X_train, y_train_df, X_val):
    preds = np.zeros((X_val.shape[0], len(TARGETS)))
    for i, t in enumerate(TARGETS):
        rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        rf.fit(X_train, y_train_df[t].to_numpy())
        preds[:, i] = rf.predict(X_val)
    return preds


def fit_multi_task(X_train, y_train_df, X_val):
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train_df[TARGETS].to_numpy())
    return rf.predict(X_val)


# ---------------------------------------------------------------------
# CV agrupada dentro de <=2023
# ---------------------------------------------------------------------
gkf = GroupKFold(n_splits=N_FOLDS)
groups = train_pool["component_id"]

fold_metrics = {"single_task_proxy": [], "multi_task_proxy": []}
for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(train_pool, groups=groups)):
    X_tr_raw, X_va_raw = train_pool.iloc[tr_idx], train_pool.iloc[va_idx]
    y_tr, y_va = train_pool.iloc[tr_idx][TARGETS], train_pool.iloc[va_idx][TARGETS]
    X_tr, X_va = preprocess(X_tr_raw, X_va_raw)

    pred_st = fit_single_task(X_tr, y_tr, X_va)
    pred_mt = fit_multi_task(X_tr, y_tr, X_va)

    m_st = metrics_per_target(y_va, pred_st)
    m_st["fold"] = fold_i
    fold_metrics["single_task_proxy"].append(m_st)

    m_mt = metrics_per_target(y_va, pred_mt)
    m_mt["fold"] = fold_i
    fold_metrics["multi_task_proxy"].append(m_mt)

summary = {}
for label in fold_metrics:
    all_folds = pd.concat(fold_metrics[label], ignore_index=True)
    per_target = all_folds.groupby("target")[["R2", "MAE", "RMSE"]].mean().reindex(TARGETS).reset_index()
    summary[label] = {
        "per_target_mean_over_folds": per_target.to_dict(orient="records"),
        "global_arithmetic_mean": {
            "MAE": float(per_target["MAE"].mean()),
            "RMSE": float(per_target["RMSE"].mean()),
            "R2": float(per_target["R2"].mean()),
        },
    }
    print(f"\n--- {label} (CV agrupada, {N_FOLDS} folds, <=2023) ---")
    print(per_target.to_string(index=False))
    print("Global (media aritmética):", summary[label]["global_arithmetic_mean"])

# ---------------------------------------------------------------------
# Final: ajustado en todo <=2023, evaluado en holdout 2024
# ---------------------------------------------------------------------
X_tr, X_va = preprocess(train_pool, holdout_2024)
y_tr = train_pool[TARGETS]

pred_st_final = fit_single_task(X_tr, y_tr, X_va)
pred_mt_final = fit_multi_task(X_tr, y_tr, X_va)

m_st_final = metrics_per_target(holdout_2024[TARGETS], pred_st_final)
m_mt_final = metrics_per_target(holdout_2024[TARGETS], pred_mt_final)

print("\n--- single_task_proxy (holdout 2024) ---")
print(m_st_final.to_string(index=False))
print("\n--- multi_task_proxy (holdout 2024) ---")
print(m_mt_final.to_string(index=False))

summary["holdout_2024"] = {
    "single_task_proxy": m_st_final.to_dict(orient="records"),
    "multi_task_proxy": m_mt_final.to_dict(orient="records"),
}

with open(os.path.join(OUT_DIR, "07_proxy_ablation_results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nGuardado: {os.path.join(OUT_DIR, '07_proxy_ablation_results.json')}")
