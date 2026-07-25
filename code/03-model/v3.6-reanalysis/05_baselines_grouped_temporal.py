"""
WP1 — Paso 5: Reanálisis principal — baselines honestos (LR/DT/RF) sobre el
dataset limpio, con:
  - split agrupado por estudiante+curso (componentes conexas, Reviewer 1 #7-8)
  - holdout temporal ≤2023 (train+CV) / 2024 (test final, Reviewer 1 #9)
  - preprocesamiento (imputación + RobustScaler) ajustado SOLO con cada
    partición de entrenamiento (Reviewer 1 #18-19)
  - fórmula de agregación de métricas UNIFICADA (Reviewer 1 #28, #30)
  - bootstrap CI por componente (estudiante+curso), no por fila
    (Reviewer 1 #29, #31)

NOTA IMPORTANTE (ver METODOLOGIA-V3.md): no se pudo instalar/ejecutar
TensorFlow/Keras en el entorno conda local `mexihc-v36` de esta sesión —
`import tensorflow` cuelga indefinidamente (proceso en estado
ininterrumpible, no responde ni a SIGKILL) tanto dentro como fuera del
sandbox de esta herramienta. Es un bloqueo de infraestructura (posible
incompatibilidad Rosetta x86_64/arm64 en este Mac), no una decisión de
alcance. Por eso este script cubre SOLO baselines clásicos (sklearn, sin
TensorFlow) con números reales; el modelo DNN multi-tarea y la ablación
multi-task vs single-task quedan en `06_dnn_multitask_singletask.py`,
escrito y listo, pero NO ejecutado.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")

SEED = 42
N_FOLDS = 5
N_BOOTSTRAP = 2000
ALPHA = 0.05

df = pd.read_csv(os.path.join(OUT_DIR, "clean_dataset.csv"))
with open(os.path.join(OUT_DIR, "feature_lists.json")) as f:
    feat_lists = json.load(f)

CLEAN_FEATURES = feat_lists["clean_features"]
LEAKY_26_FEATURES = feat_lists["all_26_features"]
TARGETS = feat_lists["targets"]

# clean_dataset.csv (paso 3) ya excluyó las 8 columnas leaky del set de
# trabajo principal; para la comparación "antes/después" de la sección 1
# hace falta reincorporarlas desde dataset_final_3_x.csv (mismo uid/course).
FINAL_CSV = os.path.join(HERE, "..", "..", "v3.6", "data", "Material", "dataset_final_3_x.csv")
df_26 = pd.read_csv(FINAL_CSV)
leaky_only_cols = [c for c in LEAKY_26_FEATURES if c not in df.columns]
df = df.merge(df_26[["uid_hash", "course_hash"] + leaky_only_cols], on=["uid_hash", "course_hash"], how="left")

train_pool = df[df["split_bucket"] == "train_le2023"].reset_index(drop=True)
holdout_2024 = df[df["split_bucket"] == "holdout_2024"].reset_index(drop=True)

print(f"Train pool (<=2023): {train_pool.shape[0]} filas / {train_pool['component_id'].nunique()} componentes")
print(f"Holdout (2024):      {holdout_2024.shape[0]} filas / {holdout_2024['component_id'].nunique()} componentes")

MODELS = {
    "LinearRegression": lambda: LinearRegression(),
    "DecisionTree": lambda: DecisionTreeRegressor(random_state=SEED, max_depth=8, min_samples_leaf=5),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, max_depth=None),
}


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def fit_predict_per_target(X_train, y_train_df, X_val, model_factory):
    """Ajusta un modelo por target (sklearn no soporta out-of-the-box RMSE-Huber
    multi-output para LR/DT en un solo objeto comparable al DNN), y devuelve
    la matriz de predicciones (n_val, n_targets)."""
    preds = {}
    for t in TARGETS:
        est = model_factory()
        est.fit(X_train, y_train_df[t].to_numpy())
        preds[t] = est.predict(X_val)
    return pd.DataFrame(preds)


def run_pipeline_on_split(X_train_raw, y_train_df, X_val_raw, feature_set):
    """Imputa + escala SOLO con X_train_raw, transforma X_val_raw, entrena los
    3 baselines target-por-target, devuelve dict {model_name: pred_df}."""
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw[feature_set]))
    X_val = scaler.transform(imputer.transform(X_val_raw[feature_set]))
    out = {}
    for model_name, factory in MODELS.items():
        out[model_name] = fit_predict_per_target(X_train, y_train_df, X_val, factory)
    return out


def compute_metrics_per_target(y_true_df, y_pred_df):
    rows = []
    for t in TARGETS:
        yt = y_true_df[t].to_numpy()
        yp = y_pred_df[t].to_numpy()
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = _rmse(yt, yp)
        rows.append({"target": t, "R2": r2, "MAE": mae, "RMSE": rmse})
    return pd.DataFrame(rows)


def aggregate_global(per_target_df):
    """Fórmula de agregación UNIFICADA (Reviewer 1 #28,#30): media aritmética
    para las 3 métricas. Se reporta también la media cuadrática de RMSE como
    referencia explícita del criterio que usaba el paper original para poder
    trazar la diferencia (ver METODOLOGIA-V3.md)."""
    return {
        "MAE_arithmetic_mean": float(per_target_df["MAE"].mean()),
        "RMSE_arithmetic_mean": float(per_target_df["RMSE"].mean()),
        "RMSE_quadratic_mean": float(np.sqrt((per_target_df["RMSE"] ** 2).mean())),
        "R2_arithmetic_mean": float(per_target_df["R2"].mean()),
    }


# =====================================================================
# 1) Comparación "ANTES / DESPUÉS" del filtro anti-fuga, con RandomForest,
#    usando el MISMO protocolo agrupado+temporal (para que sea comparable)
# =====================================================================
print("\n" + "=" * 78)
print("1) COMPARACIÓN ANTES (26 features, con fuga) / DESPUÉS (18 features, limpio)")
print("   RandomForest, entrenado en <=2023, evaluado en 2024 (holdout real)")
print("=" * 78)

before_after = {}
for label, fset in [("before_26_leaky", LEAKY_26_FEATURES), ("after_18_clean", CLEAN_FEATURES)]:
    preds = run_pipeline_on_split(train_pool, train_pool[TARGETS], holdout_2024, fset)
    m = compute_metrics_per_target(holdout_2024[TARGETS], preds["RandomForest"])
    before_after[label] = {"per_target": m.to_dict(orient="records"), "global": aggregate_global(m)}
    print(f"\n--- {label} ---")
    print(m.to_string(index=False))
    print("Global:", before_after[label]["global"])

# =====================================================================
# 2) CV agrupada (componente = estudiante+curso) DENTRO de <=2023,
#    3 modelos, features limpias. Predicciones out-of-fold (OOF) para
#    bootstrap CI "interno".
# =====================================================================
print("\n" + "=" * 78)
print(f"2) CV agrupada por componente ({N_FOLDS} folds) dentro de <=2023 -- features limpias")
print("=" * 78)

gkf = GroupKFold(n_splits=N_FOLDS)
groups = train_pool["component_id"]

oof_preds = {m: pd.DataFrame(index=train_pool.index, columns=TARGETS, dtype=float) for m in MODELS}
fold_metrics = {m: [] for m in MODELS}

for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(train_pool, groups=groups)):
    X_tr, X_va = train_pool.iloc[tr_idx], train_pool.iloc[va_idx]
    y_tr = train_pool.iloc[tr_idx][TARGETS]
    y_va = train_pool.iloc[va_idx][TARGETS]

    fold_preds = run_pipeline_on_split(X_tr, y_tr, X_va, CLEAN_FEATURES)
    for model_name, pred_df in fold_preds.items():
        pred_df.index = X_va.index
        oof_preds[model_name].loc[X_va.index, :] = pred_df.values
        m = compute_metrics_per_target(y_va, pred_df)
        m["fold"] = fold_i
        fold_metrics[model_name].append(m)
    print(f"  fold {fold_i}: train={len(tr_idx)} val={len(va_idx)} "
          f"(componentes train={train_pool.iloc[tr_idx]['component_id'].nunique()}, "
          f"val={train_pool.iloc[va_idx]['component_id'].nunique()})")

cv_summary = {}
for model_name in MODELS:
    all_folds = pd.concat(fold_metrics[model_name], ignore_index=True)
    per_target_cv = all_folds.groupby("target")[["R2", "MAE", "RMSE"]].mean().reindex(TARGETS).reset_index()
    cv_summary[model_name] = {"per_target_mean_over_folds": per_target_cv.to_dict(orient="records"),
                               "global": aggregate_global(per_target_cv)}
    print(f"\n--- {model_name} (media sobre {N_FOLDS} folds, OOF dentro de <=2023) ---")
    print(per_target_cv.to_string(index=False))
    print("Global:", cv_summary[model_name]["global"])

# =====================================================================
# 3) Modelo final: entrenado en TODO <=2023, evaluado en holdout 2024
#    (evaluación temporal principal que pide el editor / Reviewer 1 #9)
# =====================================================================
print("\n" + "=" * 78)
print("3) Modelo final (ajustado en TODO <=2023) evaluado en holdout 2024 real")
print("=" * 78)

final_preds = run_pipeline_on_split(train_pool, train_pool[TARGETS], holdout_2024, CLEAN_FEATURES)
final_summary = {}
for model_name, pred_df in final_preds.items():
    pred_df.index = holdout_2024.index
    m = compute_metrics_per_target(holdout_2024[TARGETS], pred_df)
    final_summary[model_name] = {"per_target": m.to_dict(orient="records"), "global": aggregate_global(m)}
    print(f"\n--- {model_name} (holdout 2024, n={len(holdout_2024)}) ---")
    print(m.to_string(index=False))
    print("Global:", final_summary[model_name]["global"])

# =====================================================================
# 4) Bootstrap CI por COMPONENTE (no por fila), Reviewer 1 #29/#31
#    (a) sobre las predicciones OOF dentro de <=2023 (66 componentes -> CI
#        razonablemente informativo)
#    (b) sobre el holdout 2024 real (solo 3 componentes -> CI muy ancho,
#        se reporta igual mostrando la limitación de tamaño de muestra)
# =====================================================================
print("\n" + "=" * 78)
print(f"4) Bootstrap CI por componente (estudiante+curso), B={N_BOOTSTRAP}")
print("=" * 78)

rng = np.random.default_rng(SEED)


def bootstrap_ci_by_component(y_true_df, y_pred_df, component_ids, targets, B=N_BOOTSTRAP, alpha=ALPHA):
    comp_arr = component_ids.to_numpy()
    uniq_comp = np.unique(comp_arr)
    # índice de filas por componente, para resamplear componentes completos
    idx_by_comp = {c: np.where(comp_arr == c)[0] for c in uniq_comp}
    results = {t: {"R2": [], "MAE": [], "RMSE": []} for t in targets}
    y_true_mat = y_true_df[targets].to_numpy()
    y_pred_mat = y_pred_df[targets].to_numpy()

    for _ in range(B):
        sample_comps = rng.choice(uniq_comp, size=len(uniq_comp), replace=True)
        rows = np.concatenate([idx_by_comp[c] for c in sample_comps])
        yt = y_true_mat[rows]
        yp = y_pred_mat[rows]
        for i, t in enumerate(targets):
            yt_t, yp_t = yt[:, i], yp[:, i]
            ss_res = np.sum((yt_t - yp_t) ** 2)
            ss_tot = np.sum((yt_t - yt_t.mean()) ** 2)
            results[t]["R2"].append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
            results[t]["MAE"].append(float(np.mean(np.abs(yt_t - yp_t))))
            results[t]["RMSE"].append(_rmse(yt_t, yp_t))

    summary = {}
    for t in targets:
        summary[t] = {}
        for metric in ["R2", "MAE", "RMSE"]:
            vals = np.array(results[t][metric])
            vals = vals[np.isfinite(vals)]
            summary[t][metric] = {
                "mean": float(np.mean(vals)),
                "ci_lo": float(np.quantile(vals, alpha / 2)),
                "ci_hi": float(np.quantile(vals, 1 - alpha / 2)),
            }
    return summary


bootstrap_results = {}
for model_name in MODELS:
    oof = oof_preds[model_name]
    bootstrap_results[f"{model_name}__CV_OOF_le2023_n{train_pool.shape[0]}_comp{train_pool['component_id'].nunique()}"] = (
        bootstrap_ci_by_component(train_pool[TARGETS], oof, train_pool["component_id"], TARGETS)
    )
    pred_holdout = final_preds[model_name]
    bootstrap_results[f"{model_name}__holdout_2024_n{holdout_2024.shape[0]}_comp{holdout_2024['component_id'].nunique()}"] = (
        bootstrap_ci_by_component(holdout_2024[TARGETS], pred_holdout, holdout_2024["component_id"], TARGETS)
    )

for key, res in bootstrap_results.items():
    print(f"\n--- {key} ---")
    for t, metrics in res.items():
        print(f"  {t:>26s} | R2={metrics['R2']['mean']:.3f} [{metrics['R2']['ci_lo']:.3f},{metrics['R2']['ci_hi']:.3f}]"
              f" | MAE={metrics['MAE']['mean']:.4f} [{metrics['MAE']['ci_lo']:.4f},{metrics['MAE']['ci_hi']:.4f}]"
              f" | RMSE={metrics['RMSE']['mean']:.4f} [{metrics['RMSE']['ci_lo']:.4f},{metrics['RMSE']['ci_hi']:.4f}]")

# =====================================================================
# Guardado
# =====================================================================
all_results = {
    "n_train_pool_rows": int(train_pool.shape[0]),
    "n_train_pool_components": int(train_pool["component_id"].nunique()),
    "n_holdout_2024_rows": int(holdout_2024.shape[0]),
    "n_holdout_2024_components": int(holdout_2024["component_id"].nunique()),
    "before_after_leakage_rf_holdout2024": before_after,
    "cv_grouped_le2023_clean_features": cv_summary,
    "final_model_holdout_2024_clean_features": final_summary,
    "bootstrap_ci_by_component": bootstrap_results,
}
with open(os.path.join(OUT_DIR, "05_baselines_grouped_temporal_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nGuardado: {os.path.join(OUT_DIR, '05_baselines_grouped_temporal_results.json')}")
