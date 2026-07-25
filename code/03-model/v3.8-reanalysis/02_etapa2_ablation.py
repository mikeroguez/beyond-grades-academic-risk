"""
v3.8 reanálisis — Etapa 2: ¿las 3 palancas (accesos/foros truncados,
features genuinamente tempranas, selección Pearson+LassoCV por corte)
mejoran la curva de la Etapa 1?

Prioriza el corte 50% (el más relevante operativamente para alertas) con
la matriz de ablación completa; para 25%/75% solo corre la configuración
"todo junto" (con reselección), para actualizar la curva completa sin
disparar el costo de cómputo.

Todas las features nuevas ya pasaron la auditoría de fuga de
`06_audit_new_features.py` (sin alertas, corte 50%) antes de usarse aquí.

Ninguna de las 8 columnas con fuga original se reintroduce en ningún
punto (no forman parte de ninguno de los pools de candidatas nuevas).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("V38_REANALYSIS_OUT_DIR", os.path.join(HERE, "out"))
os.makedirs(OUT_DIR, exist_ok=True)
V38_OUT = os.environ.get("V38_OUT_DIR", os.path.join(HERE, "..", "..", "v3.8", "out"))
V36_REANALYSIS_OUT = os.path.join(HERE, "..", "v3.6-reanalysis", "out")

sys.path.insert(0, os.path.join(HERE, ".."))
from shared_clean_features import CLEAN_FEATURES  # noqa: E402

SEED = 42
N_FOLDS = 5
TARGETS = ["avg_assignment_score", "missing_assignments", "assignment_procrast_rate", "avg_exam_score", "exam_accuracy"]
ACCESS_FORUM_COLS = [
    "total_access_time", "avg_access_time", "access_days", "access_sessions_desktop",
    "access_sessions_mobile", "access_imputed_desktop", "access_imputed_mobile",
    "total_access_sessions", "total_access_imputed", "access_imputed_ratio",
    "access_missing_data", "forum_interactions", "forum_threads", "forum_time_range",
    "forum_missing_data",
]
EARLY_FEATURE_COLS = [
    "assignment_grade_trend_slope", "days_since_last_engagement", "submission_pace_per_week",
    "exam_attempts_pace_per_week", "relative_position_avg_score", "relative_position_submit_rate",
    "exam_score_trend_slope",
]


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def metrics_per_target(y_true_df, y_pred_df):
    rows = []
    for t in TARGETS:
        yt = y_true_df[t].to_numpy()
        yp = y_pred_df[t].to_numpy()
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rows.append({"target": t, "R2": r2, "MAE": float(np.mean(np.abs(yt - yp))), "RMSE": _rmse(yt, yp)})
    return pd.DataFrame(rows)


def aggregate_global(per_target_df):
    return {
        "MAE": float(per_target_df["MAE"].mean()),
        "RMSE": float(per_target_df["RMSE"].mean()),
        "R2": float(per_target_df["R2"].mean()),
    }


def run_pipeline_on_split(X_train_raw, y_train_df, X_val_raw, feature_set, model_factory):
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw[feature_set]))
    X_val = scaler.transform(imputer.transform(X_val_raw[feature_set]))
    preds = {}
    for t in TARGETS:
        est = model_factory()
        est.fit(X_train, y_train_df[t].to_numpy())
        preds[t] = est.predict(X_val)
    return pd.DataFrame(preds)


def run_rf_cv(df, feature_set, label):
    train_pool = df[df["split_bucket"] == "train_le2023"].reset_index(drop=True)
    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = train_pool["component_id"]
    fold_metrics = []
    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(train_pool, groups=groups)):
        X_tr, X_va = train_pool.iloc[tr_idx], train_pool.iloc[va_idx]
        y_tr, y_va = train_pool.iloc[tr_idx][TARGETS], train_pool.iloc[va_idx][TARGETS]
        pred_df = run_pipeline_on_split(
            X_tr, y_tr, X_va, feature_set, lambda: RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        )
        pred_df.index = X_va.index
        m = metrics_per_target(y_va, pred_df)
        m["fold"] = fold_i
        fold_metrics.append(m)
    all_folds = pd.concat(fold_metrics, ignore_index=True)
    per_target = all_folds.groupby("target")[["R2", "MAE", "RMSE"]].mean().reindex(TARGETS).reset_index()
    global_metrics = aggregate_global(per_target)
    print(f"\n[{label}] ({len(feature_set)} features) Global R²={global_metrics['R2']:.4f} "
          f"MAE={global_metrics['MAE']:.4f} RMSE={global_metrics['RMSE']:.4f}")
    pt_str = " | ".join(f"{row.target}={row.R2:.3f}" for row in per_target.itertuples())
    print(f"    por target: {pt_str}")
    return {"per_target": per_target.to_dict(orient="records"), "global": global_metrics}


def reselect_pearson_lasso(df, candidate_features, corr_threshold=0.10):
    """Selección Pearson (umbral fijo, más barato que buscar por CV dado
    que ya se hizo esa búsqueda en v3.6/v3.7 y los umbrales rondaban
    0.10-0.17) + LassoCV, ajustada SOLO con el pool de entrenamiento
    ≤2023 (mismo patrón que v3.6-reanalysis/04_feature_selection_train_only.py).
    Devuelve la unión de features con coeficiente Lasso != 0 en al menos
    un target."""
    train_pool = df[df["split_bucket"] == "train_le2023"].copy()
    groups = train_pool["component_id"]

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(train_pool[candidate_features]), columns=candidate_features, index=train_pool.index)
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=candidate_features, index=train_pool.index)

    selected_union = set()
    per_target_selection = {}
    for target in TARGETS:
        y = train_pool[target]
        corrs = X_scaled.corrwith(y).abs()
        pearson_pass = corrs[corrs >= corr_threshold].index.tolist()
        if not pearson_pass:
            per_target_selection[target] = []
            continue
        gkf = GroupKFold(n_splits=N_FOLDS)
        lasso = LassoCV(cv=list(gkf.split(X_scaled[pearson_pass], y, groups=groups)), random_state=SEED, n_jobs=-1, max_iter=10000)
        lasso.fit(X_scaled[pearson_pass], y)
        coefs = pd.Series(lasso.coef_, index=pearson_pass)
        lasso_pass = coefs[coefs.abs() > 1e-6].index.tolist()
        per_target_selection[target] = lasso_pass
        selected_union.update(lasso_pass)

    return sorted(selected_union), per_target_selection


if __name__ == "__main__":
    v36_clean = pd.read_csv(os.path.join(V36_REANALYSIS_OUT, "clean_dataset.csv"))
    split_info = v36_clean[["uid_hash", "course_hash", "component_id", "split_bucket"] + TARGETS]

    all_results = {}

    # =================================================================
    # CORTE 50% -- matriz de ablación completa
    # =================================================================
    pct = 50
    feats = pd.read_csv(os.path.join(V38_OUT, f"cutoff_{pct}_features.csv"))
    af = pd.read_csv(os.path.join(V38_OUT, f"cutoff_{pct}_access_forum.csv"))
    ef = pd.read_csv(os.path.join(V38_OUT, f"cutoff_{pct}_early_features.csv"))

    df50 = split_info.merge(feats, on=["uid_hash", "course_hash"], how="left")
    df50 = df50.merge(af, on=["uid_hash", "course_hash"], how="left")
    df50 = df50.merge(ef, on=["uid_hash", "course_hash"], how="left")
    print(f"Dataset corte 50% (todas las columnas candidatas): {df50.shape}")

    configs_50 = {
        "a_base_solo_truncadas (Etapa 1)": CLEAN_FEATURES,
        "b_base_mas_accesos_foros": CLEAN_FEATURES + ACCESS_FORUM_COLS,
        "c_base_mas_tempranas": CLEAN_FEATURES + EARLY_FEATURE_COLS,
        "d_todo_junto_sin_reseleccion": CLEAN_FEATURES + ACCESS_FORUM_COLS + EARLY_FEATURE_COLS,
    }

    results_50 = {}
    for label, feature_set in configs_50.items():
        results_50[label] = run_rf_cv(df50, feature_set, f"corte 50% -- {label}")

    all_candidates = CLEAN_FEATURES + ACCESS_FORUM_COLS + EARLY_FEATURE_COLS
    reselected, per_target_sel = reselect_pearson_lasso(df50, all_candidates)
    print(f"\n[Reselección Pearson+LassoCV, corte 50%, train-only] {len(reselected)}/{len(all_candidates)} features: {reselected}")
    results_50["e_reseleccion_pearson_lasso"] = run_rf_cv(df50, reselected, "corte 50% -- e_reseleccion_pearson_lasso")

    all_results["cutoff_50_ablation"] = results_50
    all_results["cutoff_50_reselection"] = {"selected_features": reselected, "per_target_selection": per_target_sel}

    # =================================================================
    # CORTES 25% y 75% -- solo "todo junto" + reselección (para la curva)
    # =================================================================
    curve_combined = {}
    for pct in (25, 75):
        feats = pd.read_csv(os.path.join(V38_OUT, f"cutoff_{pct}_features.csv"))
        af = pd.read_csv(os.path.join(V38_OUT, f"cutoff_{pct}_access_forum.csv"))
        ef = pd.read_csv(os.path.join(V38_OUT, f"cutoff_{pct}_early_features.csv"))
        dfc = split_info.merge(feats, on=["uid_hash", "course_hash"], how="left")
        dfc = dfc.merge(af, on=["uid_hash", "course_hash"], how="left")
        dfc = dfc.merge(ef, on=["uid_hash", "course_hash"], how="left")

        reselected_c, per_target_sel_c = reselect_pearson_lasso(dfc, all_candidates)
        print(f"\n[Reselección Pearson+LassoCV, corte {pct}%] {len(reselected_c)}/{len(all_candidates)} features")
        res = run_rf_cv(dfc, reselected_c, f"corte {pct}% -- todo junto + reselección")
        curve_combined[f"cutoff_{pct}"] = {"selected_features": reselected_c, "result": res}

    curve_combined["cutoff_50"] = {"selected_features": reselected, "result": results_50["e_reseleccion_pearson_lasso"]}
    all_results["curve_combined_with_reselection"] = curve_combined

    # ------------------------------------------------------------------
    # Curva final comparativa: Etapa 1 (18 features) vs Etapa 2 (todo
    # junto + reselección por corte), en los 3 cortes + 100% (v3.6, citado)
    # ------------------------------------------------------------------
    with open(os.path.join(HERE, "out", "01_baselines_by_cutoff_results.json")) as f:
        etapa1 = json.load(f)
    etapa1_curve = etapa1["performance_vs_cutoff_curve_randomforest"]["global"]

    print(f"\n{'=' * 90}\nCURVA COMPARATIVA -- RandomForest, CV agrupada <=2023, Global R² (aritmético)\n{'=' * 90}")
    print(f"{'Corte':<15s}{'Etapa 1 (18 feat.)':>22s}{'Etapa 2 (todo+reselección)':>28s}{'Delta':>10s}")
    final_curve = {}
    for pct_key, pct_label in [("25", "25%"), ("50", "50%"), ("75", "75%")]:
        e1 = etapa1_curve[pct_key]
        e2 = curve_combined[f"cutoff_{pct_key}"]["result"]["global"]["R2"]
        final_curve[pct_label] = {"etapa1": e1, "etapa2": e2, "delta": e2 - e1}
        print(f"{pct_label:<15s}{e1:>22.4f}{e2:>28.4f}{e2 - e1:>10.4f}")
    e1_100 = etapa1_curve["100_v36"]
    final_curve["100% (v3.6)"] = {"etapa1": e1_100, "etapa2": None, "delta": None}
    print(f"{'100% (v3.6)':<15s}{e1_100:>22.4f}{'--':>28s}")

    all_results["final_curve_comparison"] = final_curve

    with open(os.path.join(OUT_DIR, "02_etapa2_ablation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nGuardado: {os.path.join(OUT_DIR, '02_etapa2_ablation_results.json')}")
