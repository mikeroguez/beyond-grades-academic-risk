"""
v3.8 reanálisis — Etapa 1: curva desempeño-vs-momento-de-corte.

Para cada corte (25%, 50%, 75% de la ventana de curso, Sección `code/v3.8/`)
se construye el dataset limpio (features de `shared_clean_features.CLEAN_FEATURES`
recalculadas SOLO con eventos hasta ese corte + los mismos 5 targets de fin
de curso de siempre) y se corre el MISMO protocolo de
`v3.6-reanalysis`/`v3.7-reanalysis`: split agrupado por componente
estudiante+curso, CV de 5 folds dentro de ≤2023, holdout temporal 2024,
agregación aritmética.

El 100% (curso completo) NO se recalcula -- es exactamente
`v3.6-reanalysis` y se lee de `05_baselines_grouped_temporal_results.json`
tal cual (no se hardcodea el número aquí -- ver el assert de sanidad más
abajo) para construir la curva completa 25/50/75/100%.

Solo baselines (LR/DT/RF) -- nada de DNN/SHAP en esta etapa.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor

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

MODELS = {
    "LinearRegression": lambda: LinearRegression(),
    "DecisionTree": lambda: DecisionTreeRegressor(random_state=SEED, max_depth=8, min_samples_leaf=5),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
}


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def fit_predict_per_target(X_train, y_train_df, X_val, model_factory):
    preds = {}
    for t in TARGETS:
        est = model_factory()
        est.fit(X_train, y_train_df[t].to_numpy())
        preds[t] = est.predict(X_val)
    return pd.DataFrame(preds)


def run_pipeline_on_split(X_train_raw, y_train_df, X_val_raw, feature_set):
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw[feature_set]))
    X_val = scaler.transform(imputer.transform(X_val_raw[feature_set]))
    out = {}
    for model_name, factory in MODELS.items():
        out[model_name] = fit_predict_per_target(X_train, y_train_df, X_val, factory)
    return out


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
        "MAE_arithmetic_mean": float(per_target_df["MAE"].mean()),
        "RMSE_arithmetic_mean": float(per_target_df["RMSE"].mean()),
        "R2_arithmetic_mean": float(per_target_df["R2"].mean()),
    }


def run_cv_and_holdout(df, feature_set, label):
    train_pool = df[df["split_bucket"] == "train_le2023"].reset_index(drop=True)
    holdout_2024 = df[df["split_bucket"] == "holdout_2024"].reset_index(drop=True)

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = train_pool["component_id"]

    fold_metrics = {m: [] for m in MODELS}
    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(train_pool, groups=groups)):
        X_tr, X_va = train_pool.iloc[tr_idx], train_pool.iloc[va_idx]
        y_tr, y_va = train_pool.iloc[tr_idx][TARGETS], train_pool.iloc[va_idx][TARGETS]
        preds = run_pipeline_on_split(X_tr, y_tr, X_va, feature_set)
        for model_name, pred_df in preds.items():
            pred_df.index = X_va.index
            m = metrics_per_target(y_va, pred_df)
            m["fold"] = fold_i
            fold_metrics[model_name].append(m)

    cv_summary = {}
    for model_name in MODELS:
        all_folds = pd.concat(fold_metrics[model_name], ignore_index=True)
        per_target_cv = all_folds.groupby("target")[["R2", "MAE", "RMSE"]].mean().reindex(TARGETS).reset_index()
        cv_summary[model_name] = {
            "per_target_mean_over_folds": per_target_cv.to_dict(orient="records"),
            "global": aggregate_global(per_target_cv),
        }

    final_preds = run_pipeline_on_split(train_pool, train_pool[TARGETS], holdout_2024, feature_set)
    holdout_summary = {}
    for model_name, pred_df in final_preds.items():
        pred_df.index = holdout_2024.index
        m = metrics_per_target(holdout_2024[TARGETS], pred_df)
        holdout_summary[model_name] = {"per_target": m.to_dict(orient="records"), "global": aggregate_global(m)}

    print(f"\n{'=' * 90}\n{label}\n{'=' * 90}")
    for model_name in MODELS:
        cv_g = cv_summary[model_name]["global"]
        ho_g = holdout_summary[model_name]["global"]
        print(f"[{label}] {model_name:>16s} | CV(<=2023) R2={cv_g['R2_arithmetic_mean']:.4f} "
              f"MAE={cv_g['MAE_arithmetic_mean']:.4f} RMSE={cv_g['RMSE_arithmetic_mean']:.4f}  ||  "
              f"holdout(2024) R2={ho_g['R2_arithmetic_mean']:.4f}")
        cv_pt = cv_summary[model_name]["per_target_mean_over_folds"]
        pt_str = " | ".join(f"{d['target']}={d['R2']:.3f}" for d in cv_pt)
        print(f"    CV por target: {pt_str}")

    return {"cv_grouped_le2023": cv_summary, "holdout_2024": holdout_summary}


if __name__ == "__main__":
    v36_clean = pd.read_csv(os.path.join(V36_REANALYSIS_OUT, "clean_dataset.csv"))
    split_info = v36_clean[["uid_hash", "course_hash", "component_id", "split_bucket"] + TARGETS]

    results_by_cutoff = {}
    for pct in (25, 50, 75):
        feat_path = os.path.join(V38_OUT, f"cutoff_{pct}_features.csv")
        feats = pd.read_csv(feat_path)
        df = split_info.merge(feats, on=["uid_hash", "course_hash"], how="left")
        missing = df[CLEAN_FEATURES[0]].isna().sum()
        print(f"\ncutoff_{pct}: {df.shape}, filas sin match de features: {missing}")

        label = f"v3.8 -- corte {pct}% ({len(CLEAN_FEATURES)} features recalculadas, mismos targets de fin de curso)"
        results_by_cutoff[f"cutoff_{pct}"] = run_cv_and_holdout(df, CLEAN_FEATURES, label)

    # ------------------------------------------------------------------
    # Curva desempeño-vs-corte -- RandomForest, CV agrupada <=2023
    # (100% citado de v3.6-reanalysis, NO recalculado; se lee siempre del
    # JSON en vivo -- no se hardcodea el numero para que este script no
    # quede desalineado la proxima vez que cambie el feature set principal)
    # ------------------------------------------------------------------
    with open(os.path.join(V36_REANALYSIS_OUT, "05_baselines_grouped_temporal_results.json")) as f:
        v36_results = json.load(f)
    v36_rf_per_target = {
        d["target"]: d for d in v36_results["cv_grouped_le2023_clean_features"]["RandomForest"]["per_target_mean_over_folds"]
    }
    v36_rf_global = v36_results["cv_grouped_le2023_clean_features"]["RandomForest"]["global"]["R2_arithmetic_mean"]
    print(f"100% (v3.6-reanalysis, citado en vivo del JSON): RandomForest global R2={v36_rf_global:.4f}")

    print(f"\n{'=' * 90}\nCURVA DESEMPEÑO-VS-CORTE -- RandomForest, CV agrupada <=2023\n{'=' * 90}")
    header = f"{'target':<28s}" + "".join(f"{c:>12s}" for c in ["25%", "50%", "75%", "100%(v3.6)"])
    print(header)
    curve_data = {"per_target": {}, "global": {}}
    for t in TARGETS:
        vals = []
        for p in (25, 50, 75):
            d = {x["target"]: x for x in results_by_cutoff[f"cutoff_{p}"]["cv_grouped_le2023"]["RandomForest"]["per_target_mean_over_folds"]}
            vals.append(d[t]["R2"])
        vals.append(v36_rf_per_target[t]["R2"])
        curve_data["per_target"][t] = {"25": vals[0], "50": vals[1], "75": vals[2], "100_v36": vals[3]}
        print(f"{t:<28s}" + "".join(f"{v:>12.4f}" for v in vals))

    global_vals = [results_by_cutoff[f"cutoff_{p}"]["cv_grouped_le2023"]["RandomForest"]["global"]["R2_arithmetic_mean"] for p in (25, 50, 75)]
    global_vals.append(v36_rf_global)
    curve_data["global"] = {"25": global_vals[0], "50": global_vals[1], "75": global_vals[2], "100_v36": global_vals[3]}
    print(f"{'GLOBAL (aritmético)':<28s}" + "".join(f"{v:>12.4f}" for v in global_vals))

    all_results = {
        "features": CLEAN_FEATURES,
        "results_by_cutoff": results_by_cutoff,
        "v36_100pct_randomforest_cited": {"per_target": v36_rf_per_target, "global": v36_rf_global},
        "performance_vs_cutoff_curve_randomforest": curve_data,
    }
    with open(os.path.join(OUT_DIR, "01_baselines_by_cutoff_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nGuardado: {os.path.join(OUT_DIR, '01_baselines_by_cutoff_results.json')}")
