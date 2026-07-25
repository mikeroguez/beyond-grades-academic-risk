"""
WP1 — Análisis de sensibilidad de la fuga PARCIAL de `avg_exam_score`
(pedido de Miguel, prioridad máxima de su revisión editorial
`code/v3.8/my-review-v3.8-1st.txt` #1).

Hallazgo ya documentado (METODOLOGIA-V3.md §2): 25.4% de las 4230 filas
(1074) tienen `avg_exam_score == min_exam_score == max_exam_score`
exactamente. `avg_exam_score` es el target con mejor desempeño reportado
(RF R²=0.975, Sección 6.2 de METODOLOGIA-V3.md) y por tanto el que más
sostiene el promedio global -- si ese R² está inflado por filas triviales
(un solo intento de examen, donde avg=min=max por construcción) y por
tener min/max/var de examen como predictores del propio promedio, es el
hallazgo que más puede dañar la credibilidad del reanálisis si lo
encuentra un revisor en vez de nosotros.

⚠️ Precisión sobre el criterio "≥2 intentos" (verificado antes de asumir):
las 1074 filas con avg=min=max NO son exactamente lo mismo que "un solo
intento de examen". Se verificó cruzando contra `total_assigned_exams`
(conteo crudo, `step_1_dataset.csv`): de esas 1074 filas, 821 tienen
`total_assigned_exams<=1` (un único intento, la causa estructural
esperada) y 253 tienen `total_assigned_exams>1` pero puntajes idénticos en
todos sus intentos (menos trivial, pero también hace que min=avg=max).
El criterio operacional correcto y más principled para "≥2 intentos
genuinos" es `total_assigned_exams>=2` (3409 filas), no simplemente las
1074 filas de avg=min=max -- se usa ese criterio aquí.

Tres configuraciones, mismo protocolo (RF, split agrupado por componente +
holdout temporal 2024, agregación aritmética) que el resto de WP1:
  (baseline) Todas las filas, las 18 features limpias -- referencia.
  (a) Solo filas con >=2 intentos de examen (excluye 821 de intento único),
      18 features limpias.
  (b) Todas las filas, sin min_exam_score/max_exam_score/exam_score_var
      como predictores de avg_exam_score (15 features).
  (c) (a) + (b) combinados: >=2 intentos Y sin esos 3 predictores.
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
TARGET = "avg_exam_score"

CLEAN_18_FEATURES = [
    "completed_exams", "exam_incidents", "max_assignment_score", "avg_exam_incidents",
    "total_exams", "exam_score_var", "all_exams", "five%_or_less_incomplete_exams",
    "perfect_exams", "min_assignment_score", "min_exam_time", "assignment_score_var",
    "max_exam_score", "incomplete_exams", "exam_submit_rate", "min_exam_score",
    "avg_assignment_delay", "outlier_exams_course",
]
SUSPECT_PREDICTORS = ["min_exam_score", "max_exam_score", "exam_score_var"]
FEATURES_WITHOUT_SUSPECTS = [c for c in CLEAN_18_FEATURES if c not in SUSPECT_PREDICTORS]

# ---------------------------------------------------------------------
# 0) Verificación del subconjunto ">=2 intentos" contra el conteo crudo
# ---------------------------------------------------------------------
step1 = pd.read_csv(os.path.join(HERE, "..", "..", "v3.6", "data", "2. Para unir", "Limpieza", "step_1_dataset.csv"))
step1 = step1.rename(columns={"curso": "course_hash", "email": "uid_hash"})

v36_clean = pd.read_csv(os.path.join(OUT_DIR, "clean_dataset.csv"))
df = v36_clean.merge(step1[["uid_hash", "course_hash", "total_assigned_exams"]], on=["uid_hash", "course_hash"], how="left")

same_avg_min_max = (df["avg_exam_score"] == df["min_exam_score"]) & (df["avg_exam_score"] == df["max_exam_score"])
single_attempt = df["total_assigned_exams"] <= 1
print(f"Filas con avg_exam_score == min == max: {same_avg_min_max.sum()} / {len(df)} ({same_avg_min_max.mean() * 100:.1f}%)")
print(f"Filas con total_assigned_exams <= 1 (un solo intento genuino): {single_attempt.sum()} / {len(df)} ({single_attempt.mean() * 100:.1f}%)")
print(f"De las {same_avg_min_max.sum()} filas avg=min=max, {(same_avg_min_max & single_attempt).sum()} tienen "
      f"1 intento y {(same_avg_min_max & ~single_attempt).sum()} tienen >1 intento con puntajes idénticos.")
print(f"Criterio usado en este análisis: total_assigned_exams >= 2 -> {(~single_attempt).sum()} filas retenidas "
      f"de {len(df)} ({(~single_attempt).mean() * 100:.1f}%).")

df["has_ge2_attempts"] = ~single_attempt


# ---------------------------------------------------------------------
# Protocolo RF estándar (idéntico al resto de WP1)
# ---------------------------------------------------------------------
def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def fit_predict(X_train, y_train, X_val, seed=SEED):
    rf = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf.predict(X_val)


def preprocess(X_train_raw, X_val_raw, feature_set):
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw[feature_set]))
    X_val = scaler.transform(imputer.transform(X_val_raw[feature_set]))
    return X_train, X_val


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def run_config(df_subset, feature_set, label):
    train_pool = df_subset[df_subset["split_bucket"] == "train_le2023"].reset_index(drop=True)
    holdout = df_subset[df_subset["split_bucket"] == "holdout_2024"].reset_index(drop=True)

    n_components = train_pool["component_id"].nunique()
    print(f"\n[{label}] train_le2023: {len(train_pool)} filas / {n_components} componentes | "
          f"holdout_2024: {len(holdout)} filas")

    if n_components < N_FOLDS:
        print(f"[{label}] ADVERTENCIA: menos de {N_FOLDS} componentes disponibles -- se omite CV agrupada para esta configuración.")
        cv_r2_mean, cv_r2_std, fold_r2s = None, None, []
    else:
        gkf = GroupKFold(n_splits=N_FOLDS)
        groups = train_pool["component_id"]
        fold_r2s = []
        for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(train_pool, groups=groups)):
            X_tr_raw, X_va_raw = train_pool.iloc[tr_idx], train_pool.iloc[va_idx]
            y_tr, y_va = train_pool.iloc[tr_idx][TARGET].to_numpy(), train_pool.iloc[va_idx][TARGET].to_numpy()
            X_tr, X_va = preprocess(X_tr_raw, X_va_raw, feature_set)
            pred = fit_predict(X_tr, y_tr, X_va)
            fold_r2s.append(r2(y_va, pred))
        cv_r2_mean = float(np.mean(fold_r2s))
        cv_r2_std = float(np.std(fold_r2s))
        print(f"[{label}] CV agrupada (<=2023, {N_FOLDS} folds): R2 por fold = {[round(x, 3) for x in fold_r2s]}")
        print(f"[{label}] CV agrupada R2 media = {cv_r2_mean:.4f} (std={cv_r2_std:.4f})")

    # Modelo final: ajustado en todo el pool de train de esta configuración,
    # evaluado en el holdout 2024 de esta misma configuración (si tiene filas).
    if len(holdout) >= 5:
        X_tr, X_ho = preprocess(train_pool, holdout, feature_set)
        pred_ho = fit_predict(X_tr, train_pool[TARGET].to_numpy(), X_ho)
        holdout_r2 = r2(holdout[TARGET].to_numpy(), pred_ho)
        print(f"[{label}] Holdout 2024 ({len(holdout)} filas): R2 = {holdout_r2:.4f}")
    else:
        holdout_r2 = None
        print(f"[{label}] Holdout 2024: muy pocas filas ({len(holdout)}) tras el filtro -- se omite.")

    return {
        "label": label,
        "n_features": len(feature_set),
        "features": feature_set,
        "n_rows_total": int(len(df_subset)),
        "n_rows_train": int(len(train_pool)),
        "n_rows_holdout": int(len(holdout)),
        "n_components_train": int(n_components),
        "cv_fold_r2": fold_r2s,
        "cv_r2_mean": cv_r2_mean,
        "cv_r2_std": cv_r2_std,
        "holdout_r2": holdout_r2,
    }


results = {}

results["baseline_all_rows_18features"] = run_config(df, CLEAN_18_FEATURES, "BASELINE: todas las filas, 18 features")

df_ge2 = df[df["has_ge2_attempts"]].copy()
results["a_subset_ge2_attempts_18features"] = run_config(
    df_ge2, CLEAN_18_FEATURES, "(a) Solo >=2 intentos, 18 features"
)

results["b_all_rows_without_suspects"] = run_config(
    df, FEATURES_WITHOUT_SUSPECTS, "(b) Todas las filas, SIN min/max/var de examen (15 features)"
)

results["c_subset_ge2_without_suspects"] = run_config(
    df_ge2, FEATURES_WITHOUT_SUSPECTS, "(c) >=2 intentos Y SIN min/max/var de examen (15 features)"
)

print("\n" + "=" * 90)
print("RESUMEN COMPARATIVO -- avg_exam_score, RandomForest, CV agrupada <=2023")
print("=" * 90)
print(f"{'Configuración':<55s}{'n filas':>10s}{'CV R2':>10s}{'Holdout R2':>12s}")
for key, r in results.items():
    cv_str = f"{r['cv_r2_mean']:.4f}" if r["cv_r2_mean"] is not None else "N/A"
    ho_str = f"{r['holdout_r2']:.4f}" if r["holdout_r2"] is not None else "N/A"
    print(f"{r['label']:<55s}{r['n_rows_total']:>10d}{cv_str:>10s}{ho_str:>12s}")

output = {
    "same_avg_min_max_count": int(same_avg_min_max.sum()),
    "single_attempt_count_total_assigned_exams_le1": int(single_attempt.sum()),
    "overlap_same_and_single": int((same_avg_min_max & single_attempt).sum()),
    "same_but_multi_attempt_identical_scores": int((same_avg_min_max & ~single_attempt).sum()),
    "results": results,
}
with open(os.path.join(OUT_DIR, "07_avg_exam_score_sensitivity_results.json"), "w") as f:
    json.dump(output, f, indent=2)
print(f"\nGuardado: {os.path.join(OUT_DIR, '07_avg_exam_score_sensitivity_results.json')}")
