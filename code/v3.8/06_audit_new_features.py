"""
v3.8 — Etapa 2: auditoría de fuga sobre las features NUEVAS (accesos/foros
truncados + las 7 genuinamente tempranas) antes de darlas por buenas para
el reanálisis. Mismo criterio que `AUDITORIA-LEAKAGE.md` (identidad
algebraica exacta -> fuga dura; R² alto con pocas features -> fuga blanda
a revisar el origen).

Por construcción, ninguna de estas features puede ser un componente
algebraico exacto de los 5 targets (que integran TODO el curso, incluidos
eventos posteriores al corte que estas features nunca ven) -- pero eso no
exime de comprobarlo empíricamente: se corre, para el corte 50% (el
prioritario), un RandomForest de reconstrucción (GroupKFold por
componente) target por target, usando SOLO las features nuevas (sin las
18 limpias), para descartar coincidencias sospechosas.

Umbral de alerta (igual que el propuesto en AUDITORIA-LEAKAGE.md §5):
R²>=0.95 con <=3 features => fuga dura a investigar antes de continuar.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("V38_OUT_DIR", os.path.join(HERE, "out"))
os.makedirs(OUT_DIR, exist_ok=True)
V36_REANALYSIS_OUT = os.path.join(HERE, "..", "03-model", "v3.6-reanalysis", "out")

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
NEW_CANDIDATE_COLS = ACCESS_FORUM_COLS + EARLY_FEATURE_COLS

v36_clean = pd.read_csv(os.path.join(V36_REANALYSIS_OUT, "clean_dataset.csv"))
split_info = v36_clean[["uid_hash", "course_hash", "component_id"] + TARGETS]

pct = 50
af = pd.read_csv(os.path.join(OUT_DIR, f"cutoff_{pct}_access_forum.csv"))
ef = pd.read_csv(os.path.join(OUT_DIR, f"cutoff_{pct}_early_features.csv"))

df = split_info.merge(af, on=["uid_hash", "course_hash"], how="left").merge(ef, on=["uid_hash", "course_hash"], how="left")
print(f"Dataset de auditoría (corte {pct}%): {df.shape}")


def rf_r2_groupkfold(X: pd.DataFrame, y: pd.Series, groups: pd.Series, seed=42) -> float:
    gkf = GroupKFold(n_splits=5)
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    scores = []
    for train_idx, val_idx in gkf.split(X_imp, y, groups=groups):
        rf = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
        rf.fit(X_imp.iloc[train_idx], y.iloc[train_idx])
        pred = rf.predict(X_imp.iloc[val_idx])
        ss_res = np.sum((y.iloc[val_idx].values - pred) ** 2)
        ss_tot = np.sum((y.iloc[val_idx].values - y.iloc[val_idx].mean()) ** 2)
        scores.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
    return float(np.mean(scores))


report = {}
ALERT_THRESHOLD = 0.95
alerts = []

for target in TARGETS:
    y = df[target]
    groups = df["component_id"]

    r2_all_new = rf_r2_groupkfold(df[NEW_CANDIDATE_COLS], y, groups)
    print(f"\n[{target}] R² reconstruyendo SOLO con las {len(NEW_CANDIDATE_COLS)} features nuevas: {r2_all_new:.4f}")

    per_feature = {}
    for col in NEW_CANDIDATE_COLS:
        r2_single = rf_r2_groupkfold(df[[col]], y, groups)
        per_feature[col] = r2_single
        if r2_single >= ALERT_THRESHOLD:
            alerts.append({"target": target, "feature": col, "r2_single": r2_single})

    top5 = sorted(per_feature.items(), key=lambda kv: -kv[1])[:5]
    print(f"  Top 5 individuales: {[(k, round(v, 3)) for k, v in top5]}")

    report[target] = {"r2_all_new_features": r2_all_new, "r2_per_feature": per_feature}

print(f"\n{'=' * 70}\nALERTAS (R² individual >= {ALERT_THRESHOLD}, revisar antes de usar)\n{'=' * 70}")
if alerts:
    for a in alerts:
        print(f"  ⚠️ {a}")
else:
    print("  Ninguna. Ninguna feature nueva reconstruye un target por sí sola con R²>=0.95.")

with open(os.path.join(OUT_DIR, "06_audit_new_features_results.json"), "w") as f:
    json.dump({"cutoff": pct, "report": report, "alerts": alerts}, f, indent=2)
print(f"\nGuardado: {os.path.join(OUT_DIR, '06_audit_new_features_results.json')}")
