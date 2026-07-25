"""
WP1 — Paso 1: Confirmación empírica de fuga de datos (Reviewer 1 #1-4).

No existe `versión 3/AUDITORIA-LEAKAGE.md` en este repo al momento de correr
este script (se verificó con `find`), pese a que el despacho de esta tarea
lo describe como insumo ya cerrado. Como sí hay acceso real a los datos
(`code/v3.6/data/`), este script reconstruye la verificación empírica por su
cuenta, en vez de asumir cifras que no se pueden trazar a un archivo real.

Confirma dos cosas para cada fuga sospechosa:
  (a) identidad algebraica exacta (o casi) entre target y un subconjunto de
      predictores, leyendo directamente el código fuente de consolidación
      (01-preprocessing/tareas/02_consolida_tareas.py,
       01-preprocessing/examenes/05_consolida_examenes.py);
  (b) una prueba estadística de reconstrucción con Random Forest: R² de
      predecir el target SOLO con los predictores sospechosos, y R² de un
      modelo con el resto de las features SIN los predictores sospechosos
      (para ver si el target retiene señal genuina una vez removida la fuga).

Input:  code/v3.6/data/2. Para unir/Limpieza/step_1_dataset.csv (4230x58,
        columnas sin escalar, nombres pre-rename — es el punto más limpio
        del pipeline para trabajar con las columnas crudas de tareas/exámenes)
Output: stdout + code/03-model/v3.6-reanalysis/out/01_leakage_confirmation.json
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
V36_DATA = os.path.join(HERE, "..", "..", "v3.6", "data")
OUT_DIR = os.path.join(HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)

STEP1_PATH = os.path.join(V36_DATA, "2. Para unir", "Limpieza", "step_1_dataset.csv")

df = pd.read_csv(STEP1_PATH)
print(f"step_1_dataset.csv: {df.shape}")

# Grupo compuesto (curso+email) para que la evaluación de reconstrucción no
# se beneficie de que un mismo estudiante-curso aparezca en train y val.
groups = df["curso"].astype(str) + "__" + df["email"].astype(str)

report = {"algebraic_identities": {}, "rf_reconstruction": {}}

# ---------------------------------------------------------------------
# 1) Identidades algebraicas (lectura directa del código de consolidación,
#    confirmadas aquí numéricamente sobre los datos reales)
# ---------------------------------------------------------------------
print("\n=== 1) Identidades algebraicas ===")

# missing = total - submitted (exacto, ver 02_consolida_tareas.py línea ~35)
resid = df["total_missing_assignments"] - (df["total_assignments"] - df["total_submitted_assignments"])
max_abs_err = resid.abs().max()
report["algebraic_identities"]["missing_assignments = total_assignments - submitted_assignments"] = {
    "max_abs_error": float(max_abs_err),
    "exact": bool(max_abs_err < 1e-9),
}
print(f"missing_assignments == total_assignments - submitted_assignments : max|err| = {max_abs_err:.2e}")

# procrastination_rate = late / submitted (exacto, ver 02_consolida_tareas.py)
mask = df["total_submitted_assignments"] > 0
resid2 = (
    df.loc[mask, "assignment_procrastination_rate"]
    - (df.loc[mask, "total_late_assignments"] / df.loc[mask, "total_submitted_assignments"])
)
max_abs_err2 = resid2.abs().max()
report["algebraic_identities"]["assignment_procrast_rate = late_assignments / submitted_assignments"] = {
    "max_abs_error": float(max_abs_err2),
    "exact": bool(max_abs_err2 < 1e-9),
}
print(f"assignment_procrast_rate == late_assignments / submitted_assignments : max|err| = {max_abs_err2:.2e}")

# accuracy_rate = correct / questions (exacto, ver 05_consolida_examenes.py)
mask3 = df["total_questions_answered"] > 0
resid3 = (
    df.loc[mask3, "accuracy_rate"]
    - (df.loc[mask3, "total_correct_answers"] / df.loc[mask3, "total_questions_answered"])
)
max_abs_err3 = resid3.abs().max()
report["algebraic_identities"]["exam_accuracy = exam_correct_answers / exam_questions"] = {
    "max_abs_error": float(max_abs_err3),
    "exact": bool(max_abs_err3 < 1e-9),
}
print(f"exam_accuracy == exam_correct_answers / exam_questions : max|err| = {max_abs_err3:.2e}")

# ---------------------------------------------------------------------
# 2) Prueba de reconstrucción con Random Forest (GroupKFold por curso+email)
# ---------------------------------------------------------------------
print("\n=== 2) Reconstrucción con Random Forest (5-fold GroupKFold) ===")

TARGETS_RAW = {
    "missing_assignments": "total_missing_assignments",
    "assignment_procrast_rate": "assignment_procrastination_rate",
    "exam_accuracy": "accuracy_rate",
    "avg_assignment_score": "avg_assignment_normalized_score",
    "avg_exam_score": "avg_exam_normalized_score",
}

LEAKY_PREDICTORS_RAW = {
    "missing_assignments": ["total_assignments", "total_submitted_assignments"],
    "assignment_procrast_rate": ["total_late_assignments", "total_submitted_assignments"],
    "exam_accuracy": ["total_correct_answers", "total_questions_answered"],
}

# columnas meta que nunca deben ser predictores
META_COLS = ["curso", "email"]
ALL_TARGET_RAW_COLS = list(TARGETS_RAW.values())

numeric_cols = [c for c in df.columns if c not in META_COLS and df[c].dtype != object]


def rf_r2_groupkfold(X: pd.DataFrame, y: pd.Series, groups: pd.Series, seed=42) -> float:
    gkf = GroupKFold(n_splits=5)
    scores = []
    X_ = X.fillna(X.median(numeric_only=True))
    for train_idx, val_idx in gkf.split(X_, y, groups=groups):
        rf = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1, max_depth=None)
        rf.fit(X_.iloc[train_idx], y.iloc[train_idx])
        pred = rf.predict(X_.iloc[val_idx])
        ss_res = np.sum((y.iloc[val_idx].values - pred) ** 2)
        ss_tot = np.sum((y.iloc[val_idx].values - y.iloc[val_idx].mean()) ** 2)
        scores.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
    return float(np.mean(scores))


for tgt_final, tgt_raw in TARGETS_RAW.items():
    y = df[tgt_raw]

    # (a) reconstrucción SOLO desde los predictores sospechosos (si existen para este target)
    leaky_preds = LEAKY_PREDICTORS_RAW.get(tgt_final)
    r2_from_leaky_only = None
    if leaky_preds:
        X_leaky = df[leaky_preds]
        r2_from_leaky_only = rf_r2_groupkfold(X_leaky, y, groups)

    # (b) R² con TODO el resto de las features (incluye los sospechosos) -> "antes"
    other_cols_incl = [c for c in numeric_cols if c not in ALL_TARGET_RAW_COLS]
    r2_all_incl_leaky = rf_r2_groupkfold(df[other_cols_incl], y, groups)

    # (c) R² con el resto de las features SIN los predictores sospechosos -> "después" (limpio)
    excl = leaky_preds or []
    other_cols_excl = [c for c in other_cols_incl if c not in excl]
    r2_all_excl_leaky = rf_r2_groupkfold(df[other_cols_excl], y, groups)

    report["rf_reconstruction"][tgt_final] = {
        "leaky_predictors": leaky_preds,
        "r2_from_leaky_predictors_only": r2_from_leaky_only,
        "r2_all_features_incl_leaky": r2_all_incl_leaky,
        "r2_all_features_excl_leaky": r2_all_excl_leaky,
    }
    print(f"\n[{tgt_final}]")
    print(f"  predictores sospechosos: {leaky_preds}")
    if r2_from_leaky_only is not None:
        print(f"  R² reconstruyendo SOLO desde predictores sospechosos: {r2_from_leaky_only:.4f}")
    print(f"  R² con TODAS las features (incl. sospechosas): {r2_all_incl_leaky:.4f}")
    print(f"  R² con el resto de features (SIN las sospechosas): {r2_all_excl_leaky:.4f}")

with open(os.path.join(OUT_DIR, "01_leakage_confirmation.json"), "w") as f:
    json.dump(report, f, indent=2)
print(f"\nGuardado: {os.path.join(OUT_DIR, '01_leakage_confirmation.json')}")
