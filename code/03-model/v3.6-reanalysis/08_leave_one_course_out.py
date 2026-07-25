"""
WP1 — Leave-one-course-out (Reviewer 1 #8), pedido de Miguel vía su
revisión editorial (`code/v3.8/my-review-v3.8-1st.txt` #6, "Leave-one-course-out
(R1 #8)... siguen sin reportarse").

R1 #8 pide evidencia directa de que el modelo generaliza a cursos nunca
vistos en entrenamiento (no solo a estudiantes nuevos dentro de cursos ya
vistos). El split agrupado por componente (estudiante+curso) de WP1 ya
evita la fuga de curso DENTRO de cada fold de la CV, pero no da,
específicamente, la distribución de desempeño curso por curso que R1 pide
explícitamente ("no solo el promedio").

Diseño: leave-one-course-out EXHAUSTIVO (no muestreado) -- 165 cursos
únicos en el dataset, computacionalmente barato con RandomForest (165
entrenamientos de ~4 segundos cada uno). Para cada curso c: se entrena RF
(300 árboles, 18 features limpias, los 5 targets) con TODAS las filas de
los otros 164 cursos, y se evalúa sobre las filas de c. No se aplica el
filtro temporal ≤2023/2024 aquí -- LOCO evalúa específicamente
generalización entre CURSOS, no entre periodos; ambos ya están cubiertos
por separado en el resto de WP1 (holdout temporal 2024).

R² por curso excluido solo es interpretable cuando ese curso tiene
suficientes filas Y varianza no nula en el target (si todos los
estudiantes de un curso tienen el mismo valor exacto de un target, R² no
está definido) -- se reporta explícitamente cuántos cursos se excluyen de
cada resumen por esta razón, no se ocultan.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
SEED = 42

with open(os.path.join(OUT_DIR, "feature_lists.json")) as f:
    feat_lists = json.load(f)
CLEAN_FEATURES = feat_lists["clean_features"]
TARGETS = feat_lists["targets"]

df = pd.read_csv(os.path.join(OUT_DIR, "clean_dataset.csv"))
courses = sorted(df["course_hash"].unique())
print(f"Cursos únicos: {len(courses)}. Filas totales: {len(df)}.")
print(f"Tamaño de curso: min={df.groupby('course_hash').size().min()}, "
      f"mediana={df.groupby('course_hash').size().median()}, "
      f"max={df.groupby('course_hash').size().max()}")


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return np.nan  # sin varianza en el curso excluido -> R2 no definido
    return float(1 - ss_res / ss_tot)


rows = []
for i, course in enumerate(courses):
    train_df = df[df["course_hash"] != course]
    test_df = df[df["course_hash"] == course]

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(train_df[CLEAN_FEATURES]))
    X_test = scaler.transform(imputer.transform(test_df[CLEAN_FEATURES]))

    row = {"course_hash": course, "n_students": len(test_df)}
    for target in TARGETS:
        rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        rf.fit(X_train, train_df[target].to_numpy())
        pred = rf.predict(X_test)
        row[f"{target}_R2"] = r2(test_df[target].to_numpy(), pred)
        row[f"{target}_MAE"] = float(np.mean(np.abs(test_df[target].to_numpy() - pred)))
    rows.append(row)
    if (i + 1) % 20 == 0 or (i + 1) == len(courses):
        print(f"  {i + 1}/{len(courses)} cursos procesados...")

loco_df = pd.DataFrame(rows)
loco_df.to_csv(os.path.join(OUT_DIR, "08_leave_one_course_out_per_course.csv"), index=False)

print(f"\n{'=' * 90}\nDISTRIBUCIÓN DE R² POR CURSO EXCLUIDO (leave-one-course-out, {len(courses)} cursos)\n{'=' * 90}")
summary = {}
for target in TARGETS:
    col = f"{target}_R2"
    valid = loco_df[col].dropna()
    n_excluded_novar = int(loco_df[col].isna().sum())
    desc = valid.describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    n_negative = int((valid < 0).sum())
    print(f"\n[{target}] cursos con R² definido: {len(valid)}/{len(loco_df)} "
          f"({n_excluded_novar} excluidos por varianza cero en el curso)")
    print(desc.to_string())
    print(f"  cursos con R² < 0: {n_negative}/{len(valid)} ({n_negative / len(valid) * 100:.1f}%)")
    summary[target] = {
        "n_courses_with_defined_r2": int(len(valid)),
        "n_courses_excluded_zero_variance": n_excluded_novar,
        "mean": float(valid.mean()),
        "median": float(valid.median()),
        "std": float(valid.std()),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "p05": float(valid.quantile(0.05)),
        "p10": float(valid.quantile(0.10)),
        "p25": float(valid.quantile(0.25)),
        "p75": float(valid.quantile(0.75)),
        "p90": float(valid.quantile(0.90)),
        "p95": float(valid.quantile(0.95)),
        "n_courses_r2_negative": n_negative,
        "pct_courses_r2_negative": float(n_negative / len(valid) * 100),
    }

# Global (media aritmética de los 5 targets, por curso, luego resumida)
loco_df["global_R2"] = loco_df[[f"{t}_R2" for t in TARGETS]].mean(axis=1, skipna=True)
valid_global = loco_df["global_R2"].dropna()
print(f"\n[GLOBAL, media aritmética de 5 targets por curso] "
      f"media={valid_global.mean():.4f}, mediana={valid_global.median():.4f}, "
      f"std={valid_global.std():.4f}, min={valid_global.min():.4f}, max={valid_global.max():.4f}")
summary["global"] = {
    "mean": float(valid_global.mean()),
    "median": float(valid_global.median()),
    "std": float(valid_global.std()),
    "min": float(valid_global.min()),
    "max": float(valid_global.max()),
    "p05": float(valid_global.quantile(0.05)),
    "p25": float(valid_global.quantile(0.25)),
    "p75": float(valid_global.quantile(0.75)),
    "p95": float(valid_global.quantile(0.95)),
}

# Comparación con la CV agrupada estándar (Sección 6.2 de METODOLOGIA-V3.md, ya conocida)
print("\nReferencia (ya conocida, NO recalculada aquí): CV agrupada por componente estudiante+curso, "
      "RandomForest, 18 features, dataset <=2023 -- R² global (aritmético) = 0.632 (METODOLOGIA-V3.md §6.2).")

# 10 cursos con peor y mejor R² global, para inspección
worst = loco_df.dropna(subset=["global_R2"]).nsmallest(10, "global_R2")[["course_hash", "n_students", "global_R2"]]
best = loco_df.dropna(subset=["global_R2"]).nlargest(10, "global_R2")[["course_hash", "n_students", "global_R2"]]
print("\n10 cursos con PEOR R² global al excluirlos:")
print(worst.to_string(index=False))
print("\n10 cursos con MEJOR R² global al excluirlos:")
print(best.to_string(index=False))

with open(os.path.join(OUT_DIR, "08_leave_one_course_out_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nGuardado: out/08_leave_one_course_out_per_course.csv, out/08_leave_one_course_out_summary.json")
