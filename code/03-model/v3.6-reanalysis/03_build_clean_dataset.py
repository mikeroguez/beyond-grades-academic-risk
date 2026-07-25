"""
WP1 — Paso 3: Ensamblado del dataset limpio de trabajo para el reanálisis.

Aplica, sobre `dataset_final_3_x.csv` (4230x33, ya seudonimizado):
  1. Exclusión de variables leaky (Reviewer 1 #1-4) — lista blanket basada en
     AUDITORIA-LEAKAGE.md Sección 3 (clasificación FD/FB) + confirmación
     empírica propia en 01_leakage_confirmation.py: se excluyen del set de
     predictores TODAS las columnas que son componente algebraico directo
     (numerador/denominador/complemento exacto) de la fórmula de origen de
     CUALQUIERA de los 5 targets, de forma uniforme para los 5 (el modelo es
     multi-tarea con un tronco compartido -> un solo X para los 5 targets).
  2. Componente de agrupamiento estudiante+curso (Reviewer 1 #7-8): union-find
     sobre (uid_hash, course_hash) para que ningún split separe filas que
     comparten estudiante O curso -- ver justificación abajo.
  3. Bucket temporal (≤2023 / 2024) de 02_build_course_year.py (Reviewer 1 #9).

Por qué union-find y no `GroupKFold(groups=uid_hash)` a secas: GroupKFold
agrupa solo por una clave. Si agrupamos solo por estudiante, dos filas del
MISMO curso pero de estudiantes distintos pueden caer una en train y otra en
test -> el modelo ve señal específica del curso (dificultad del profesor,
calendario, tipo de examen) en entrenamiento y la "reconoce" en test. Si
agrupamos solo por curso, análogamente un estudiante que repite materia
podría aparecer en ambos lados. La solución estándar para "agrupar por A Y
por B simultáneamente" es tratar (estudiante, curso) como un grafo bipartito
y usar sus componentes conexas como la unidad de agrupamiento: dos filas
quedan en el mismo componente si comparten estudiante O curso (directamente
o transitivamente, p. ej. est1-cursoA-est2-cursoB-est3). Cada componente se
asigna ENTERO a un solo lado del split.

Output: code/03-model/v3.6-reanalysis/out/clean_dataset.csv
        (uid_hash, course_hash, component_id, split_bucket, year, min_date,
         5 targets, features limpias)
        code/03-model/v3.6-reanalysis/out/feature_lists.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(HERE, ".."))
from shared_clean_features import LEAKY_PREDICTORS_BLANKET, CLEAN_FEATURES as EXPECTED_CLEAN_FEATURES  # noqa: E402

FINAL_CSV = os.path.join(HERE, "..", "..", "v3.6", "data", "Material", "dataset_final_3_x.csv")
ROW_BUCKET_CSV = os.path.join(OUT_DIR, "row_split_bucket.csv")

ID_COLS = ["uid_hash", "course_hash"]
TARGETS = ["avg_assignment_score", "missing_assignments", "assignment_procrast_rate", "avg_exam_score", "exam_accuracy"]

# LEAKY_PREDICTORS_BLANKET vive en shared_clean_features.py -- es la MISMA
# lista que usa el pipeline de cortes temporales (code/v3.8/,
# 03-model/v3.8-reanalysis/). Ver ese módulo para la justificación de cada
# exclusión (AUDITORIA-LEAKAGE.md Sección 3 y 5, 01_leakage_confirmation.py,
# 07_avg_exam_score_sensitivity.py).

# Features "borderline" señaladas por la auditoría (estructuralmente ligadas
# al target pero no un componente algebraico de su fórmula) -- se conservan
# en el set principal, pero se guardan aparte para la ablación de
# sensibilidad (05_baselines_dnn.py hace un análisis adicional quitándolas).
BORDERLINE_FEATURES = ["exam_submit_rate", "incomplete_exams", "completed_exams", "exam_incidents", "avg_exam_incidents"]

df = pd.read_csv(FINAL_CSV)
bucket = pd.read_csv(ROW_BUCKET_CSV)
df = df.merge(bucket, on=["uid_hash", "course_hash"], how="left")

print(f"dataset_final_3_x.csv: {df.shape}")
print(f"split_bucket tras merge:\n{df['split_bucket'].value_counts()}")

# ---------------------------------------------------------------------
# Union-Find sobre (uid_hash, course_hash)
# ---------------------------------------------------------------------
parent = {}


def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb


for _, row in df.iterrows():
    u = f"stu::{row['uid_hash']}"
    c = f"crs::{row['course_hash']}"
    find(u)
    find(c)
    union(u, c)

df["component_id"] = df["uid_hash"].apply(lambda u: find(f"stu::{u}"))

n_components = df["component_id"].nunique()
comp_sizes = df.groupby("component_id").size()
print(f"\nComponentes conexas (estudiante+curso): {n_components}")
print(f"Tamaño de componente: min={comp_sizes.min()}, mediana={comp_sizes.median()}, "
      f"media={comp_sizes.mean():.1f}, max={comp_sizes.max()}")
print("(un componente grande es esperable: si dos cursos comparten aunque sea un estudiante, "
      "y esos cursos tienen muchos otros estudiantes, todo se funde en una sola componente -- "
      "ver nota de limitación en METODOLOGIA-V3.md)")

# ---------------------------------------------------------------------
# Features limpias
# ---------------------------------------------------------------------
all_feature_cols = [c for c in df.columns if c not in ID_COLS + TARGETS + ["min_date", "year", "split_bucket", "component_id"]]
clean_features = [c for c in all_feature_cols if c not in LEAKY_PREDICTORS_BLANKET]

print(f"\nFeatures originales (26): {len(all_feature_cols)}")
print(f"Excluidas por fuga ({len(LEAKY_PREDICTORS_BLANKET)}): {LEAKY_PREDICTORS_BLANKET}")
print(f"Features limpias resultantes ({len(clean_features)}): {clean_features}")

assert set(clean_features) == set(EXPECTED_CLEAN_FEATURES), (
    "clean_features calculado aqui no coincide con shared_clean_features.CLEAN_FEATURES -- "
    "actualiza ese modulo, no solo este script, para que el pipeline de cortes (v3.8) "
    f"no quede desalineado. Calculado: {sorted(clean_features)}. "
    f"Esperado: {sorted(EXPECTED_CLEAN_FEATURES)}."
)

keep_cols = ID_COLS + ["component_id", "split_bucket", "year", "min_date"] + TARGETS + clean_features
out_df = df[keep_cols].copy()
out_df.to_csv(os.path.join(OUT_DIR, "clean_dataset.csv"), index=False)

with open(os.path.join(OUT_DIR, "feature_lists.json"), "w") as f:
    json.dump(
        {
            "all_26_features": all_feature_cols,
            "leaky_predictors_excluded": LEAKY_PREDICTORS_BLANKET,
            "clean_features": clean_features,
            "borderline_features_kept_but_flagged": BORDERLINE_FEATURES,
            "targets": TARGETS,
        },
        f,
        indent=2,
    )

print(f"\nGuardado: {os.path.join(OUT_DIR, 'clean_dataset.csv')} ({out_df.shape})")
print(f"Guardado: {os.path.join(OUT_DIR, 'feature_lists.json')}")
