"""
v3.8 — Paso 1: mapa (uid_hash, course_hash) -> groupKey para las 4230 filas
del dataset objetivo (v3.6-reanalysis/out/clean_dataset.csv).

Por qué hace falta: los eventos crudos de exámenes (`examenes_sin_los_de_otra_materia.csv`)
NO traen `curso` directamente -- se atribuyen por (groupKey, email) en el
merge original (`code/v3.6/02-merge/01_merge_datasets.py`, línea 37:
`pd.merge(assignments_df, exams_df, how="left", on=["groupKey", "email"])`).
Para recomputar features con corte de fecha necesitamos saber, para cada
fila (uid_hash, course_hash) de nuestro dataset objetivo, cuál es su
`groupKey` -- así los eventos de tareas Y de exámenes se filtran/agregan
consistentemente con el mismo criterio de atribución que usó el pipeline
original.

Verificado antes de asumir que esto es seguro: (curso, email) es único en
`tareas_consolidado_cleaned_sin_cursos_reutilizados.csv` (0 duplicados en
6010 filas) y las 4230 filas de `clean_dataset.csv` matchean 1:1 sin
ambigüedad contra ese archivo (0 filas sin match, 0 filas con groupKey
ambiguo) -- aunque SÍ existe ambigüedad groupKey->curso a nivel de TODO el
dataset (169 pares email+groupKey con más de un curso, típicamente cursos
repetidos/groupKey reciclado), esa ambigüedad no afecta a ninguna de
nuestras 4230 filas objetivo.

Output: out/valid_triples.csv (uid_hash, course_hash, groupKey)
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("V38_OUT_DIR", os.path.join(HERE, "out"))
os.makedirs(OUT_DIR, exist_ok=True)

V36_DATA = os.path.join(HERE, "..", "v3.6", "data")
V36_REANALYSIS_OUT = os.path.join(HERE, "..", "03-model", "v3.6-reanalysis", "out")

tareas_consolidado = pd.read_csv(
    os.path.join(V36_DATA, "1. Para tratar", "tareas", "tareas_consolidado_cleaned_sin_cursos_reutilizados.csv")
)
clean_dataset = pd.read_csv(os.path.join(V36_REANALYSIS_OUT, "clean_dataset.csv"))

dup_check = tareas_consolidado.groupby(["curso", "email"]).size()
n_dup = int((dup_check > 1).sum())
print(f"(curso,email) duplicados en tareas_consolidado_cleaned_sin_cursos_reutilizados.csv: {n_dup} (esperado: 0)")

triples = clean_dataset[["uid_hash", "course_hash"]].merge(
    tareas_consolidado[["curso", "email", "groupKey"]],
    left_on=["course_hash", "uid_hash"],
    right_on=["curso", "email"],
    how="left",
)

n_unmatched = triples["groupKey"].isna().sum()
print(f"Filas objetivo (de {len(clean_dataset)}) sin groupKey encontrado: {n_unmatched} (esperado: 0)")
if n_unmatched > 0:
    raise ValueError(f"{n_unmatched} filas del dataset objetivo no tienen groupKey -- revisar antes de continuar.")

dup2 = triples.groupby(["uid_hash", "course_hash"]).size()
n_ambiguous = int((dup2 > 1).sum())
print(f"Filas objetivo con groupKey ambiguo (>1 match): {n_ambiguous} (esperado: 0)")
if n_ambiguous > 0:
    raise ValueError(f"{n_ambiguous} filas del dataset objetivo tienen groupKey ambiguo -- revisar antes de continuar.")

out = triples[["uid_hash", "course_hash", "groupKey"]].copy()
out.to_csv(os.path.join(OUT_DIR, "valid_triples.csv"), index=False)
print(f"\nGuardado: {os.path.join(OUT_DIR, 'valid_triples.csv')} ({out.shape})")
