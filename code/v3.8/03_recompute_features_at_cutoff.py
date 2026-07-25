"""
v3.8 — Paso 3: recomputar las features limpias (sin fuga, la MISMA lista
`shared_clean_features.CLEAN_FEATURES` que usa v3.6-reanalysis -- 15 desde
el 21 jul 2026, tras excluir también min/max/var_exam_score) usando SOLO
eventos hasta el corte de 25%/50%/75% de la ventana de curso (Paso 2). Los
targets NO se tocan -- siguen siendo los agregados de curso completo de
`v3.6-reanalysis/out/clean_dataset.csv`.

Fórmulas de agregación: EXACTAMENTE las mismas de
`01-preprocessing/tareas/02_consolida_tareas.py` y
`01-preprocessing/examenes/05_consolida_examenes.py` -- lo único que
cambia es que se filtran los eventos por fecha ANTES de agregar. Fuente de
eventos: `code/v3.6/data/1. Para tratar/{tareas,examenes}/*.csv` (no se
recomputan normalización/detección de outliers -- se reutilizan los
campos ya calculados sobre el evento individual, ver docstring de
`03_detecta_outliers_tiempo.py`: la clasificación de un intento como
"outlier de tiempo" es una propiedad de ESE intento frente a la
distribución global de su `applicationKey`, no algo que dependa de cuándo
se hace el corte -- se documenta como simplificación deliberada, no un
error).

Los 3 datasets de salida (25/50/75) NO reciben el escalado Min-Max/Robust
global que sí tenía v3.6/v3.7 (`02_escala_y_renombra.py`) -- es
innecesario aquí: el protocolo de evaluación (`v3.8-reanalysis/`) ya
ajusta su propio imputer+scaler SOLO con la partición de entrenamiento de
cada corte, así que además de simplificar el pipeline, esto elimina una
fuga leve que sí tenían v3.6/v3.7 (ver METODOLOGIA-V3.md §5) -- no es un
error, es una mejora incidental que vale la pena mencionar al comparar
"100%" (v3.6, con esa fuga leve de preprocesamiento) vs. 25/50/75% (v3.8,
sin ella).

Cobertura: para cada corte, se reporta explícitamente qué fracción de las
4230 filas objetivo tienen AL MENOS un evento de tareas y de exámenes
dentro de la ventana -- no se imputa en silencio, la ausencia real de
datos tempranos se dejará como NaN (median-imputada después, en el mismo
paso train-only de siempre) y se documenta cuánta es.

Output (por corte X en 25,50,75):
  out/cutoff_X_features.csv   (uid_hash, course_hash, features crudas)
  out/coverage_report.json    (cobertura de datos por corte, un solo archivo)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("V38_OUT_DIR", os.path.join(HERE, "out"))
os.makedirs(OUT_DIR, exist_ok=True)
V36_DATA_1 = os.path.join(HERE, "..", "v3.6", "data", "1. Para tratar")

sys.path.insert(0, os.path.join(HERE, "..", "03-model"))
from shared_clean_features import CLEAN_FEATURES  # noqa: E402

valid_triples = pd.read_csv(os.path.join(OUT_DIR, "valid_triples.csv"))
course_windows = pd.read_csv(os.path.join(OUT_DIR, "course_windows.csv"), parse_dates=["cutoff_25", "cutoff_50", "cutoff_75"])
# Solo las columnas necesarias para el filtro de fecha -- `course_windows`
# tiene sus PROPIAS columnas `start_date`/`end_date` (ventana del CURSO),
# que colisionan de nombre con `start_date` de exámenes (fecha del EVENTO)
# si se arrastran todas al merge. Se seleccionan solo cutoff_25/50/75.
triples = valid_triples.merge(
    course_windows[["course_hash", "cutoff_25", "cutoff_50", "cutoff_75"]], on="course_hash", how="left"
)
print(f"valid_triples con ventana: {triples['cutoff_25'].notna().sum()} / {len(triples)}")

# ---------------------------------------------------------------------
# Carga de eventos crudos (una sola vez, se filtran por corte más abajo)
# ---------------------------------------------------------------------
tareas = pd.read_csv(os.path.join(V36_DATA_1, "tareas", "tareas_normalizado.csv"))
tareas["fecha_compromiso"] = pd.to_datetime(tareas["fecha_compromiso"], errors="coerce")
# ⚠️ usar `triples` (valid_triples + ventana/cortes ya unidos), NO
# `valid_triples` a secas -- de lo contrario faltan las columnas
# cutoff_25/50/75 y revienta más abajo (bug real de la corrida anterior,
# corregido aquí).
tareas = tareas.merge(
    triples, left_on=["curso", "email", "groupKey"], right_on=["course_hash", "uid_hash", "groupKey"], how="inner"
)
print(f"Eventos de tareas restringidos a filas objetivo: {len(tareas)}")

examenes = pd.read_csv(os.path.join(V36_DATA_1, "examenes", "examenes_sin_los_de_otra_materia.csv"))
examenes["start_date"] = pd.to_datetime(examenes["start_date"], errors="coerce")
# atribución por (groupKey, email) -- igual que el merge original (01_merge_datasets.py línea 37)
examenes = examenes.merge(
    triples, left_on=["groupKey", "email"], right_on=["groupKey", "uid_hash"], how="inner"
)
print(f"Eventos de examen restringidos a filas objetivo: {len(examenes)}")

coverage_report = {}


def aggregate_tareas(df_events: pd.DataFrame) -> pd.DataFrame:
    """Réplica de 02_consolida_tareas.py, solo los campos que alimentan las 18 features limpias."""
    agg = df_events.groupby(["uid_hash", "course_hash"]).agg(
        max_assignment_score=("calificacion_normalizada", "max"),
        min_assignment_score=("calificacion_normalizada", "min"),
        assignment_score_var=("calificacion_normalizada", "std"),
    ).reset_index()

    # avg_assignment_delay = mean(dias_anticipacion | fue_entregada==1) -- réplica exacta
    delayed = df_events[df_events["fue_entregada"] == 1]
    delay_agg = delayed.groupby(["uid_hash", "course_hash"])["dias_anticipacion"].mean().reset_index()
    delay_agg.columns = ["uid_hash", "course_hash", "avg_assignment_delay"]

    agg = agg.merge(delay_agg, on=["uid_hash", "course_hash"], how="left")
    return agg


def aggregate_examenes(df_events: pd.DataFrame) -> pd.DataFrame:
    """Réplica de 05_consolida_examenes.py, solo los campos que alimentan las 18 features limpias."""
    agg = df_events.groupby(["uid_hash", "course_hash"]).agg(
        completed_exams=("finished", "sum"),
        incomplete_exams=("finished", lambda x: (x == 0).sum()),
        exam_submit_rate=("finished", "mean"),
        max_exam_score=("normalized_score", "max"),
        min_exam_score=("normalized_score", "min"),
        exam_score_var=("normalized_score", "std"),
        perfect_exams=("normalized_score", lambda x: (x == 10).sum()),
        min_exam_time=("elapsed_time", "min"),
        outlier_exams_course=("outlier_elapsed_time", "sum"),
        total_exams=("applicationKey", "count"),
    ).reset_index()

    events = df_events.copy()
    events["exam_incidents_event"] = (
        (events["finished"] == 0).astype(int) + events["outlier_elapsed_time"] + events["general_outlier_elapsed_time"]
    )
    incidents = events.groupby(["uid_hash", "course_hash"])["exam_incidents_event"].agg(
        exam_incidents="sum", avg_exam_incidents="mean"
    ).reset_index()

    agg = agg.merge(incidents, on=["uid_hash", "course_hash"], how="left")
    return agg


def build_exam_modality_flags(exam_agg: pd.DataFrame, all_target_rows: pd.DataFrame) -> pd.DataFrame:
    """Réplica de 02_flags_modalidad_faltante.py (no_exam_data 0/1/2/3) + el
    one-hot de 02_escala_y_renombra.py (incluye el mismo 'bug' de origen:
    el código 3, <=5% cobertura, no tiene columna dummy propia -- se
    replica tal cual, no se corrige aquí)."""
    merged = all_target_rows[["uid_hash", "course_hash"]].merge(
        exam_agg[["uid_hash", "course_hash", "total_exams"]], on=["uid_hash", "course_hash"], how="left"
    )
    exam_presence = merged["total_exams"].notna() & (merged["total_exams"] > 0)
    merged["has_exam"] = exam_presence.astype(int)

    coverage = merged.groupby("course_hash")["has_exam"].agg(["sum", "count"]).reset_index()
    coverage.columns = ["course_hash", "n_with_exam", "n_total"]
    coverage["ratio"] = coverage["n_with_exam"] / coverage["n_total"]

    courses_all = coverage.loc[coverage["ratio"] == 1.0, "course_hash"]
    courses_none = coverage.loc[coverage["ratio"] == 0.0, "course_hash"]
    courses_some_missing = coverage.loc[(coverage["ratio"] >= 0.95) & (coverage["ratio"] < 1.0), "course_hash"]
    courses_few = coverage.loc[(coverage["ratio"] <= 0.05) & (coverage["ratio"] > 0.0), "course_hash"]

    merged["no_exam_data"] = 0
    merged.loc[(merged["has_exam"] == 0), "no_exam_data"] = 1
    merged.loc[(merged["course_hash"].isin(courses_some_missing)) & (merged["has_exam"] == 0), "no_exam_data"] = 2
    merged.loc[(merged["course_hash"].isin(courses_few)) & (merged["has_exam"] == 0), "no_exam_data"] = 3

    merged["all_exams"] = (merged["no_exam_data"] == 0).astype(int)
    merged["no_exams_flag"] = (merged["no_exam_data"] == 1).astype(int)
    merged["five%_or_less_incomplete_exams"] = (merged["no_exam_data"] == 2).astype(int)
    # no_exam_data == 3 no tiene columna dummy propia (mismo comportamiento
    # que el 02_escala_y_renombra.py original) -- queda con all_exams=0,
    # no_exams_flag=0, five%_or_less_incomplete_exams=0.

    return merged[["uid_hash", "course_hash", "all_exams", "five%_or_less_incomplete_exams"]]


all_target_rows = valid_triples[["uid_hash", "course_hash"]].drop_duplicates()

for pct in (25, 50, 75):
    cutoff_col = f"cutoff_{pct}"
    print(f"\n{'=' * 70}\nCORTE {pct}%\n{'=' * 70}")

    t_cut = tareas[tareas["fecha_compromiso"] <= tareas[cutoff_col]]
    e_cut = examenes[examenes["start_date"] <= examenes[cutoff_col]]
    print(f"Eventos de tareas <= corte: {len(t_cut)} / {len(tareas)} ({len(t_cut) / len(tareas) * 100:.1f}%)")
    print(f"Eventos de examen <= corte: {len(e_cut)} / {len(examenes)} ({len(e_cut) / len(examenes) * 100:.1f}%)")

    tareas_agg = aggregate_tareas(t_cut)
    exam_agg = aggregate_examenes(e_cut)
    exam_flags = build_exam_modality_flags(exam_agg, all_target_rows)

    out_df = all_target_rows.merge(tareas_agg, on=["uid_hash", "course_hash"], how="left")
    out_df = out_df.merge(exam_agg, on=["uid_hash", "course_hash"], how="left")
    out_df = out_df.merge(exam_flags, on=["uid_hash", "course_hash"], how="left")

    # Conteos: ausencia real de eventos ANTES del corte = 0 (no NaN) --
    # es información genuina ("todavía no hizo/tomó nada"), no un dato
    # faltante a imputar.
    count_cols = ["completed_exams", "incomplete_exams", "perfect_exams", "outlier_exams_course",
                  "total_exams", "exam_incidents"]
    for c in count_cols:
        out_df[c] = out_df[c].fillna(0)

    # Cobertura -- reportada explícitamente, NO imputada en silencio.
    n_total = len(out_df)
    n_with_tareas = (tareas_agg.shape[0])
    n_with_exam = int((out_df["total_exams"] > 0).sum())
    n_with_min_exam_score = int(out_df["min_exam_score"].notna().sum())
    n_with_avg_delay = int(out_df["avg_assignment_delay"].notna().sum())

    print(f"Cobertura al {pct}%: {n_with_tareas}/{n_total} filas con >=1 evento de tareas "
          f"({n_with_tareas / n_total * 100:.1f}%); {n_with_exam}/{n_total} con >=1 examen "
          f"({n_with_exam / n_total * 100:.1f}%)")
    print(f"  -> min_exam_score no-nulo: {n_with_min_exam_score}/{n_total} ({n_with_min_exam_score / n_total * 100:.1f}%)")
    print(f"  -> avg_assignment_delay no-nulo (requiere >=1 entrega a tiempo): "
          f"{n_with_avg_delay}/{n_total} ({n_with_avg_delay / n_total * 100:.1f}%)")

    coverage_report[f"cutoff_{pct}"] = {
        "n_total_rows": n_total,
        "n_with_at_least_1_tareas_event": int(n_with_tareas),
        "pct_with_at_least_1_tareas_event": float(n_with_tareas / n_total * 100),
        "n_with_at_least_1_exam_event": n_with_exam,
        "pct_with_at_least_1_exam_event": float(n_with_exam / n_total * 100),
        "n_with_min_exam_score_nonnull": n_with_min_exam_score,
        "pct_with_min_exam_score_nonnull": float(n_with_min_exam_score / n_total * 100),
        "n_with_avg_assignment_delay_nonnull": n_with_avg_delay,
        "pct_with_avg_assignment_delay_nonnull": float(n_with_avg_delay / n_total * 100),
        "n_tareas_events_le_cutoff": int(len(t_cut)),
        "n_tareas_events_total": int(len(tareas)),
        "n_exam_events_le_cutoff": int(len(e_cut)),
        "n_exam_events_total": int(len(examenes)),
    }

    keep_cols = ["uid_hash", "course_hash"] + CLEAN_FEATURES
    out_df = out_df[keep_cols]
    out_path = os.path.join(OUT_DIR, f"cutoff_{pct}_features.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({out_df.shape})")

with open(os.path.join(OUT_DIR, "coverage_report.json"), "w") as f:
    json.dump(coverage_report, f, indent=2)
print(f"\nGuardado: {os.path.join(OUT_DIR, 'coverage_report.json')}")
