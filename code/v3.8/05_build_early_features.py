"""
v3.8 — Etapa 2, palanca 2: features genuinamente tempranas -- no agregados
estáticos truncados (eso ya lo hace `03_recompute_features_at_cutoff.py`),
sino señales que solo tienen sentido EN un punto de corte t: tendencia
dentro de la ventana, recencia, ritmo, y posición relativa a los pares del
curso en ese mismo instante t.

7 features nuevas, cada una definida y justificada abajo. TODAS usan
ÚNICAMENTE eventos con fecha <= corte -- ninguna usa información posterior
al corte ni del propio target (verificado con auditoría empírica en
`06_audit_new_features.py`, ver esa salida antes de dar estas features por
buenas).

1. `assignment_grade_trend_slope` -- pendiente de una regresión lineal de
   `calificacion_normalizada` vs. día transcurrido (día = fecha_de_entrega
   - inicio de curso), sobre las entregas del estudiante hasta el corte.
   Requiere >=2 entregas calificadas; si no, NaN (imputado después).
   Captura si el estudiante va mejorando o empeorando DENTRO de la
   ventana observada -- una tendencia, no un promedio estático.

2. `days_since_last_engagement` -- (fecha de corte) - (fecha del último
   evento real de compromiso del estudiante hasta el corte), en días.
   "Evento real de compromiso" = entrega de tarea (fecha_de_entrega
   cuando fue_entregada==1) O inicio de intento de examen (start_date).
   Si no hay ningún evento aún, se imputa como (fecha de corte - inicio
   de curso), es decir, "tan inactivo como el curso es viejo" -- un valor
   alto y genuino, no un NaN oculto.

3. `submission_pace_per_week` -- entregas realizadas hasta el corte /
   semanas transcurridas desde el inicio del curso hasta el corte
   (mínimo 1 semana en el denominador para evitar división por casi-cero
   en cortes muy tempranos de cursos muy cortos).

4. `exam_attempts_pace_per_week` -- análogo a 3 pero para intentos de
   examen (`applicationKey` iniciados hasta el corte / semanas
   transcurridas).

5. `relative_position_avg_score` -- (promedio de calificacion_normalizada
   del estudiante hasta el corte) - (promedio de calificacion_normalizada
   de TODOS los estudiantes del mismo curso hasta el corte). Positivo =
   por encima del promedio del curso EN ESE MOMENTO, no al final.
   Nota de diseño: el promedio del curso se calcula sobre todos los
   estudiantes del curso (train+test da igual, es una estadística
   poblacional transversal, no usa el target) -- mismo tratamiento que ya
   usa el pipeline original para normalizar calificaciones por actividad
   (ver 01_normaliza_tareas.py, stats por id_actividad sobre todos los
   estudiantes) -- se documenta para que quede trazable, no es una
   inconsistencia nueva de v3.8.

6. `relative_position_submit_rate` -- análogo a 5 pero para tasa de
   entrega (entregadas/asignadas hasta el corte) del estudiante vs. el
   promedio de su curso hasta el corte.

7. `exam_score_trend_slope` -- análogo a 1 pero para
   `normalized_score` de exámenes vs. día transcurrido, sobre los
   intentos de examen del estudiante hasta el corte. Requiere >=2
   intentos calificados; si no, NaN.

Output (por corte): out/cutoff_X_early_features.csv
  (uid_hash, course_hash, + las 7 columnas de arriba)
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("V38_OUT_DIR", os.path.join(HERE, "out"))
os.makedirs(OUT_DIR, exist_ok=True)
V36_DATA_1 = os.path.join(HERE, "..", "v3.6", "data", "1. Para tratar")

EARLY_FEATURE_COLS = [
    "assignment_grade_trend_slope", "days_since_last_engagement", "submission_pace_per_week",
    "exam_attempts_pace_per_week", "relative_position_avg_score", "relative_position_submit_rate",
    "exam_score_trend_slope",
]

valid_triples = pd.read_csv(os.path.join(OUT_DIR, "valid_triples.csv"))
course_windows = pd.read_csv(
    os.path.join(OUT_DIR, "course_windows.csv"),
    parse_dates=["start_date", "end_date", "cutoff_25", "cutoff_50", "cutoff_75"],
)
# ⚠️ renombrar start_date/end_date del CURSO antes de mergear -- ambas
# fuentes de eventos (tareas, exámenes) tienen sus PROPIAS columnas de
# fecha con nombres que colisionan (examenes trae su propio "start_date"
# por evento) -- mismo tipo de bug ya corregido en 03/04, evitado aquí
# desde el diseño.
course_windows_renamed = course_windows.rename(columns={"start_date": "course_start_date", "end_date": "course_end_date"})
triples = valid_triples.merge(
    course_windows_renamed[["course_hash", "course_start_date", "course_end_date", "cutoff_25", "cutoff_50", "cutoff_75"]],
    on="course_hash", how="left",
)
all_target_rows = valid_triples[["uid_hash", "course_hash"]].drop_duplicates()

tareas = pd.read_csv(os.path.join(V36_DATA_1, "tareas", "tareas_normalizado.csv"))
tareas["fecha_compromiso"] = pd.to_datetime(tareas["fecha_compromiso"], errors="coerce")
tareas["fecha_de_entrega"] = pd.to_datetime(tareas["fecha_de_entrega"], errors="coerce")
tareas = tareas.merge(
    triples, left_on=["curso", "email", "groupKey"], right_on=["course_hash", "uid_hash", "groupKey"], how="inner"
)

examenes = pd.read_csv(os.path.join(V36_DATA_1, "examenes", "examenes_sin_los_de_otra_materia.csv"))
examenes["start_date"] = pd.to_datetime(examenes["start_date"], errors="coerce")
examenes = examenes.merge(triples, left_on=["groupKey", "email"], right_on=["groupKey", "uid_hash"], how="inner")


def slope(days: np.ndarray, values: np.ndarray) -> float:
    mask = ~(pd.isna(days) | pd.isna(values))
    days, values = days[mask], values[mask]
    if len(days) < 2 or np.ptp(days) == 0:
        return np.nan
    return float(np.polyfit(days, values, 1)[0])


for pct in (25, 50, 75):
    cutoff_col = f"cutoff_{pct}"
    print(f"\n{'=' * 70}\nCORTE {pct}%\n{'=' * 70}")

    t_cut = tareas[tareas["fecha_compromiso"] <= tareas[cutoff_col]].copy()
    e_cut = examenes[examenes["start_date"] <= examenes[cutoff_col]].copy()

    rows = []
    for (uid, cid), grp in t_cut.groupby(["uid_hash", "course_hash"]):
        row = {"uid_hash": uid, "course_hash": cid}
        graded = grp.dropna(subset=["calificacion_normalizada"])
        row["assignment_grade_trend_slope"] = slope(
            graded["fecha_compromiso"].values.astype("datetime64[D]").astype(float),
            graded["calificacion_normalizada"].values,
        )
        rows.append(row)
    trend_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["uid_hash", "course_hash", "assignment_grade_trend_slope"])

    rows2 = []
    for (uid, cid), grp in e_cut.groupby(["uid_hash", "course_hash"]):
        row = {"uid_hash": uid, "course_hash": cid}
        graded = grp.dropna(subset=["normalized_score"])
        row["exam_score_trend_slope"] = slope(
            graded["start_date"].values.astype("datetime64[D]").astype(float),
            graded["normalized_score"].values,
        )
        rows2.append(row)
    exam_trend_df = pd.DataFrame(rows2) if rows2 else pd.DataFrame(columns=["uid_hash", "course_hash", "exam_score_trend_slope"])

    # --- recencia: último evento real de compromiso (entrega o inicio de examen) ---
    last_submit = t_cut[t_cut["fue_entregada"] == 1].groupby(["uid_hash", "course_hash"])["fecha_de_entrega"].max()
    last_exam = e_cut.groupby(["uid_hash", "course_hash"])["start_date"].max()
    last_events = pd.concat([last_submit.rename("d1"), last_exam.rename("d2")], axis=1)
    last_events["last_engagement"] = last_events[["d1", "d2"]].max(axis=1)

    # --- ritmo ---
    submit_count = t_cut[t_cut["fue_entregada"] == 1].groupby(["uid_hash", "course_hash"]).size().rename("n_submits")
    exam_count = e_cut.groupby(["uid_hash", "course_hash"]).size().rename("n_exam_attempts")

    # --- promedio propio hasta el corte (para posición relativa) ---
    own_avg_score = t_cut.groupby(["uid_hash", "course_hash"])["calificacion_normalizada"].mean().rename("own_avg_score")
    own_submit_rate = (
        t_cut.groupby(["uid_hash", "course_hash"])["fue_entregada"].mean().rename("own_submit_rate")
    )

    course_avg_score = t_cut.groupby("course_hash")["calificacion_normalizada"].mean().rename("course_avg_score")
    course_submit_rate = t_cut.groupby("course_hash")["fue_entregada"].mean().rename("course_submit_rate")

    out_df = all_target_rows.merge(triples[["uid_hash", "course_hash", "course_start_date", cutoff_col]].drop_duplicates(),
                                    on=["uid_hash", "course_hash"], how="left")
    out_df["weeks_elapsed"] = ((out_df[cutoff_col] - out_df["course_start_date"]).dt.total_seconds() / (86400 * 7)).clip(lower=1.0)

    out_df = out_df.merge(trend_df, on=["uid_hash", "course_hash"], how="left")
    out_df = out_df.merge(exam_trend_df, on=["uid_hash", "course_hash"], how="left")
    out_df = out_df.merge(last_events[["last_engagement"]].reset_index(), on=["uid_hash", "course_hash"], how="left")
    out_df = out_df.merge(submit_count.reset_index(), on=["uid_hash", "course_hash"], how="left")
    out_df = out_df.merge(exam_count.reset_index(), on=["uid_hash", "course_hash"], how="left")
    out_df = out_df.merge(own_avg_score.reset_index(), on=["uid_hash", "course_hash"], how="left")
    out_df = out_df.merge(own_submit_rate.reset_index(), on=["uid_hash", "course_hash"], how="left")
    out_df = out_df.merge(course_avg_score.reset_index(), on="course_hash", how="left")
    out_df = out_df.merge(course_submit_rate.reset_index(), on="course_hash", how="left")

    out_df["n_submits"] = out_df["n_submits"].fillna(0)
    out_df["n_exam_attempts"] = out_df["n_exam_attempts"].fillna(0)
    out_df["submission_pace_per_week"] = out_df["n_submits"] / out_df["weeks_elapsed"]
    out_df["exam_attempts_pace_per_week"] = out_df["n_exam_attempts"] / out_df["weeks_elapsed"]

    out_df["days_since_last_engagement"] = np.where(
        out_df["last_engagement"].notna(),
        (out_df[cutoff_col] - out_df["last_engagement"]).dt.total_seconds() / 86400.0,
        (out_df[cutoff_col] - out_df["course_start_date"]).dt.total_seconds() / 86400.0,
    )

    out_df["relative_position_avg_score"] = out_df["own_avg_score"] - out_df["course_avg_score"]
    out_df["relative_position_submit_rate"] = out_df["own_submit_rate"] - out_df["course_submit_rate"]

    n_trend = int(out_df["assignment_grade_trend_slope"].notna().sum())
    n_exam_trend = int(out_df["exam_score_trend_slope"].notna().sum())
    n_relpos = int(out_df["relative_position_avg_score"].notna().sum())
    print(f"Cobertura: assignment_grade_trend_slope no-nulo: {n_trend}/{len(out_df)} ({n_trend/len(out_df)*100:.1f}%) "
          f"(requiere >=2 entregas calificadas)")
    print(f"           exam_score_trend_slope no-nulo: {n_exam_trend}/{len(out_df)} ({n_exam_trend/len(out_df)*100:.1f}%) "
          f"(requiere >=2 intentos de examen)")
    print(f"           relative_position_avg_score no-nulo: {n_relpos}/{len(out_df)} ({n_relpos/len(out_df)*100:.1f}%) "
          f"(requiere >=1 entrega calificada)")

    out_df = out_df[["uid_hash", "course_hash"] + EARLY_FEATURE_COLS]
    out_path = os.path.join(OUT_DIR, f"cutoff_{pct}_early_features.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({out_df.shape})")
