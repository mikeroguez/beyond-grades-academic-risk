"""
v3.8 — Paso 2: ventana temporal [inicio, fin] por curso y fechas de corte
al 25%/50%/75% de la duración.

⚠️ HALLAZGO durante la construcción (reportado tal cual, no se oculta):
la versión ingenua de esta ventana -- min/max de TODAS las marcas de
tiempo de las 4 fuentes (tareas, exámenes, accesos, foros) -- da
duraciones de curso absurdas: mediana 652 días (~21 meses), máximo 1334
días (~3.7 años). Eso NO es un semestre. Investigado: usando SOLO
`Fecha_compromiso` (fecha de vencimiento de tareas, agendada por el
profesor, no un log de actividad del estudiante) la mediana baja a 112
días (~16 semanas) con rango intercuartílico 79-126 días -- exactamente
del orden de un semestre real. Los logs de accesos y foros (y en menor
medida los exámenes) tienen una cola larga de eventos mucho después de
que el curso "terminó" (estudiantes revisando material, profesores
corrigiendo tarde, o el mismo código de `curso` reutilizado en otro
periodo -- ver `course_retake_probability` en
`01-preprocessing/tareas/02_consolida_tareas.py`, que ya intenta detectar
esto mismo mecanismo pero evidentemente no para el 100% de los casos).

**Decisión de diseño (documentada, no oculta):** la ventana de curso se
define usando ÚNICAMENTE `Fecha_compromiso` de tareas (min/max) -- son
fechas AGENDADAS por el profesor al crear cada actividad, no eventos de
actividad del estudiante, así que no sufren la cola larga de accesos
tardíos. Sigue habiendo una cola larga residual (algunos cursos con
`curso` reciclado entre periodos, duración >300 días) -- se reporta
explícitamente cuántos cursos caen en ese caso, no se excluyen (mismas
165 cursos que v3.6/v3.7, para mantener el dataset comparable), pero se
deja como advertencia de interpretación para esos cursos específicos.

Cortes: para cada curso, cutoff_X = inicio + X% * (fin - inicio), X en
{25, 50, 75}. El 100% (curso completo) YA es exactamente v3.6 -- no se
recalcula aquí ni en el paso 3.

Output: out/course_windows.csv
        (course_hash, start_date, end_date, duration_days,
         cutoff_25, cutoff_50, cutoff_75, n_timestamps)
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("V38_OUT_DIR", os.path.join(HERE, "out"))
os.makedirs(OUT_DIR, exist_ok=True)
V36_DATA_1 = os.path.join(HERE, "..", "v3.6", "data", "1. Para tratar")

valid_triples = pd.read_csv(os.path.join(OUT_DIR, "valid_triples.csv"))
target_courses = set(valid_triples["course_hash"].unique())
print(f"Cursos objetivo: {len(target_courses)}")

# ---------------------------------------------------------------------
# Comparación explícita: ventana "ingenua" (4 fuentes) vs. la elegida
# (solo Fecha_compromiso) -- se calculan ambas y se reporta la diferencia,
# para que quede trazable por qué se descartó la primera.
# ---------------------------------------------------------------------
groupkey_to_course = valid_triples.set_index("groupKey")["course_hash"].to_dict()
frames_naive = []

tareas_raw = pd.read_csv(os.path.join(V36_DATA_1, "tareas", "tareas.csv"))
tareas_raw = tareas_raw[tareas_raw["curso"].isin(target_courses)]
for col in ["Fecha_compromiso", "Fecha_de_entrega"]:
    d = pd.to_datetime(tareas_raw[col], errors="coerce")
    frames_naive.append(pd.DataFrame({"course_hash": tareas_raw["curso"], "date": d}))

examenes = pd.read_csv(os.path.join(V36_DATA_1, "examenes", "examenes_sin_los_de_otra_materia.csv"))
examenes["course_hash"] = examenes["groupKey"].map(groupkey_to_course)
examenes = examenes[examenes["course_hash"].notna()]
for col in ["start_date", "finish_date"]:
    d = pd.to_datetime(examenes[col], errors="coerce")
    frames_naive.append(pd.DataFrame({"course_hash": examenes["course_hash"], "date": d}))

accesos = pd.read_csv(os.path.join(V36_DATA_1, "accesos", "accesos.csv"))
accesos = accesos[accesos["curso"].isin(target_courses)]
for col in ["fecha_ingreso", "fecha_egreso"]:
    d = pd.to_datetime(accesos[col], errors="coerce")
    frames_naive.append(pd.DataFrame({"course_hash": accesos["curso"], "date": d}))

foros = pd.read_csv(os.path.join(V36_DATA_1, "foros", "foros.csv"))
foros = foros[foros["curso"].isin(target_courses)]
d = pd.to_datetime(foros["fecha_y_hora"], errors="coerce")
frames_naive.append(pd.DataFrame({"course_hash": foros["curso"], "date": d}))

naive_dates = pd.concat(frames_naive, ignore_index=True).dropna()
naive_windows = naive_dates.groupby("course_hash")["date"].agg(["min", "max"]).reset_index()
naive_windows["duration_days"] = (naive_windows["max"] - naive_windows["min"]).dt.total_seconds() / 86400.0
print("\n[Ventana ingenua, 4 fuentes -- SOLO para comparar, NO se usa]")
print(naive_windows["duration_days"].describe())

# ---------------------------------------------------------------------
# Ventana elegida: solo Fecha_compromiso de tareas
# ---------------------------------------------------------------------
tareas_dates = pd.to_datetime(tareas_raw["Fecha_compromiso"], errors="coerce")
commit_df = pd.DataFrame({"course_hash": tareas_raw["curso"], "date": tareas_dates}).dropna()

windows = commit_df.groupby("course_hash")["date"].agg(["min", "max", "count"]).reset_index()
windows.columns = ["course_hash", "start_date", "end_date", "n_timestamps"]
windows["duration_days"] = (windows["end_date"] - windows["start_date"]).dt.total_seconds() / 86400.0

for pct in (25, 50, 75):
    frac = pct / 100.0
    windows[f"cutoff_{pct}"] = windows["start_date"] + frac * (windows["end_date"] - windows["start_date"])

missing_courses = target_courses - set(windows["course_hash"])
print(f"\nCursos objetivo SIN ventana reconstruible (Fecha_compromiso): {len(missing_courses)} (de {len(target_courses)})")
if missing_courses:
    print(f"  -> {sorted(missing_courses)}")

zero_duration = (windows["duration_days"] <= 0).sum()
print(f"Cursos con duración <= 0 días (una sola fecha de compromiso -- el corte 25/50/75% coincide con esa fecha): {zero_duration}")

long_tail = (windows["duration_days"] > 300).sum()
print(f"Cursos con duración > 300 días (posible curso/código reciclado entre periodos, cola larga residual): {long_tail} "
      f"de {len(windows)} -- se conservan (mismo set de 165 cursos que v3.6/v3.7), advertencia de interpretación.")

print("\n[Ventana elegida: solo Fecha_compromiso de tareas]")
print(windows["duration_days"].describe())

windows.to_csv(os.path.join(OUT_DIR, "course_windows.csv"), index=False)
print(f"\nGuardado: {os.path.join(OUT_DIR, 'course_windows.csv')} ({windows.shape})")
