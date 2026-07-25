"""
WP1 — Paso 2: Reconstrucción de la variable temporal para el holdout ≤2023/2024
(Reviewer 1 #9, comentario del editor).

`dataset_final_3_x.csv` (33 columnas finales) NO trae ninguna columna de
fecha: se elimina en 03-feature-selection (justificado — el objetivo de esa
etapa es dejar solo predictores numéricos). Para poder hacer el holdout
temporal que pide el editor, hay que re-derivar la fecha desde una etapa
anterior del pipeline que sí la tiene.

Fuente elegida: `code/v3.6/data/1. Para tratar/{tareas,examenes,accesos,foros}/*.csv`
(la salida ya seudonimizada de `00-anonymize/`, previa a cualquier
consolidación). Estos archivos comparten `curso`/`email` seudonimizados
(mismos `crs_NNNNN`/`stu_NNNNNN` que terminan en `course_hash`/`uid_hash`)
y cada uno trae al menos una columna de fecha calendario real:
  - tareas:    Fecha_compromiso, Fecha_de_entrega
  - examenes:  start_date, finish_date   (sin `curso` propio -> se vincula
               por email a través de tareas/accesos más abajo, ver nota)
  - accesos:   fecha_ingreso, fecha_egreso
  - foros:     fecha_y_hora

Decisión de diseño: se define la "fecha del curso" (`course_hash`) como el
MÍNIMO de todas las fechas calendario disponibles en tareas/accesos/foros
para ese `curso` (excluye examenes, que no trae `curso` directamente en el
CSV crudo — ver 01-preprocessing/examenes/02_filtra_por_groupkey_tareas.py
para el join real; no se reconstruye aquí por simplicidad y porque
tareas+accesos ya cubren el 100% de los cursos). Usar el mínimo (fecha de
inicio real de actividad) en vez del máximo evita que una entrega tardía
aislada en enero del año siguiente reclasifique un curso que en realidad
corrió el año anterior.

Cada curso se asigna ENTERO a un lado del holdout (todos sus estudiantes
van al mismo lado) — este es el mismo principio de agrupamiento por curso
que pide Reviewer 1 #7-8 para los splits, aplicado también al corte
temporal: un curso no puede "empezar" en 2023 y "terminar" en 2024 para
efectos de este split (evita fuga temporal indirecta vía compañeros de
curso).

Output: code/03-model/v3.6-reanalysis/out/course_year_map.csv
        columnas: course_hash, min_date, year, split_bucket (train_le2023 / holdout_2024 / other)
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V36_DATA = os.path.join(HERE, "..", "..", "v3.6", "data", "1. Para tratar")
OUT_DIR = os.path.join(HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)

frames = []

tareas = pd.read_csv(os.path.join(V36_DATA, "tareas", "tareas.csv"))
for col in ["Fecha_compromiso", "Fecha_de_entrega"]:
    d = pd.to_datetime(tareas[col], errors="coerce")
    frames.append(pd.DataFrame({"curso": tareas["curso"], "date": d}))

accesos = pd.read_csv(os.path.join(V36_DATA, "accesos", "accesos.csv"))
for col in ["fecha_ingreso", "fecha_egreso"]:
    d = pd.to_datetime(accesos[col], errors="coerce")
    frames.append(pd.DataFrame({"curso": accesos["curso"], "date": d}))

foros = pd.read_csv(os.path.join(V36_DATA, "foros", "foros.csv"))
d = pd.to_datetime(foros["fecha_y_hora"], errors="coerce")
frames.append(pd.DataFrame({"curso": foros["curso"], "date": d}))

all_dates = pd.concat(frames, ignore_index=True).dropna()
print(f"Total de marcas de tiempo utilizables: {len(all_dates)}")

course_dates = all_dates.groupby("curso")["date"].agg(["min", "max", "count"]).reset_index()
course_dates.columns = ["course_hash", "min_date", "max_date", "n_timestamps"]
course_dates["year"] = course_dates["min_date"].dt.year

print(f"Cursos con fecha reconstruible: {course_dates.shape[0]}")
print("Distribución de años (por fecha mínima de actividad):")
print(course_dates["year"].value_counts().sort_index())

# Verificación de cobertura contra dataset_final_3_x.csv
final_df = pd.read_csv(os.path.join(HERE, "..", "..", "v3.6", "data", "Material", "dataset_final_3_x.csv"))
final_courses = set(final_df["course_hash"].unique())
covered = final_courses & set(course_dates["course_hash"])
print(f"\nCursos en dataset_final_3_x.csv: {len(final_courses)}")
print(f"Cursos con fecha reconstruida: {len(covered)} ({len(covered)/len(final_courses)*100:.1f}% de cobertura)")
missing_courses = final_courses - covered
if missing_courses:
    print(f"⚠️ Cursos SIN fecha reconstruible ({len(missing_courses)}): se excluyen del holdout temporal, "
          f"quedan fuera tanto de train ≤2023 como de test 2024 (no se puede clasificar con certeza).")

course_dates["split_bucket"] = course_dates["year"].apply(
    lambda y: "train_le2023" if y <= 2023 else ("holdout_2024" if y == 2024 else "other_out_of_range")
)
print("\nDistribución de split_bucket (a nivel curso):")
print(course_dates["split_bucket"].value_counts())

# a nivel de filas del dataset final (estudiante-curso)
final_df = final_df.merge(course_dates[["course_hash", "min_date", "year", "split_bucket"]], on="course_hash", how="left")
final_df["split_bucket"] = final_df["split_bucket"].fillna("no_date_available")
print("\nDistribución de split_bucket a nivel de FILAS (estudiante-curso) del dataset final:")
print(final_df["split_bucket"].value_counts())
print(f"\nTotal filas dataset_final_3_x.csv: {len(final_df)}")

course_dates.to_csv(os.path.join(OUT_DIR, "course_year_map.csv"), index=False)
final_df[["uid_hash", "course_hash", "min_date", "year", "split_bucket"]].to_csv(
    os.path.join(OUT_DIR, "row_split_bucket.csv"), index=False
)
print(f"\nGuardado: {os.path.join(OUT_DIR, 'course_year_map.csv')}")
print(f"Guardado: {os.path.join(OUT_DIR, 'row_split_bucket.csv')}")
