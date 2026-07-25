"""
v3.8 — Etapa 2, palanca 1: recomputar los candidatos de accesos/foros
(mismo pool de 15 columnas que probó `code/v3.7/`) usando SOLO eventos
hasta el corte de 25/50/75%. Misma lógica que
`03_recompute_features_at_cutoff.py` pero para
`01-preprocessing/accesos/02_consolida_accesos.py` y
`01-preprocessing/foros/02_consolida_foros.py`.

Motivación (pedido de Miguel): al 25-50% del curso muchos estudiantes
todavía no tienen exámenes (ver `coverage_report.json` de la Etapa 1),
pero los logs de acceso existen desde el día 1 -- en cortes tempranos su
peso relativo debería ser mayor que en v3.7 (que solo probó el curso
completo, +0.016 global).

Fuente: `code/v3.6/data/1. Para tratar/accesos/accesos_normalizado.csv` y
`.../foros/foros_normalizado.csv` -- ya tienen `curso` directamente (a
diferencia de exámenes), no hace falta el mapa groupKey->course_hash del
Paso 1, solo filtrar por curso objetivo + fecha.

No se recalcula la imputación de `tiempo` (`is_imputed`, ya normalizado
sobre TODO el dataset por `01_normaliza_accesos.py`) -- es una estadística
poblacional por (email,dispositivo), igual tratamiento que la
normalización de calificaciones de tareas (ver docstring de
`03_recompute_features_at_cutoff.py`): no depende de cuándo se hace el
corte, es una propiedad de la sesión individual.

Output (por corte): out/cutoff_X_access_forum.csv
  (uid_hash, course_hash, + las 15 columnas candidatas de accesos/foros,
  mismos nombres finales que v3.7: total_access_time, avg_access_time,
  access_days, access_sessions_desktop, access_sessions_mobile,
  access_imputed_desktop, access_imputed_mobile, total_access_sessions,
  total_access_imputed, access_imputed_ratio, access_missing_data,
  forum_interactions, forum_threads, forum_time_range, forum_missing_data)
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("V38_OUT_DIR", os.path.join(HERE, "out"))
os.makedirs(OUT_DIR, exist_ok=True)
V36_DATA_1 = os.path.join(HERE, "..", "v3.6", "data", "1. Para tratar")

valid_triples = pd.read_csv(os.path.join(OUT_DIR, "valid_triples.csv"))
course_windows = pd.read_csv(os.path.join(OUT_DIR, "course_windows.csv"), parse_dates=["cutoff_25", "cutoff_50", "cutoff_75"])
target_courses = set(valid_triples["course_hash"].unique())
all_target_rows = valid_triples[["uid_hash", "course_hash"]].drop_duplicates()

accesos = pd.read_csv(os.path.join(V36_DATA_1, "accesos", "accesos_normalizado.csv"))
accesos = accesos[accesos["curso"].isin(target_courses)].copy()
accesos["fecha_ingreso"] = pd.to_datetime(accesos["fecha_ingreso"], errors="coerce")
accesos["dispositivo"] = accesos["dispositivo"].str.strip()
accesos["desktop_sessions"] = (accesos["dispositivo"] == "Desktop").astype(int)
accesos["mobile_sessions"] = (accesos["dispositivo"] == "Mobile").astype(int)
accesos["desktop_imputed_sessions"] = ((accesos["dispositivo"] == "Desktop") & (accesos["is_imputed"] == 1)).astype(int)
accesos["mobile_imputed_sessions"] = ((accesos["dispositivo"] == "Mobile") & (accesos["is_imputed"] == 1)).astype(int)
accesos = accesos.merge(course_windows[["course_hash", "cutoff_25", "cutoff_50", "cutoff_75"]],
                         left_on="curso", right_on="course_hash", how="left")
print(f"Eventos de acceso restringidos a cursos objetivo: {len(accesos)}")

foros = pd.read_csv(os.path.join(V36_DATA_1, "foros", "foros_normalizado.csv"))
foros = foros[foros["curso"].isin(target_courses)].copy()
foros["fecha_y_hora"] = pd.to_datetime(foros["fecha_y_hora"], errors="coerce")
foros = foros.merge(course_windows[["course_hash", "cutoff_25", "cutoff_50", "cutoff_75"]],
                     left_on="curso", right_on="course_hash", how="left")
print(f"Eventos de foro restringidos a cursos objetivo: {len(foros)}")


def aggregate_accesos(df_events: pd.DataFrame) -> pd.DataFrame:
    g = df_events.groupby(["email", "curso"], as_index=False).agg(
        total_access_time=("tiempo", "sum"),
        avg_access_time=("tiempo", "mean"),
        access_days=("fecha_ingreso", "nunique"),
        access_sessions_desktop=("desktop_sessions", "sum"),
        access_sessions_mobile=("mobile_sessions", "sum"),
        access_imputed_desktop=("desktop_imputed_sessions", "sum"),
        access_imputed_mobile=("mobile_imputed_sessions", "sum"),
    )
    g["total_access_sessions"] = g["access_sessions_desktop"] + g["access_sessions_mobile"]
    g["total_access_imputed"] = g["access_imputed_desktop"] + g["access_imputed_mobile"]
    g["access_imputed_ratio"] = (g["total_access_imputed"] / g["total_access_sessions"]).fillna(0)
    g = g.rename(columns={"email": "uid_hash", "curso": "course_hash"})
    return g


def aggregate_foros(df_events: pd.DataFrame) -> pd.DataFrame:
    g = df_events.groupby(["email", "curso"], as_index=False).agg(
        forum_interactions=("id_foro", "count"),
        forum_threads=("id_foro", "nunique"),
        _first=("fecha_y_hora", "min"),
        _last=("fecha_y_hora", "max"),
    )
    g["forum_time_range"] = (g["_last"] - g["_first"]).dt.total_seconds() / 60.0
    g = g.drop(columns=["_first", "_last"]).rename(columns={"email": "uid_hash", "curso": "course_hash"})
    return g


ACCESS_FORUM_COLS = [
    "total_access_time", "avg_access_time", "access_days", "access_sessions_desktop",
    "access_sessions_mobile", "access_imputed_desktop", "access_imputed_mobile",
    "total_access_sessions", "total_access_imputed", "access_imputed_ratio",
    "access_missing_data", "forum_interactions", "forum_threads", "forum_time_range",
    "forum_missing_data",
]

for pct in (25, 50, 75):
    cutoff_col = f"cutoff_{pct}"
    a_cut = accesos[accesos["fecha_ingreso"] <= accesos[cutoff_col]]
    f_cut = foros[foros["fecha_y_hora"] <= foros[cutoff_col]]
    print(f"\nCorte {pct}%: accesos {len(a_cut)}/{len(accesos)} ({len(a_cut)/max(len(accesos),1)*100:.1f}%), "
          f"foros {len(f_cut)}/{len(foros)} ({len(f_cut)/max(len(foros),1)*100:.1f}%)")

    a_agg = aggregate_accesos(a_cut)
    f_agg = aggregate_foros(f_cut)

    out_df = all_target_rows.merge(a_agg, on=["uid_hash", "course_hash"], how="left")
    out_df = out_df.merge(f_agg, on=["uid_hash", "course_hash"], how="left")

    out_df["access_missing_data"] = out_df["total_access_time"].isna().astype(int)
    out_df["forum_missing_data"] = out_df["forum_interactions"].isna().astype(int)

    # conteos -> 0 cuando no hay evento aún (info genuina, no NaN a imputar)
    count_like = ["total_access_time", "access_days", "access_sessions_desktop", "access_sessions_mobile",
                  "access_imputed_desktop", "access_imputed_mobile", "total_access_sessions",
                  "total_access_imputed", "access_imputed_ratio", "forum_interactions", "forum_threads"]
    for c in count_like:
        out_df[c] = out_df[c].fillna(0)
    # avg_access_time / forum_time_range: NaN genuino si no hay eventos -> imputación median aguas abajo

    n_access = int((out_df["total_access_time"] > 0).sum())
    n_forum = int((out_df["forum_interactions"] > 0).sum())
    print(f"  Cobertura: {n_access}/{len(out_df)} ({n_access/len(out_df)*100:.1f}%) con >=1 acceso, "
          f"{n_forum}/{len(out_df)} ({n_forum/len(out_df)*100:.1f}%) con >=1 interacción de foro")

    out_df = out_df[["uid_hash", "course_hash"] + ACCESS_FORUM_COLS]
    out_path = os.path.join(OUT_DIR, f"cutoff_{pct}_access_forum.csv")
    out_df.to_csv(out_path, index=False)
    print(f"  Guardado: {out_path} ({out_df.shape})")
