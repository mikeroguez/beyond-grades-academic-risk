"""
v3.8 reanálisis — Etapa 3, paso 0: ensamblar un CSV limpio por corte
(25/50/75%) con las features RESELECCIONADAS de la Etapa 2 (Pearson+LassoCV
train-only, distintas por corte -- 21/18/20 features respectivamente) +
los 5 targets de fin de curso + component_id/split_bucket, listo para
alimentar la búsqueda de hiperparámetros de la DNN (04_dnn_hp_search_by_cutoff.py).

No recalcula nada -- solo une columnas ya calculadas en
`code/v3.8/out/cutoff_X_{features,access_forum,early_features}.csv` y
`code/03-model/v3.6-reanalysis/out/clean_dataset.csv` (targets/splits),
filtrando a las columnas que `02_etapa2_ablation.py` ya determinó como la
selección final de cada corte.
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("V38_REANALYSIS_OUT_DIR", os.path.join(HERE, "out"))
os.makedirs(OUT_DIR, exist_ok=True)
V38_OUT = os.environ.get("V38_OUT_DIR", os.path.join(HERE, "..", "..", "v3.8", "out"))
V36_REANALYSIS_OUT = os.path.join(HERE, "..", "v3.6-reanalysis", "out")

TARGETS = ["avg_assignment_score", "missing_assignments", "assignment_procrast_rate", "avg_exam_score", "exam_accuracy"]

with open(os.path.join(OUT_DIR, "02_etapa2_ablation_results.json")) as f:
    etapa2 = json.load(f)

selected_by_cutoff = {
    25: etapa2["curve_combined_with_reselection"]["cutoff_25"]["selected_features"],
    50: etapa2["cutoff_50_reselection"]["selected_features"],
    75: etapa2["curve_combined_with_reselection"]["cutoff_75"]["selected_features"],
}

v36_clean = pd.read_csv(os.path.join(V36_REANALYSIS_OUT, "clean_dataset.csv"))
split_info = v36_clean[["uid_hash", "course_hash", "component_id", "split_bucket"] + TARGETS]

for pct, selected_features in selected_by_cutoff.items():
    feats = pd.read_csv(os.path.join(V38_OUT, f"cutoff_{pct}_features.csv"))
    af = pd.read_csv(os.path.join(V38_OUT, f"cutoff_{pct}_access_forum.csv"))
    ef = pd.read_csv(os.path.join(V38_OUT, f"cutoff_{pct}_early_features.csv"))

    df = split_info.merge(feats, on=["uid_hash", "course_hash"], how="left")
    df = df.merge(af, on=["uid_hash", "course_hash"], how="left")
    df = df.merge(ef, on=["uid_hash", "course_hash"], how="left")

    missing_cols = [c for c in selected_features if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Corte {pct}%: faltan columnas seleccionadas: {missing_cols}")

    keep_cols = ["uid_hash", "course_hash", "component_id", "split_bucket"] + TARGETS + selected_features
    out_df = df[keep_cols].copy()

    out_path = os.path.join(OUT_DIR, f"cutoff_{pct}_clean_dataset.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Corte {pct}%: {out_df.shape} ({len(selected_features)} features) -> {out_path}")

with open(os.path.join(OUT_DIR, "03_selected_features_by_cutoff.json"), "w") as f:
    json.dump({str(k): v for k, v in selected_by_cutoff.items()}, f, indent=2)
print(f"Guardado: {os.path.join(OUT_DIR, '03_selected_features_by_cutoff.json')}")
