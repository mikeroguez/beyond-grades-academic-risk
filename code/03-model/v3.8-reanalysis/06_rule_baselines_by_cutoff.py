"""
v3.8 reanálisis — Etapa 6: baselines de regla por corte.

Reviewer 1 pidió comparadores sencillos tipo "last observed value" o
"cumulative average". Esta etapa evalúa reglas deterministas sobre los
mismos cortes 25/50/75 ya reconstruidos en `code/v3.8/`, sin entrenar DNN,
SHAP ni modelos pesados.

Reglas evaluadas:
- avg_assignment_score: promedio acumulado de calificación de tareas hasta
  el corte, usando `calificacion_normalizada` y reescalado a la escala del
  target con el pool de entrenamiento de cada fold / holdout.
- missing_assignments: conteo acumulado de tareas no entregadas hasta el
  corte, reescalado linealmente a la escala del target usando solo el pool
  de entrenamiento de cada fold / holdout.
- assignment_procrast_rate: late_assignments/submitted_assignments acumulado
  hasta el corte.
- avg_exam_score: promedio acumulado de calificación de exámenes hasta el
  corte, usando `normalized_score` y reescalado a la escala del target con
  el pool de entrenamiento de cada fold / holdout.
- exam_accuracy: correct_answers/total_questions acumulado hasta el corte.

Cuando una regla no tiene observaciones tempranas suficientes para una fila
(por ejemplo, ningún examen antes del corte), se imputa con la media del
target en el conjunto de entrenamiento correspondiente. Esto define un
baseline simple y honesto: "usar el valor acumulado si existe; si no, usar
la media histórica de entrenamiento".
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


HERE = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get("V38_REANALYSIS_OUT_DIR", HERE / "out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

V38_OUT = Path(os.environ.get("V38_OUT_DIR", HERE / ".." / ".." / "v3.8" / "out")).resolve()
V36_REANALYSIS_OUT = (HERE / ".." / "v3.6-reanalysis" / "out").resolve()
V36_DATA_1 = (HERE / ".." / ".." / "v3.6" / "data" / "1. Para tratar").resolve()

TARGETS = [
    "avg_assignment_score",
    "missing_assignments",
    "assignment_procrast_rate",
    "avg_exam_score",
    "exam_accuracy",
]
N_FOLDS = 5


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def metrics_per_target(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        yt = y_true_df[target].to_numpy(dtype=float)
        yp = y_pred_df[target].to_numpy(dtype=float)
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rows.append(
            {
                "target": target,
                "R2": float(r2),
                "MAE": float(np.mean(np.abs(yt - yp))),
                "RMSE": _rmse(yt, yp),
            }
        )
    return pd.DataFrame(rows)


def aggregate_global(per_target_df: pd.DataFrame) -> dict[str, float]:
    return {
        "MAE_arithmetic_mean": float(per_target_df["MAE"].mean()),
        "RMSE_arithmetic_mean": float(per_target_df["RMSE"].mean()),
        "R2_arithmetic_mean": float(per_target_df["R2"].mean()),
    }


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean = pd.read_csv(V36_REANALYSIS_OUT / "clean_dataset.csv")
    split_info = clean[["uid_hash", "course_hash", "component_id", "split_bucket"] + TARGETS].copy()

    valid_triples = pd.read_csv(V38_OUT / "valid_triples.csv")
    course_windows = pd.read_csv(
        V38_OUT / "course_windows.csv",
        parse_dates=["start_date", "end_date", "cutoff_25", "cutoff_50", "cutoff_75"],
    )
    triples = valid_triples.merge(
        course_windows[["course_hash", "start_date", "end_date", "cutoff_25", "cutoff_50", "cutoff_75"]],
        on="course_hash",
        how="left",
    )

    tareas = pd.read_csv(V36_DATA_1 / "tareas" / "tareas_normalizado.csv")
    tareas["fecha_compromiso"] = pd.to_datetime(tareas["fecha_compromiso"], errors="coerce")
    tareas = tareas.merge(
        triples,
        left_on=["curso", "email", "groupKey"],
        right_on=["course_hash", "uid_hash", "groupKey"],
        how="inner",
    )

    examenes = pd.read_csv(V36_DATA_1 / "examenes" / "examenes_sin_los_de_otra_materia.csv")
    examenes["start_date_event"] = pd.to_datetime(examenes["start_date"], errors="coerce")
    examenes = examenes.merge(
        triples,
        left_on=["groupKey", "email"],
        right_on=["groupKey", "uid_hash"],
        how="inner",
        suffixes=("_event_original", ""),
    )
    return split_info, triples, tareas, examenes


def build_rule_features_for_cutoff(
    pct: int,
    split_info: pd.DataFrame,
    triples: pd.DataFrame,
    tareas: pd.DataFrame,
    examenes: pd.DataFrame,
) -> pd.DataFrame:
    cutoff_col = f"cutoff_{pct}"
    base = split_info[["uid_hash", "course_hash", "component_id", "split_bucket"] + TARGETS].copy()

    t_cut = tareas[tareas["fecha_compromiso"] <= tareas[cutoff_col]].copy()
    e_cut = examenes[examenes["start_date_event"] <= examenes[cutoff_col]].copy()

    if not t_cut.empty:
        t_agg = (
            t_cut.groupby(["uid_hash", "course_hash"])
            .agg(
                avg_assignment_observed=("calificacion_normalizada", "mean"),
                missing_assignments_observed=("fue_entregada", lambda x: (x == 0).sum()),
                submitted_assignments_observed=("fue_entregada", "sum"),
                late_assignments_observed=("tipo_entrega", lambda x: (x == "tardía").sum()),
            )
            .reset_index()
        )
        t_agg["rule_assignment_procrast_rate"] = np.where(
            t_agg["submitted_assignments_observed"] > 0,
            t_agg["late_assignments_observed"] / t_agg["submitted_assignments_observed"],
            np.nan,
        )
    else:
        t_agg = pd.DataFrame(columns=["uid_hash", "course_hash"])

    if not e_cut.empty:
        e_agg = (
            e_cut.groupby(["uid_hash", "course_hash"])
            .agg(
                avg_exam_observed=("normalized_score", "mean"),
                correct_answers_observed=("CorrectAnswers", "sum"),
                questions_observed=("TotalQuestions", "sum"),
            )
            .reset_index()
        )
        e_agg["rule_exam_accuracy"] = np.where(
            e_agg["questions_observed"] > 0,
            e_agg["correct_answers_observed"] / e_agg["questions_observed"],
            np.nan,
        )
    else:
        e_agg = pd.DataFrame(columns=["uid_hash", "course_hash"])

    out = base.merge(t_agg, on=["uid_hash", "course_hash"], how="left")
    out = out.merge(e_agg, on=["uid_hash", "course_hash"], how="left")
    if "missing_assignments_observed" not in out:
        out["missing_assignments_observed"] = np.nan

    return out


def scale_observed_to_target(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    observed_col: str,
    target: str,
) -> pd.Series:
    train_values = train_df[observed_col].dropna() if observed_col in train_df else pd.Series(dtype=float)
    y_train = train_df[target]

    if len(train_values) > 0 and train_values.max() > train_values.min():
        observed_min = float(train_values.min())
        observed_max = float(train_values.max())
        target_min = float(y_train.min())
        target_max = float(y_train.max())
        values = (eval_df[observed_col] - observed_min) / (observed_max - observed_min)
        values = values * (target_max - target_min) + target_min
    else:
        values = pd.Series(np.nan, index=eval_df.index)

    return values.fillna(float(y_train.mean())).clip(lower=0.0, upper=1.0)


def fill_and_scale_predictions(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    preds = pd.DataFrame(index=eval_df.index)
    direct_rule_cols = {
        "assignment_procrast_rate": "rule_assignment_procrast_rate",
        "exam_accuracy": "rule_exam_accuracy",
    }

    for target, col in direct_rule_cols.items():
        y_mean = float(train_df[target].mean())
        values = eval_df[col] if col in eval_df else pd.Series(np.nan, index=eval_df.index)
        preds[target] = values.fillna(y_mean).clip(lower=0.0, upper=1.0)

    # These observed quantities are pre-target-scale values. Use only the
    # training split to map each rule onto the corresponding target scale.
    scale_rules = {
        "avg_assignment_score": "avg_assignment_observed",
        "missing_assignments": "missing_assignments_observed",
        "avg_exam_score": "avg_exam_observed",
    }
    for target, col in scale_rules.items():
        preds[target] = scale_observed_to_target(train_df, eval_df, col, target)

    return preds[TARGETS]


def evaluate_rule_baseline(df: pd.DataFrame) -> dict:
    train_pool = df[df["split_bucket"] == "train_le2023"].reset_index(drop=True)
    holdout = df[df["split_bucket"] == "holdout_2024"].reset_index(drop=True)

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_metrics = []
    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(train_pool, groups=train_pool["component_id"])):
        tr = train_pool.iloc[tr_idx]
        va = train_pool.iloc[va_idx]
        preds = fill_and_scale_predictions(tr, va)
        metrics = metrics_per_target(va[TARGETS], preds)
        metrics["fold"] = fold_i
        fold_metrics.append(metrics)

    all_folds = pd.concat(fold_metrics, ignore_index=True)
    per_target_cv = all_folds.groupby("target")[["R2", "MAE", "RMSE"]].mean().reindex(TARGETS).reset_index()

    holdout_preds = fill_and_scale_predictions(train_pool, holdout)
    holdout_metrics = metrics_per_target(holdout[TARGETS], holdout_preds)

    coverage = {}
    rule_cols = {
        "avg_assignment_score": "avg_assignment_observed",
        "missing_assignments": "missing_assignments_observed",
        "assignment_procrast_rate": "rule_assignment_procrast_rate",
        "avg_exam_score": "avg_exam_observed",
        "exam_accuracy": "rule_exam_accuracy",
    }
    for target, col in rule_cols.items():
        coverage[target] = {
            "train_le2023_nonmissing_pct": float(train_pool[col].notna().mean() * 100.0),
            "holdout_2024_nonmissing_pct": float(holdout[col].notna().mean() * 100.0),
        }

    return {
        "cv_grouped_le2023": {
            "per_target_mean_over_folds": per_target_cv.to_dict(orient="records"),
            "global": aggregate_global(per_target_cv),
        },
        "holdout_2024": {
            "per_target": holdout_metrics.to_dict(orient="records"),
            "global": aggregate_global(holdout_metrics),
        },
        "coverage": coverage,
    }


def main() -> None:
    split_info, triples, tareas, examenes = read_inputs()
    results_by_cutoff = {}
    summary_rows = []

    for pct in (25, 50, 75):
        df = build_rule_features_for_cutoff(pct, split_info, triples, tareas, examenes)
        out_csv = OUT_DIR / f"06_rule_features_cutoff{pct}.csv"
        df.to_csv(out_csv, index=False)

        result = evaluate_rule_baseline(df)
        results_by_cutoff[f"cutoff_{pct}"] = result
        cv = result["cv_grouped_le2023"]
        print(f"\ncutoff {pct}% rule baseline")
        print(f"  CV global R2={cv['global']['R2_arithmetic_mean']:.4f}, "
              f"MAE={cv['global']['MAE_arithmetic_mean']:.4f}, RMSE={cv['global']['RMSE_arithmetic_mean']:.4f}")
        for row in cv["per_target_mean_over_folds"]:
            print(f"  {row['target']}: R2={row['R2']:.4f}, MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}")
            summary_rows.append(
                {
                    "cutoff": pct,
                    "target": row["target"],
                    "R2": row["R2"],
                    "MAE": row["MAE"],
                    "RMSE": row["RMSE"],
                    "coverage_train_pct": result["coverage"][row["target"]]["train_le2023_nonmissing_pct"],
                    "coverage_holdout_pct": result["coverage"][row["target"]]["holdout_2024_nonmissing_pct"],
                }
            )
        summary_rows.append(
            {
                "cutoff": pct,
                "target": "Global (arithmetic mean)",
                "R2": cv["global"]["R2_arithmetic_mean"],
                "MAE": cv["global"]["MAE_arithmetic_mean"],
                "RMSE": cv["global"]["RMSE_arithmetic_mean"],
                "coverage_train_pct": np.nan,
                "coverage_holdout_pct": np.nan,
            }
        )

    out_json = OUT_DIR / "06_rule_baselines_by_cutoff_results.json"
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "description": "Rule-based cumulative baselines for 25/50/75% course-completion cutoffs.",
                "targets": TARGETS,
                "results_by_cutoff": results_by_cutoff,
            },
            handle,
            indent=2,
        )
    out_summary = OUT_DIR / "06_rule_baselines_by_cutoff_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    print(f"\nGuardado: {out_json}")
    print(f"Guardado: {out_summary}")


if __name__ == "__main__":
    main()
