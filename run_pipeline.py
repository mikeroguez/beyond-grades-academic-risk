#!/usr/bin/env python3
"""Run the full reanalysis pipeline end to end: the 100%-course benchmark
(`code/03-model/v3.6-reanalysis/`, Table 5/7, SHAP Fig. 6, clustering
Fig. 8) followed by the v3.8 temporal-cutoff pipeline (Table 9, Sec. 5.3),
with isolated outputs, logging, and resume. The two are chained here
because v3.8's own final packaging step and its "100%" reference point in
the cutoff curve both read directly from the 100%-course benchmark's
output -- until 21 jul 2026 that meant running the 100%-course scripts by
hand before ever touching this file.

This runner orchestrates existing scripts. It does not reimplement model
logic or change the scientific methodology.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
V38_DIR = CODE_DIR / "v3.8"
V38_REANALYSIS_DIR = CODE_DIR / "03-model" / "v3.8-reanalysis"
V36_REANALYSIS_DIR = CODE_DIR / "03-model" / "v3.6-reanalysis"
V36_REANALYSIS_OUT = V36_REANALYSIS_DIR / "out"
DEFAULT_RUNS_DIR = ROOT / "pipeline_runs"
SEED = 42
CUTOFFS = ("25", "50", "75")
TARGETS = (
    "avg_assignment_score",
    "missing_assignments",
    "assignment_procrast_rate",
    "avg_exam_score",
    "exam_accuracy",
)


@dataclass(frozen=True)
class Stage:
    name: str
    label: str
    script: Path
    cwd: Path
    outputs: tuple[Path, ...]
    env: dict[str, str] = field(default_factory=dict)
    long_running: bool = False


class Tee:
    def __init__(self, *streams, quiet: bool = False):
        self.streams = streams
        self.quiet = quiet

    def write(self, text: str) -> None:
        for stream in self.streams:
            if self.quiet and stream is sys.stdout:
                continue
            stream.write(text)
            stream.flush()


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_run_id() -> str:
    return datetime.now().strftime("v38_%Y%m%d_%H%M%S")


def duration_text(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def make_stages(v38_out: Path, reanalysis_out: Path, smoke_test: bool) -> list[Stage]:
    suffix = "_smoketest" if smoke_test else ""
    dnn_env = {
        "DNN_SEARCH_MAX_TRIALS": "2",
        "DNN_SEARCH_INITIAL_POINTS": "1",
        "DNN_SEARCH_EPOCHS": "3",
        "DNN_SEARCH_PATIENCE": "1",
        "DNN_FINAL_EPOCHS": "3",
        "DNN_FINAL_PATIENCE": "1",
        "DNN_RESULTS_SUFFIX": suffix,
    } if smoke_test else {}
    shap_env = {
        "SHAP_K_BG": "5",
        "SHAP_N_SAMPLE_MAX": "8",
        "SHAP_MODEL_SUFFIX": suffix,
    } if smoke_test else {}

    # Etapas del benchmark de 100% (v3.6-reanalysis). A diferencia de las
    # etapas v3.8 de abajo, estos scripts todavia no soportan redirigir su
    # salida via variables de entorno -- escriben siempre en su propia
    # carpeta fija `V36_REANALYSIS_OUT` (igual que hoy), no en una carpeta
    # aislada por corrida. El mecanismo de resume/skip funciona igual
    # (compara contra esos archivos fijos); lo que no hay es aislamiento
    # entre corridas para estas etapas especificas -- una corrida nueva
    # sobreescribe la anterior ahi. No soportan --smoke-test (siempre
    # corren completas: el 40-trial Bayesiano de la DNN es el paso largo).
    main_stages: list[Stage] = [
        Stage(
            "main_clean_dataset",
            "100%: construir clean_dataset.csv + feature_lists.json (exclusion de fuga)",
            V36_REANALYSIS_DIR / "03_build_clean_dataset.py",
            V36_REANALYSIS_DIR,
            (V36_REANALYSIS_OUT / "clean_dataset.csv", V36_REANALYSIS_OUT / "feature_lists.json"),
        ),
        Stage(
            "main_feature_selection",
            "100%: seleccion Pearson+LassoCV (informativa, no poda features)",
            V36_REANALYSIS_DIR / "04_feature_selection_train_only.py",
            V36_REANALYSIS_DIR,
            (V36_REANALYSIS_OUT / "04_feature_selection_train_only.json",),
        ),
        Stage(
            "main_baselines",
            "100%: baselines LR/DT/RF, split agrupado + holdout temporal",
            V36_REANALYSIS_DIR / "05_baselines_grouped_temporal.py",
            V36_REANALYSIS_DIR,
            (V36_REANALYSIS_OUT / "05_baselines_grouped_temporal_results.json",),
        ),
        Stage(
            "main_dnn_hp_search",
            "100%: DNN multi-tarea, busqueda Bayesiana de 40 trials (el paso largo)",
            V36_REANALYSIS_DIR / "08_dnn_hp_search.py",
            V36_REANALYSIS_DIR,
            (
                V36_REANALYSIS_OUT / "08_dnn_tuned_results.json",
                V36_REANALYSIS_OUT / "08_best_hyperparameters.json",
                V36_REANALYSIS_OUT / "08_tuner_trial_history.json",
                V36_REANALYSIS_OUT / "08_final_tuned_model.keras",
                V36_REANALYSIS_OUT / "08_final_imputer.joblib",
                V36_REANALYSIS_OUT / "08_final_scaler.joblib",
            ),
            long_running=True,
        ),
        Stage(
            "main_learning_curve",
            "100%: generar curva loss/val_loss del entrenamiento final DNN (Fig. 5)",
            V36_REANALYSIS_DIR / "08_plot_learning_curve.py",
            V36_REANALYSIS_DIR,
            (
                V36_REANALYSIS_OUT / "08_final_learning_curve.png",
                V36_REANALYSIS_OUT / "08_final_training_history.csv",
                V36_REANALYSIS_OUT / "08_final_learning_curve_source.json",
            ),
        ),
        Stage(
            "main_shap",
            "100%: SHAP sobre el modelo tuneado (Fig. 6 + Apendice C)",
            V36_REANALYSIS_DIR / "09_shap_analysis.py",
            V36_REANALYSIS_DIR,
            (V36_REANALYSIS_OUT / "09_shap_summary.json",)
            + tuple(V36_REANALYSIS_OUT / f"09_shap_summary_{target}.png" for target in TARGETS)
            + tuple(V36_REANALYSIS_OUT / f"09_shap_values_{target}.npy" for target in TARGETS),
            long_running=True,
        ),
        Stage(
            "main_clustering",
            "100%: clustering sobre top-11 SHAP (Fig. 8, Tabla 11)",
            V36_REANALYSIS_DIR / "06_clustering_clean.py",
            V36_REANALYSIS_DIR,
            (
                V36_REANALYSIS_OUT / "06_clustering_clean_results.json",
                V36_REANALYSIS_OUT / "06_pca_scatter_CLEAN.png",
                V36_REANALYSIS_OUT / "06_cluster_means_CLEAN_k7.csv",
            ),
        ),
    ]

    stages: list[Stage] = [
        *main_stages,
        Stage(
            "valid_triples",
            "Construir mapa uid/course -> groupKey",
            V38_DIR / "01_build_valid_triples.py",
            V38_DIR,
            (v38_out / "valid_triples.csv",),
        ),
        Stage(
            "course_windows",
            "Construir ventanas de curso y cortes 25/50/75",
            V38_DIR / "02_build_course_windows.py",
            V38_DIR,
            (v38_out / "course_windows.csv",),
        ),
        Stage(
            "cutoff_clean_features",
            "Recomputar features limpias por corte (shared_clean_features.CLEAN_FEATURES)",
            V38_DIR / "03_recompute_features_at_cutoff.py",
            V38_DIR,
            tuple(v38_out / f"cutoff_{pct}_features.csv" for pct in CUTOFFS)
            + (v38_out / "coverage_report.json",),
        ),
        Stage(
            "access_forum_features",
            "Recomputar features de accesos/foros por corte",
            V38_DIR / "04_recompute_access_forum_at_cutoff.py",
            V38_DIR,
            tuple(v38_out / f"cutoff_{pct}_access_forum.csv" for pct in CUTOFFS),
        ),
        Stage(
            "early_features",
            "Construir features tempranas por corte",
            V38_DIR / "05_build_early_features.py",
            V38_DIR,
            tuple(v38_out / f"cutoff_{pct}_early_features.csv" for pct in CUTOFFS),
        ),
        Stage(
            "audit_new_features",
            "Auditar fuga en features nuevas",
            V38_DIR / "06_audit_new_features.py",
            V38_DIR,
            (v38_out / "06_audit_new_features_results.json",),
        ),
        Stage(
            "baselines_by_cutoff",
            "Correr baselines por corte",
            V38_REANALYSIS_DIR / "01_baselines_by_cutoff.py",
            V38_REANALYSIS_DIR,
            (reanalysis_out / "01_baselines_by_cutoff_results.json",),
        ),
        Stage(
            "etapa2_ablation",
            "Correr ablacion Etapa 2 y reseleccion Pearson+LassoCV",
            V38_REANALYSIS_DIR / "02_etapa2_ablation.py",
            V38_REANALYSIS_DIR,
            (reanalysis_out / "02_etapa2_ablation_results.json",),
        ),
        Stage(
            "rule_baselines_by_cutoff",
            "Correr baselines acumulativos de regla por corte",
            V38_REANALYSIS_DIR / "06_rule_baselines_by_cutoff.py",
            V38_REANALYSIS_DIR,
            (reanalysis_out / "06_rule_baselines_by_cutoff_results.json", reanalysis_out / "06_rule_baselines_by_cutoff_summary.csv"),
        ),
        Stage(
            "build_cutoff_datasets",
            "Ensamblar datasets finales por corte para DNN",
            V38_REANALYSIS_DIR / "03_build_cutoff_datasets.py",
            V38_REANALYSIS_DIR,
            tuple(reanalysis_out / f"cutoff_{pct}_clean_dataset.csv" for pct in CUTOFFS)
            + (reanalysis_out / "03_selected_features_by_cutoff.json",),
        ),
    ]

    for pct in CUTOFFS:
        stages.append(
            Stage(
                f"dnn_hp_cutoff_{pct}",
                f"DNN tuneada para corte {pct}%",
                V38_REANALYSIS_DIR / "04_dnn_hp_search_by_cutoff.py",
                V38_REANALYSIS_DIR,
                (
                    reanalysis_out / f"04_dnn_tuned_results_cutoff{pct}{suffix}.json",
                    reanalysis_out / f"04_best_hyperparameters_cutoff{pct}{suffix}.json",
                    reanalysis_out / f"04_tuner_trial_history_cutoff{pct}{suffix}.json",
                    reanalysis_out / f"04_final_tuned_model_cutoff{pct}{suffix}.keras",
                    reanalysis_out / f"04_final_imputer_cutoff{pct}{suffix}.joblib",
                    reanalysis_out / f"04_final_scaler_cutoff{pct}{suffix}.joblib",
                ),
                env={"CUTOFF": pct, **dnn_env},
                long_running=True,
            )
        )

    for pct in CUTOFFS:
        stages.append(
            Stage(
                f"shap_cutoff_{pct}",
                f"SHAP para corte {pct}%",
                V38_REANALYSIS_DIR / "05_shap_by_cutoff.py",
                V38_REANALYSIS_DIR,
                (reanalysis_out / f"05_shap_summary_cutoff{pct}{suffix}.json",)
                + tuple(
                    reanalysis_out / f"05_shap_values_{target}_cutoff{pct}{suffix}.npy"
                    for target in TARGETS
                )
                + tuple(
                    reanalysis_out / f"05_shap_summary_{target}_cutoff{pct}{suffix}.png"
                    for target in TARGETS
                ),
                env={"CUTOFF": pct, **shap_env},
                long_running=True,
            )
        )

    return stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full reanalysis pipeline (100% benchmark + v3.8 cutoffs) with isolated outputs and resumable stages."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Run output directory. Defaults to pipeline_runs/v38_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume a previous run directory.")
    parser.add_argument("--force", action="store_true", help="Run all stages even if outputs already exist.")
    parser.add_argument("--start-from", help="Start from a specific stage name.")
    parser.add_argument("--quiet", action="store_true", help="Only write detailed output to logs.")
    parser.add_argument("--verbose", action="store_true", help="Print full subprocess output live.")
    parser.add_argument(
        "--tf-cpu",
        action="store_true",
        help="Force TensorFlow stages to hide Metal/GPU devices. Useful when Apple Metal stalls.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use tiny DNN/SHAP settings and suffixed outputs for a fast orchestration test.",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Print stage names and exit.",
    )
    return parser.parse_args()


def output_dir_from_args(args: argparse.Namespace) -> Path:
    if args.output:
        return args.output.expanduser().resolve()
    return (DEFAULT_RUNS_DIR / safe_run_id()).resolve()


def load_status(path: Path) -> dict:
    if not path.exists():
        return {"stages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def all_outputs_exist(outputs: Iterable[Path]) -> bool:
    return all(path.exists() for path in outputs)


def collect_generated_files(run_dir: Path) -> list[dict]:
    generated: list[dict] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        generated.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "size_bytes": path.stat().st_size,
            }
        )
    return generated


def build_env(args: argparse.Namespace, v38_out: Path, reanalysis_out: Path, stage: Stage) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(SEED)
    env["TF_DETERMINISTIC_OPS"] = env.get("TF_DETERMINISTIC_OPS", "1")
    env["V38_OUT_DIR"] = str(v38_out)
    env["V38_REANALYSIS_OUT_DIR"] = str(reanalysis_out)
    env["MPLCONFIGDIR"] = str(v38_out.parent / ".matplotlib")
    if args.tf_cpu:
        env["V38_TF_CPU_ONLY"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    env.update(stage.env)
    return env


def run_stage(
    args: argparse.Namespace,
    stage: Stage,
    stage_index: int,
    total_stages: int,
    run_dir: Path,
    log_dir: Path,
    v38_out: Path,
    reanalysis_out: Path,
    status: dict,
    pipeline_log,
) -> None:
    pct = stage_index / total_stages * 100
    stage_status = status.setdefault("stages", {}).get(stage.name, {})
    completed = stage_status.get("status") == "completed" and all_outputs_exist(stage.outputs)

    if args.resume and completed and not args.force:
        msg = f"[{timestamp()}] [{stage_index}/{total_stages} {pct:5.1f}%] SKIP {stage.name}: ya completada\n"
        Tee(sys.stdout, pipeline_log, quiet=args.quiet).write(msg)
        return

    if not stage.script.exists():
        raise FileNotFoundError(f"No existe el script de etapa: {stage.script}")

    stage_log_path = log_dir / f"{stage_index:02d}_{stage.name}.log"
    start = time.time()
    start_iso = timestamp()
    header = (
        f"\n[{start_iso}] [{stage_index}/{total_stages} {pct:5.1f}%] INICIA {stage.name}\n"
        f"  {stage.label}\n"
        f"  script: {rel(stage.script)}\n"
        f"  log: {stage_log_path.relative_to(run_dir).as_posix()}\n"
    )
    Tee(sys.stdout, pipeline_log, quiet=args.quiet).write(header)

    status["stages"][stage.name] = {
        "status": "running",
        "label": stage.label,
        "started_at": start_iso,
        "script": rel(stage.script),
        "expected_outputs": [str(path.relative_to(run_dir)) if path.is_relative_to(run_dir) else str(path) for path in stage.outputs],
    }
    save_json(run_dir / "stage_status.json", status)

    env = build_env(args, v38_out, reanalysis_out, stage)
    cmd = [sys.executable, str(stage.script)]
    with stage_log_path.open("w", encoding="utf-8") as stage_log:
        stage_log.write(header)
        stage_log.write(f"command: {' '.join(cmd)}\n")
        stage_log.write(f"cwd: {stage.cwd}\n")
        if stage.env:
            stage_log.write(f"stage_env: {json.dumps(stage.env, ensure_ascii=False, sort_keys=True)}\n")
        stage_log.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(stage.cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                stage_log.write(line)
                stage_log.flush()
                pipeline_log.write(line)
                pipeline_log.flush()
                if args.verbose and not args.quiet:
                    sys.stdout.write(line)
                    sys.stdout.flush()
            return_code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                return_code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                return_code = proc.wait()
            elapsed = time.time() - start
            status["stages"][stage.name].update(
                {
                    "status": "interrupted",
                    "finished_at": timestamp(),
                    "duration_seconds": elapsed,
                    "return_code": return_code,
                }
            )
            save_json(run_dir / "stage_status.json", status)
            interrupted_msg = (
                f"[{timestamp()}] INTERRUMPIDO {stage.name} en {duration_text(elapsed)}. "
                f"Puede reanudar con --resume --output {run_dir}\n"
            )
            stage_log.write(interrupted_msg)
            pipeline_log.write(interrupted_msg)
            pipeline_log.flush()
            if not args.quiet:
                sys.stdout.write(interrupted_msg)
                sys.stdout.flush()
            raise

    elapsed = time.time() - start
    missing_outputs = [path for path in stage.outputs if not path.exists()]
    if return_code != 0 or missing_outputs:
        status["stages"][stage.name].update(
            {
                "status": "failed",
                "finished_at": timestamp(),
                "duration_seconds": elapsed,
                "return_code": return_code,
                "missing_outputs": [str(path) for path in missing_outputs],
            }
        )
        save_json(run_dir / "stage_status.json", status)
        detail = f"return_code={return_code}"
        if missing_outputs:
            detail += f"; missing_outputs={len(missing_outputs)}"
        raise RuntimeError(f"Falló la etapa {stage.name}: {detail}. Ver {stage_log_path}")

    status["stages"][stage.name].update(
        {
            "status": "completed",
            "finished_at": timestamp(),
            "duration_seconds": elapsed,
            "return_code": return_code,
            "outputs": [str(path.relative_to(run_dir)) if path.is_relative_to(run_dir) else str(path) for path in stage.outputs],
        }
    )
    save_json(run_dir / "stage_status.json", status)
    footer = f"[{timestamp()}] TERMINA {stage.name} en {duration_text(elapsed)}\n"
    Tee(sys.stdout, pipeline_log, quiet=args.quiet).write(footer)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def package_known_materials(run_dir: Path, reanalysis_out: Path) -> dict:
    materials_dir = run_dir / "supplementary_materials"
    figures_dir = run_dir / "paper_figures"
    copied: list[dict] = []
    missing: list[str] = []

    candidates = [
        (ROOT / "Supplementary_Appendix_A.xlsx", materials_dir / "Supplementary_Appendix_A.xlsx"),
        (ROOT / "Supplementary_Appendix_A_prime.xlsx", materials_dir / "Supplementary_Appendix_A_prime.xlsx"),
        (ROOT / "Supplementary_Appendix_B.xlsx", materials_dir / "Supplementary_Appendix_B.xlsx"),
        (ROOT / "Supplementary_Appendix_C.pdf", materials_dir / "Supplementary_Appendix_C.pdf"),
        (ROOT / "Apendice-Diccionario-Variables.md", materials_dir / "Apendice-Diccionario-Variables.md"),
        (CODE_DIR / "v3.6" / "data" / "synthetic_example" / "synthetic_dataset_final_3_x.csv", materials_dir / "synthetic_dataset_final_3_x.csv"),
        (CODE_DIR / "v3.6" / "data" / "synthetic_example" / "README.md", materials_dir / "synthetic_example_README.md"),
        (V36_REANALYSIS_OUT / "08_final_learning_curve.png", figures_dir / "Figure05_candidate_08_final_learning_curve.png"),
        (V36_REANALYSIS_OUT / "06_pca_scatter_CLEAN.png", figures_dir / "Figure08_candidate_06_pca_scatter_CLEAN.png"),
        (V36_REANALYSIS_OUT / "09_shap_summary_missing_assignments.png", figures_dir / "Figure06_candidate_09_shap_summary_missing_assignments.png"),
    ]

    for src, dst in candidates:
        if copy_if_exists(src, dst):
            copied.append({"source": rel(src), "dest": dst.relative_to(run_dir).as_posix()})
        else:
            missing.append(rel(src))

    for path in sorted(reanalysis_out.glob("05_shap_summary_*_cutoff*.png")):
        dst = figures_dir / "v3.8_shap_by_cutoff" / path.name
        if copy_if_exists(path, dst):
            copied.append({"source": path.relative_to(run_dir).as_posix(), "dest": dst.relative_to(run_dir).as_posix()})

    return {"copied": copied, "missing": missing}


def write_final_summary(run_dir: Path, args: argparse.Namespace, status: dict, material_report: dict) -> None:
    generated = collect_generated_files(run_dir)
    save_json(run_dir / "generated_files.json", {"files": generated})

    lines = [
        "# Pipeline completo (100% + v3.8) — resumen final",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Finalizado: {timestamp()}",
        f"- Smoke test: `{args.smoke_test}`",
        f"- TensorFlow CPU only: `{args.tf_cpu}`",
        f"- Semilla base: `{SEED}`",
        "",
        "## Etapas",
    ]
    for name, info in status.get("stages", {}).items():
        duration = duration_text(info.get("duration_seconds", 0)) if "duration_seconds" in info else "n/a"
        lines.append(f"- `{name}`: {info.get('status')} ({duration})")

    lines.extend(["", "## Salidas principales"])
    for key in ("v3.8_out", "v3.8_reanalysis_out", "logs", "supplementary_materials", "paper_figures"):
        path = run_dir / key
        if path.exists():
            lines.append(f"- `{key}/`")

    lines.extend(["", "## Materiales empaquetados"])
    for item in material_report["copied"]:
        lines.append(f"- `{item['dest']}`")
    if material_report["missing"]:
        lines.extend(["", "## Materiales no encontrados"])
        for item in material_report["missing"]:
            lines.append(f"- `{item}`")

    lines.extend(["", f"Total de archivos generados o empaquetados: `{len(generated)}`"])
    (run_dir / "summary").mkdir(exist_ok=True)
    (run_dir / "summary" / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = output_dir_from_args(args)
    v38_out = run_dir / "v3.8_out"
    reanalysis_out = run_dir / "v3.8_reanalysis_out"
    log_dir = run_dir / "logs"
    stages = make_stages(v38_out, reanalysis_out, args.smoke_test)

    if args.list_stages:
        for index, stage in enumerate(stages, start=1):
            print(f"{index:02d}. {stage.name}: {stage.label}")
        return 0

    stage_names = [stage.name for stage in stages]
    if args.start_from and args.start_from not in stage_names:
        print(f"Etapa desconocida: {args.start_from}", file=sys.stderr)
        print("Etapas validas:", ", ".join(stage_names), file=sys.stderr)
        return 2
    if args.resume and not run_dir.exists():
        print(f"No se puede reanudar: no existe {run_dir}", file=sys.stderr)
        return 2
    if run_dir.exists() and any(run_dir.iterdir()) and not (args.resume or args.force):
        print(f"La carpeta de salida ya existe y no esta vacia: {run_dir}", file=sys.stderr)
        print("Usa --resume, --force o elige otra carpeta con --output.", file=sys.stderr)
        return 2
    if args.force and run_dir.exists() and not args.resume:
        print(f"--force sobre carpeta existente: se reutilizara {run_dir} y se regeneraran etapas.")

    for path in (run_dir, v38_out, reanalysis_out, log_dir, run_dir / "summary"):
        path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_dir": str(run_dir),
        "created_at": timestamp(),
        "root": str(ROOT),
        "seed": SEED,
        "smoke_test": args.smoke_test,
        "tf_cpu": args.tf_cpu,
        "python": sys.executable,
        "v38_out": str(v38_out),
        "v38_reanalysis_out": str(reanalysis_out),
        "stages": [
            {
                "name": stage.name,
                "label": stage.label,
                "script": rel(stage.script),
                "long_running": stage.long_running,
                "env": stage.env,
                "expected_outputs": [str(path) for path in stage.outputs],
            }
            for stage in stages
        ],
    }
    save_json(run_dir / "run_manifest.json", manifest)
    status = load_status(run_dir / "stage_status.json")

    start_index = stage_names.index(args.start_from) if args.start_from else 0
    selected_stages = stages[start_index:]
    total = len(stages)
    pipeline_start = time.time()

    with (log_dir / "pipeline.log").open("a", encoding="utf-8") as pipeline_log:
        Tee(sys.stdout, pipeline_log, quiet=args.quiet).write(
            f"[{timestamp()}] Pipeline completo (100% + v3.8) iniciado\n"
            f"  run_dir: {run_dir}\n"
            f"  smoke_test: {args.smoke_test}\n"
            f"  tf_cpu: {args.tf_cpu}\n"
            f"  stages: {len(selected_stages)} de {total}\n"
        )
        try:
            for stage in selected_stages:
                stage_index = stages.index(stage) + 1
                run_stage(
                    args,
                    stage,
                    stage_index,
                    total,
                    run_dir,
                    log_dir,
                    v38_out,
                    reanalysis_out,
                    status,
                    pipeline_log,
                )

            material_report = package_known_materials(run_dir, reanalysis_out)
            save_json(run_dir / "packaged_materials.json", material_report)
            write_final_summary(run_dir, args, status, material_report)

            elapsed = time.time() - pipeline_start
            Tee(sys.stdout, pipeline_log, quiet=args.quiet).write(
                f"\n[{timestamp()}] Pipeline completo (100% + v3.8) terminado correctamente en {duration_text(elapsed)}\n"
                f"Resumen: {run_dir / 'summary' / 'final_summary.md'}\n"
            )
            return 0
        except Exception as exc:
            elapsed = time.time() - pipeline_start
            Tee(sys.stdout, pipeline_log, quiet=False).write(
                f"\n[{timestamp()}] ERROR: pipeline detenido despues de {duration_text(elapsed)}\n"
                f"{exc}\n"
                f"Log completo: {log_dir / 'pipeline.log'}\n"
            )
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
