# Beyond grades: multi-target deep learning for early academic risk detection — code

This repository contains the analysis code for the manuscript **"Beyond grades: multi-target
deep learning for early academic risk detection"** (Rodríguez-Ortiz, Anido-Rifón,
Santana-Mancilla), submitted to *Applied Sciences* (MDPI).

It is released as the code-availability companion to the paper, in response to a
reviewer request for reproducibility. Because the underlying institutional data
cannot be shared publicly (see [Data availability](#data-availability) below), this
repository ships a **synthetic example dataset** with the same schema as the real
one, so the full pipeline can be run end to end by anyone, without access to real
student records.

## What this reproduces

Running this code on the synthetic dataset reproduces the *pipeline structure and
every reported computation* — feature engineering, the leakage audit, the
leakage-free 15-feature benchmark (Tables 5–7), the early temporal-cutoff analysis
(Table 9), the multi-task ablation (Section 6.3), leave-one-course-out and 2024
holdout robustness checks (Section 6.4), the behavioral clustering (Table 10,
Figure 8), and SHAP interpretability (Figure 6, Supplementary Appendix C) — but the
**numeric results obtained on the synthetic data will not match the numbers reported
in the paper**, which were computed on the real, pseudonymized institutional dataset
described in the manuscript. The synthetic data exists to demonstrate that the code
runs correctly end to end and to make the pipeline's logic auditable, not to
reproduce the paper's exact figures.

## Repository layout

```
code/
  v3.6/
    data/synthetic_example/ synthetic stand-in dataset + generators (see below)
  v3.8/                     recomputes features at 25/50/75% of course duration
                            for the early-prediction analysis (Section 5.3)
  03-model/
    shared_clean_features.py        single source of truth for the leakage-free
                                     feature list used across all model scripts
    v3.6-reanalysis/                full-course (100%) benchmark: baselines, DNN
                                     hyperparameter search, clustering, SHAP,
                                     leave-one-course-out, avg_exam_score sensitivity
    v3.8-reanalysis/                early-cutoff benchmark (25/50/75%): baselines,
                                     DNN search, SHAP, cumulative rule baselines
run_pipeline.py                     orchestrates the model-stage scripts end to end,
                                     with resume/skip support and isolated per-run logs
requirements.txt
LICENSE
```

This repository covers the **analysis stage only**: everything needed to go from an
analysis-ready, student-course-level dataset (real or, here, synthetic) to every
reported table and figure. It does not include the upstream pipeline that builds
that dataset from raw institutional exports (pseudonymization, per-source
cleaning, merging, feature selection): those stages consume raw, dated
institutional records that will never be shared, so including that code here would
not add anything an external reader could actually run — only the analysis stages
are runnable with the provided synthetic dataset, orchestrated by
`run_pipeline.py`.

## Running the pipeline on the synthetic dataset

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Put the synthetic dataset where the model-stage scripts expect the analysis-ready
# dataset to live (this mirrors where the real, pseudonymized dataset sits in the
# authors' working environment):
mkdir -p code/v3.6/data/Material
cp code/v3.6/data/synthetic_example/synthetic_dataset_final_3_x.csv \
   code/v3.6/data/Material/dataset_final_3_x.csv

# Generate the synthetic train/holdout time bucket (stands in for the real,
# dated raw-source files that 02_build_course_year.py normally reads, which
# are not included in this repository — see the script's docstring):
python code/v3.6/data/synthetic_example/generate_synthetic_row_split_bucket.py

# List available stages
python run_pipeline.py --list-stages

# Run everything (the DNN hyperparameter-search stages are the slow part: by
# design they always run the full 40-trial Bayesian search regardless of
# --smoke-test, since that search is the one thing the original authors
# never wanted to fake even for a quick check — see the comment above
# `main_stages` in run_pipeline.py. Budget real time for this: the search
# runs 4 times in total (once for the full-course benchmark, once per
# 25/50/75% cutoff). On the 1,016-row synthetic dataset with CPU-only
# TensorFlow (--tf-cpu) this is faster per trial than on the real
# 4,230-row dataset, but still on the order of hours, not minutes.)
python run_pipeline.py --output pipeline_runs/example_run --tf-cpu
```

Each stage's expected output files are declared in `run_pipeline.py`; the runner
skips a stage if its declared outputs already exist, so an interrupted run can be
resumed with `--resume`.

## Validation status

This repository's early, fast stages (`main_clean_dataset` through
`main_feature_selection`) were run end to end against the synthetic dataset
and confirmed to produce the expected schema (15 leakage-free features, 11
excluded, matching Appendix A). The DNN hyperparameter-search stages were
*not* run to completion as part of preparing this repository, because — as
noted above — they always run the full 40-trial search regardless of
`--smoke-test`, which was not a worthwhile use of time solely to verify that
already-working TensorFlow/Keras Tuner code (unchanged from the authors'
private working copy) still imports and executes correctly. If you run the
full pipeline and hit an issue in a later stage, please open an issue.

## Data availability

The data supporting the results reported in the paper were obtained from
Universidad de Colima institutional platforms under permit 1I.1.2/205000/611/2025.
Raw and pseudonymized data cannot be shared publicly due to privacy and
institutional policy; a pseudonymized version may be available from the
corresponding author upon reasonable request, subject to institutional
authorization. See `code/v3.6/data/synthetic_example/README.md` for details on the
synthetic substitute dataset provided in this repository.

Supplementary materials (variable dictionary, literature benchmark table, extended
clustering/SHAP analyses) are archived on Zenodo:
<https://doi.org/10.5281/zenodo.20060485>.

## What is intentionally not in this repository

- Any real or pseudonymized student/course data, trained model weights, or
  data-derived intermediate files (`out/` directories in the authors' working
  copy) — everything here is code, plus the fully synthetic example dataset.
- The upstream raw-data pipeline (pseudonymization of raw institutional
  exports, per-source cleaning, merging, feature selection): it operates on
  real, dated raw records that are never shared, so it is not runnable by an
  external reader regardless of whether it is included, and omitting it keeps
  this repository focused on the part that is actually reproducible.
- Exploratory analyses that did not lead to a reported result (e.g., an earlier
  full-course access/forum feature exploration superseded by the early-cutoff
  analysis in Section 5.3).
- Superseded, pre-audit versions of the modeling notebooks that predate the
  leakage audit described in the manuscript.

## Notes on code style

Code comments and some variable names are primarily in Spanish, the lead author's
working language during analysis; docstrings explain the methodological reasoning
behind each script in detail. English translations of specific files can be
requested via the corresponding author.

## License

Code is released under the MIT License (see `LICENSE`). The synthetic dataset is
released under the same license; it contains no real personal data.

## Citation

If you use this code, please cite the manuscript (citation details to be added
once the DOI is assigned) and, if relevant, this repository.
