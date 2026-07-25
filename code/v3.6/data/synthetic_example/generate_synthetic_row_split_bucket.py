"""
Added for this public code repository (not part of the original private
working copy): generates a synthetic `row_split_bucket.csv` so that
`code/03-model/v3.6-reanalysis/03_build_clean_dataset.py` can run end to end
on the synthetic example dataset without the real, dated raw-source files
that `02_build_course_year.py` normally uses (those files contain real
calendar dates tied to real, pseudonymized institutional records and are
not included in this repository).

This script assigns each synthetic course a synthetic year (2020-2024) and
the same train/holdout split rule used on the real data (courses starting
on or before 2023 go to `train_le2023`, courses starting in 2024 go to
`holdout_2024`), producing a file with the same schema that
`03_build_clean_dataset.py` expects: `uid_hash, course_hash, min_date, year,
split_bucket`.

Run this after `generate_synthetic_dataset.py` and before
`run_pipeline.py`.
"""
import os

import numpy as np
import pandas as pd

SEED = 43  # different from generate_synthetic_dataset.py's SEED on purpose
rng = np.random.default_rng(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
SYNTHETIC_CSV = os.path.join(HERE, "synthetic_dataset_final_3_x.csv")
OUT_PATH = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "03-model", "v3.6-reanalysis", "out", "row_split_bucket.csv")
)

df = pd.read_csv(SYNTHETIC_CSV, usecols=["uid_hash", "course_hash"])
courses = sorted(df["course_hash"].unique())
n_courses = len(courses)

# Mirror the real dataset's proportions loosely: most courses start on or
# before 2023, a minority start in 2024 (temporal holdout).
years = rng.choice([2020, 2021, 2022, 2023, 2024], size=n_courses, p=[0.20, 0.25, 0.25, 0.20, 0.10])
course_year = dict(zip(courses, years))
course_min_date = {c: f"{y}-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 28)):02d}" for c, y in course_year.items()}

df["year"] = df["course_hash"].map(course_year)
df["min_date"] = df["course_hash"].map(course_min_date)
df["split_bucket"] = np.where(df["year"] <= 2023, "train_le2023", "holdout_2024")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df[["uid_hash", "course_hash", "min_date", "year", "split_bucket"]].to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH} ({df.shape})")
print(df["split_bucket"].value_counts())
