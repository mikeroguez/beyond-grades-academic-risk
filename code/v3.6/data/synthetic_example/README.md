# Synthetic example dataset (`synthetic_dataset_final_3_x.csv`)

**This file is entirely synthetic.** No value in it comes from a real student —
it was generated with `generate_synthetic_dataset.py` from simple marginal
distributions (Poisson, Binomial, Beta, Normal), without resampling or
perturbing the real study data in any way. It has no relationship whatsoever
to real students, courses, or records at Universidad de Colima.

**Purpose:** to address a reviewer request for reproducibility (pipeline,
seeds, split indices, configuration). The real raw data cannot be shared
publicly (it contains student information), but this file has exactly the
same schema as `dataset_final_3_x.csv` (33 columns: `uid_hash`,
`course_hash`, 5 targets, 26 features — same names, types, and
approximate ranges), so the pipeline code can be run end to end on a public
file without depending on private data.

**Technical note:** the algebraic identities documented in
`AUDITORIA-LEAKAGE.md` (`missing_assignments = MinMax(total_assignments -
submitted_assignments)`, `assignment_procrast_rate = late_assignments /
submitted_assignments`, `exam_accuracy = exam_correct_answers /
exam_questions`) are deliberately preserved in the generation process — these
are purely arithmetic relationships between columns, not information about
students, and keeping them lets anyone verify the data-leakage audit
described in the manuscript directly on this public file, without needing
the real data.

IDs use the `syn_` prefix (`syn_stu_NNNNNN`, `syn_crs_NNNNN`) so they can
never be confused with the real pseudonyms (`stu_NNNNNN`, `crs_NNNNN`) used
elsewhere in the authors' working repository.

1,016 rows, 40 synthetic courses (~10-40 students per course, similar to the
real range). Regenerate with `python generate_synthetic_dataset.py`.
