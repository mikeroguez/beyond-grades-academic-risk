"""Single source of truth for the leakage-free predictor set shared by the
100%-course benchmark (`v3.6-reanalysis/`) and the temporal-cutoff pipeline
(`../v3.8/`, `v3.8-reanalysis/`).

Both pipelines must exclude exactly the same predictors from the same 26
engineered features. Before 21 jul 2026 this list was hardcoded
independently in 4 different scripts; that duplication is what let the
min/max/var_exam_score partial leakage (flagged by Miguel,
`code/v3.8/my-review-v3.8-1st.txt` #1) get fixed on the 100%-course side
without anyone updating the 3 copies used by the cutoff pipeline, so the
two would have silently disagreed on what "leakage-free" means for
avg_exam_score. Import from here instead of re-declaring the list.
"""

# Reconstruction of another target's own defining formula (numerator,
# denominator, or exact/near-exact algebraic complement) -- see
# AUDITORIA-LEAKAGE.md Section 3 and 01_leakage_confirmation.py for the
# empirical confirmation of each identity.
LEAKY_PREDICTORS_BLANKET = [
    "total_assignments",        # FD: total_assignments - submitted_assignments = missing_assignments (exacto)
    "submitted_assignments",    # FD/FB: complemento de missing_assignments; denominador de assignment_procrast_rate
    "ungraded_assignments",     # FB: casi duplicado de missing_assignments (r=0.926); total=graded+ungraded (identidad)
    "graded_assignments",       # FB: socio de la identidad afín total=graded+ungraded
    "assignment_submit_rate",   # FB: razón derivada de submitted/total (mismo origen que missing_assignments)
    "late_assignments",         # FB: numerador exacto de assignment_procrast_rate
    "exam_correct_answers",     # FB: numerador exacto de exam_accuracy (accuracy_rate)
    "exam_questions",           # FB: denominador exacto de exam_accuracy (accuracy_rate)
    # FB: casi-duplicados de avg_exam_score (07_avg_exam_score_sensitivity.py,
    # config (a), 20 jul 2026): quitar solo las filas de un intento único
    # apenas mueve el R2 (0.975->0.960) mientras min/max se conserven como
    # predictores -- es decir, min/max determinan avg_exam_score casi por
    # completo en TODO el dataset, no solo en el subconjunto trivial de un
    # intento. Mismo criterio aplicado de forma uniforme a los 5 targets que
    # las 8 exclusiones de arriba, no un caso especial (21 jul 2026).
    "min_exam_score",
    "max_exam_score",
    "exam_score_var",
]

# Resultado de excluir LEAKY_PREDICTORS_BLANKET de las 26 features originales
# (ver 03_build_clean_dataset.py). Se deja también como lista explícita
# porque varios scripts del pipeline de cortes (v3.8) construyen sus propios
# datasets sin pasar por feature_lists.json.
CLEAN_FEATURES = [
    "completed_exams", "exam_incidents", "max_assignment_score", "avg_exam_incidents",
    "total_exams", "all_exams", "five%_or_less_incomplete_exams",
    "perfect_exams", "min_assignment_score", "min_exam_time", "assignment_score_var",
    "incomplete_exams", "exam_submit_rate",
    "avg_assignment_delay", "outlier_exams_course",
]
