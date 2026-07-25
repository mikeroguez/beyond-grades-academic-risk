"""
Dataset sintético con el mismo esquema que `dataset_final_3_x.csv`
(2 identificadores + 5 targets + 26 features, mismos nombres de columna),
para el Data Availability Statement (Reviewer 1 #17, "impossible to
reproduce" -- pedido de Miguel vía `code/v3.8/my-review-v3.8-1st.txt` #4).

**Genuinamente sintético**: NINGÚN valor de este archivo proviene de
remuestrear, perturbar o derivar estadísticos de los datos reales de
estudiantes. Todo se genera desde distribuciones marginales simples
(Poisson, Binomial, Beta, Normal) con parámetros elegidos a mano para que
el rango/tipo de cada columna sea plausible (aproximado, no ajustado a los
datos reales) -- el único objetivo es que el código del pipeline
(`code/v3.6/03-feature-selection/`, `code/03-model/v3.6-reanalysis/`)
corra de punta a punta sobre un archivo con la forma correcta, para que
"impossible to reproduce" quede resuelto sin comprometer la privacidad de
ningún estudiante real.

Excepción deliberada: las identidades algebraicas exactas ya documentadas
en `AUDITORIA-LEAKAGE.md` (`missing_assignments = total_assignments -
submitted_assignments`, `assignment_procrast_rate = late_assignments /
submitted_assignments`, `exam_accuracy = exam_correct_answers /
exam_questions`) SÍ se preservan aquí a propósito -- son relaciones
puramente aritméticas entre columnas, no información de estudiantes
reales, y preservarlas permite que cualquiera pueda verificar la fuga
descrita en la auditoría directamente sobre este archivo sintético, sin
necesitar los datos reales.

IDs con prefijo "syn_" para que sea imposible confundirlos con
seudónimos reales (`stu_NNNNNN`/`crs_NNNNN`).
"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SEED = 42
rng = np.random.default_rng(SEED)

N_COURSES = 40
STUDENTS_PER_COURSE_RANGE = (10, 40)  # aprox. al rango real (mediana ~26)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------
# 1) Estructura estudiante-curso (IDs sintéticos, nunca reales)
# ---------------------------------------------------------------------
rows = []
student_counter = 0
for c in range(N_COURSES):
    course_hash = f"syn_crs_{c:05d}"
    n_students = int(rng.integers(*STUDENTS_PER_COURSE_RANGE))
    for _ in range(n_students):
        uid_hash = f"syn_stu_{student_counter:06d}"
        student_counter += 1
        rows.append({"uid_hash": uid_hash, "course_hash": course_hash})

n = len(rows)
df = pd.DataFrame(rows)
print(f"Filas sintéticas generadas: {n} ({N_COURSES} cursos sintéticos)")

# ---------------------------------------------------------------------
# 2) Conteos crudos de tareas -- preservan las identidades algebraicas
#    documentadas en AUDITORIA-LEAKAGE.md (a propósito)
# ---------------------------------------------------------------------
total_assignments_raw = rng.poisson(15, size=n) + 1
submit_p = rng.uniform(0.5, 0.95, size=n)
submitted_assignments_raw = rng.binomial(total_assignments_raw, submit_p)
missing_assignments_raw = total_assignments_raw - submitted_assignments_raw  # identidad exacta

graded_p = rng.uniform(0.7, 1.0, size=n)
graded_assignments_raw = rng.binomial(submitted_assignments_raw, graded_p)
ungraded_assignments_raw = total_assignments_raw - graded_assignments_raw

late_p = rng.uniform(0.05, 0.6, size=n)
late_assignments_raw = rng.binomial(submitted_assignments_raw, late_p)
with np.errstate(divide="ignore", invalid="ignore"):
    assignment_procrast_rate = np.where(
        submitted_assignments_raw > 0, late_assignments_raw / np.maximum(submitted_assignments_raw, 1), 0.0
    )  # identidad exacta: late / submitted

assignment_submit_rate = submitted_assignments_raw / total_assignments_raw  # identidad exacta: submitted/total

# ---------------------------------------------------------------------
# 3) Conteos crudos de exámenes -- misma lógica
# ---------------------------------------------------------------------
exam_questions_raw = rng.poisson(40, size=n) + 5
correct_p = rng.uniform(0.3, 0.95, size=n)
exam_correct_answers_raw = rng.binomial(exam_questions_raw, correct_p)
exam_accuracy = exam_correct_answers_raw / exam_questions_raw  # identidad exacta: correct/questions

total_exams_raw = rng.integers(0, 6, size=n)
completed_exams_raw = np.minimum(total_exams_raw, rng.binomial(np.maximum(total_exams_raw, 1), 0.8))
incomplete_exams_raw = total_exams_raw - completed_exams_raw
exam_submit_rate = np.where(total_exams_raw > 0, completed_exams_raw / np.maximum(total_exams_raw, 1), 0.0)
perfect_exams_raw = rng.binomial(np.maximum(completed_exams_raw, 0), 0.05)
outlier_exams_raw = rng.binomial(np.maximum(total_exams_raw, 0), 0.03)
exam_incidents_raw = rng.poisson(0.3, size=n)
avg_exam_incidents = exam_incidents_raw / np.maximum(total_exams_raw, 1)

# ---------------------------------------------------------------------
# 4) Escalado Min-Max de las columnas tipo conteo (mismo método que
#    02_escala_y_renombra.py -- ajustado sobre esta MISMA muestra
#    sintética, no sobre datos reales)
# ---------------------------------------------------------------------
def minmax(x):
    return MinMaxScaler().fit_transform(np.asarray(x, dtype=float).reshape(-1, 1)).flatten()


df["total_assignments"] = minmax(total_assignments_raw)
df["submitted_assignments"] = minmax(submitted_assignments_raw)
df["graded_assignments"] = minmax(graded_assignments_raw)
df["ungraded_assignments"] = minmax(ungraded_assignments_raw)
df["late_assignments"] = minmax(late_assignments_raw)
df["missing_assignments"] = minmax(missing_assignments_raw)  # target
df["assignment_submit_rate"] = assignment_submit_rate  # ya en [0,1], sin escalar (método "None" original)
df["assignment_procrast_rate"] = assignment_procrast_rate  # target, ya en [0,1]

df["exam_questions"] = minmax(exam_questions_raw)
df["exam_correct_answers"] = minmax(exam_correct_answers_raw)
df["exam_accuracy"] = exam_accuracy  # target, ya en [0,1]
df["total_exams"] = minmax(total_exams_raw)
df["completed_exams"] = minmax(completed_exams_raw)
df["incomplete_exams"] = minmax(incomplete_exams_raw)
df["exam_submit_rate"] = exam_submit_rate  # ya en [0,1]
df["perfect_exams"] = minmax(perfect_exams_raw)
df["outlier_exams_course"] = minmax(outlier_exams_raw)
df["exam_incidents"] = minmax(exam_incidents_raw)
df["avg_exam_incidents"] = minmax(avg_exam_incidents)

# ---------------------------------------------------------------------
# 5) Columnas de puntaje (Min-Max [0,1], min<=max por construcción) y
#    targets de desempeño -- distribuciones Beta aproximadas, SIN
#    relación con datos reales
# ---------------------------------------------------------------------
a_hi, a_lo = rng.beta(5, 2, size=n), rng.beta(2, 5, size=n)
df["max_assignment_score"] = np.maximum(a_hi, a_lo)
df["min_assignment_score"] = np.minimum(a_hi, a_lo)
df["assignment_score_var"] = rng.beta(1.5, 8, size=n)  # sesgada hacia 0, como una varianza típica
df["avg_assignment_score"] = np.clip((df["max_assignment_score"] + df["min_assignment_score"]) / 2
                                      + rng.normal(0, 0.05, size=n), 0, 1)  # target

e_hi, e_lo = rng.beta(4, 2, size=n), rng.beta(2, 4, size=n)
df["max_exam_score"] = np.maximum(e_hi, e_lo)
df["min_exam_score"] = np.minimum(e_hi, e_lo)
df["exam_score_var"] = rng.beta(1.5, 8, size=n)
df["avg_exam_score"] = np.clip((df["max_exam_score"] + df["min_exam_score"]) / 2
                                + rng.normal(0, 0.05, size=n), 0, 1)  # target

# ---------------------------------------------------------------------
# 6) Columnas con RobustScaler en el pipeline original (centradas en 0,
#    ambos signos) -- Normal aproximada, SIN relación con datos reales
# ---------------------------------------------------------------------
df["avg_assignment_delay"] = rng.normal(0, 3, size=n)
df["min_exam_time"] = rng.normal(0, 2, size=n)

# ---------------------------------------------------------------------
# 7) Banderas one-hot de modalidad de examen (mutuamente casi excluyentes,
#    aproximación simple del one-hot real de 02_escala_y_renombra.py)
# ---------------------------------------------------------------------
modality = rng.choice(["all_exams", "five_or_less", "neither"], size=n, p=[0.75, 0.10, 0.15])
df["all_exams"] = (modality == "all_exams").astype(int)
df["five%_or_less_incomplete_exams"] = (modality == "five_or_less").astype(int)

# ---------------------------------------------------------------------
# Orden de columnas idéntico al de dataset_final_3_x.csv real
# ---------------------------------------------------------------------
COLUMN_ORDER = [
    "uid_hash", "course_hash",
    "avg_assignment_score", "missing_assignments", "completed_exams", "submitted_assignments",
    "exam_incidents", "assignment_submit_rate", "max_assignment_score", "avg_exam_incidents",
    "total_exams", "exam_score_var", "all_exams", "exam_questions", "five%_or_less_incomplete_exams",
    "total_assignments", "graded_assignments", "ungraded_assignments", "perfect_exams",
    "min_assignment_score", "min_exam_time", "exam_correct_answers", "assignment_score_var",
    "max_exam_score", "incomplete_exams", "exam_submit_rate", "min_exam_score", "late_assignments",
    "avg_assignment_delay", "outlier_exams_course", "exam_accuracy", "assignment_procrast_rate",
    "avg_exam_score",
]
missing_cols = set(COLUMN_ORDER) - set(df.columns)
if missing_cols:
    raise ValueError(f"Faltan columnas: {missing_cols}")
df = df[COLUMN_ORDER]

assert df.shape[1] == 33, f"Se esperaban 33 columnas, hay {df.shape[1]}"

out_path = os.path.join(HERE, "synthetic_dataset_final_3_x.csv")
df.to_csv(out_path, index=False)
print(f"Guardado: {out_path} ({df.shape})")

# Verificación rápida de las identidades preservadas
check1 = np.allclose(
    MinMaxScaler().fit_transform(missing_assignments_raw.reshape(-1, 1)).flatten(), df["missing_assignments"]
)
print(f"Verificación: missing_assignments = MinMax(total_assignments_raw - submitted_assignments_raw)? {check1}")
