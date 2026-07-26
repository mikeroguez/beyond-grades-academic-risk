"""
WP1 — Reviewer 1 #39-41 ("clustering is not proof of robustness") +
R1 #40 (pide explícitamente k, silhouette, Davies-Bouldin, % varianza PCA,
que el paper actual no reporta con números).

`code/03-model/nuanced_profiles.ipynb` (el notebook que produjo la Fig. 8
y el §6.2 del paper, "seven-cluster structures") agrupa usando:
    targets (5) + shap_feats (11):
        ["exam_correct_answers", "exam_questions", "exam_submit_rate",
         "submitted_assignments", "avg_assignment_delay", "incomplete_exams",
         "completed_exams", "late_assignments", "total_assignments",
         "graded_assignments", "assignment_submit_rate"]

De esos 11 "shap_feats", **7 son columnas con fuga confirmada**
(`exam_correct_answers`, `exam_questions`, `submitted_assignments`,
`late_assignments`, `total_assignments`, `graded_assignments`,
`assignment_submit_rate` -- las mismas 8 de AUDITORIA-LEAKAGE.md, menos
`ungraded_assignments`, que no se usaba aquí). El notebook además carga
desde `.../v3.5/dataset_final_3_x.csv` -- el dataset pre-corrección.

Este script rehace TODO el análisis (K-Means sweep k=2-10 con
Elbow/Silhouette/Davies-Bouldin/Calinski-Harabasz, Ward jerárquico,
PCA 2D con % de varianza explicada -- que el notebook original NUNCA
reportó, y que R1 #40 pide explícitamente) en dos versiones EXACTAMENTE
paralelas, mismo pipeline, misma semilla, para que la comparación sea
limpia:

  (A) "OLD" -- misma receta que el notebook original (targets + 11
      shap_feats parcialmente contaminados), sobre
      `code/v3.6/data/Material/dataset_final_3_x.csv` (mismo archivo,
      mismos valores, que consumía `v3.4.ipynb`/`v3.5.ipynb`/el notebook
      de clustering -- ver `code/v3.6/README.md` para la trazabilidad).
  (B) "CLEAN" -- misma receta (targets + top-11 features por importancia
      SHAP combinada), pero usando el SHAP REAL del modelo limpio de WP1
      (`out/09_shap_summary.json`, sin ninguna de las columnas con
      fuga) en vez de las variables "derivadas de tus gráficos" que
      usaba el notebook original.

Ambas comparten metodología (StandardScaler, mismo rango de k, misma
semilla) para que cualquier diferencia en resultados sea atribuible a la
fuga, no a un cambio de procedimiento.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
K_MIN, K_MAX = 2, 10
K_PAPER = 7  # el k que el paper actual reporta (§6.2, Fig. 8)

TARGETS = ["avg_assignment_score", "missing_assignments", "assignment_procrast_rate", "avg_exam_score", "exam_accuracy"]

# ---------------------------------------------------------------------
# (A) OLD: misma receta que nuanced_profiles.ipynb
# ---------------------------------------------------------------------
OLD_SHAP_FEATS = [
    "exam_correct_answers", "exam_questions", "exam_submit_rate",
    "submitted_assignments", "avg_assignment_delay", "incomplete_exams",
    "completed_exams", "late_assignments", "total_assignments",
    "graded_assignments", "assignment_submit_rate",
]
LEAKY_8 = [
    "total_assignments", "submitted_assignments", "ungraded_assignments", "graded_assignments",
    "assignment_submit_rate", "late_assignments", "exam_correct_answers", "exam_questions",
]
old_leaky_in_shap_feats = [c for c in OLD_SHAP_FEATS if c in LEAKY_8]
print(f"De los {len(OLD_SHAP_FEATS)} 'shap_feats' del notebook original, "
      f"{len(old_leaky_in_shap_feats)} son columnas con fuga confirmada: {old_leaky_in_shap_feats}")

OLD_DATA_PATH = os.path.join(HERE, "..", "..", "v3.6", "data", "Material", "dataset_final_3_x.csv")
df_old_full = pd.read_csv(OLD_DATA_PATH)
print(f"\nDataset OLD (dataset_final_3_x.csv, el mismo que consumía v3.4/v3.5/nuanced_profiles): {df_old_full.shape}")

use_cols_old = TARGETS + OLD_SHAP_FEATS
df_old = df_old_full[use_cols_old].copy()
for c in use_cols_old:
    if df_old[c].isna().any():
        df_old[c] = df_old[c].fillna(df_old[c].mean())

# ---------------------------------------------------------------------
# (B) CLEAN: targets + top-11 features por SHAP REAL del modelo limpio
# ---------------------------------------------------------------------
with open(os.path.join(OUT_DIR, "09_shap_summary.json")) as f:
    shap_summary = json.load(f)

agg_shap = {}
for t in TARGETS:
    for feat, val in shap_summary["per_target_mean_abs_shap"][t].items():
        agg_shap[feat] = agg_shap.get(feat, 0.0) + val
ranked_clean = sorted(agg_shap.items(), key=lambda kv: -kv[1])
CLEAN_SHAP_FEATS = [feat for feat, _ in ranked_clean[:11]]
print(f"\nTop-11 features por |SHAP| combinado del modelo LIMPIO (sin fuga), usadas para el clustering CLEAN:")
for feat, val in ranked_clean[:11]:
    print(f"  {feat:35s} {val:.4f}")

clean_dataset = pd.read_csv(os.path.join(OUT_DIR, "clean_dataset.csv"))
use_cols_clean = TARGETS + CLEAN_SHAP_FEATS
df_clean = clean_dataset[use_cols_clean].copy()
for c in use_cols_clean:
    if df_clean[c].isna().any():
        df_clean[c] = df_clean[c].fillna(df_clean[c].mean())

print(f"\nDataset CLEAN (clean_dataset.csv, {len(clean_dataset.columns)} columnas totales, WP1 limpio): {df_clean.shape} "
      f"(mismas {len(clean_dataset)} filas que OLD, mismo orden -- ambos derivan de dataset_final_3_x.csv)")

# ---------------------------------------------------------------------
# Verificación: mismas filas en el mismo orden (para poder comparar ARI)
# ---------------------------------------------------------------------
same_rows = (df_old_full["uid_hash"].values == clean_dataset["uid_hash"].values).all() and \
            (df_old_full["course_hash"].values == clean_dataset["course_hash"].values).all()
print(f"\n¿Mismo orden de filas en OLD y CLEAN? {same_rows}")
if not same_rows:
    raise RuntimeError("Los datasets OLD y CLEAN no están alineados fila a fila -- no se puede comparar ARI directamente.")


# ---------------------------------------------------------------------
# Pipeline de clustering (idéntico para OLD y CLEAN)
# ---------------------------------------------------------------------
def run_clustering_pipeline(df_use: pd.DataFrame, label: str):
    scaler = StandardScaler()
    X = scaler.fit_transform(df_use.values)

    # --- K-Means sweep k=2..10 ---
    sweep_rows = []
    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED)
        labels = km.fit_predict(X)
        sweep_rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        })
    sweep_df = pd.DataFrame(sweep_rows)
    k_best_silhouette = int(sweep_df.loc[sweep_df["silhouette"].idxmax(), "k"])
    k_best_db = int(sweep_df.loc[sweep_df["davies_bouldin"].idxmin(), "k"])

    print(f"\n[{label}] K-Means sweep (k={K_MIN}..{K_MAX}):")
    print(sweep_df.to_string(index=False))
    print(f"[{label}] k óptimo por Silhouette (mayor mejor): {k_best_silhouette} "
          f"(silhouette={sweep_df['silhouette'].max():.3f})")
    print(f"[{label}] k óptimo por Davies-Bouldin (menor mejor): {k_best_db} "
          f"(DB={sweep_df['davies_bouldin'].min():.3f})")

    # --- Ward jerárquico, en k_paper (7, el que reporta el paper actual)
    #     Y en el k óptimo por silhouette de esta versión (honesto: puede
    #     no ser 7) ---
    Z = linkage(X, method="ward")

    def ward_at_k(k):
        labels = fcluster(Z, t=k, criterion="maxclust")
        return {
            "k": k,
            "silhouette": float(silhouette_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "labels": labels,
            "sizes": pd.Series(labels).value_counts().sort_index().to_dict(),
        }

    ward_k7 = ward_at_k(K_PAPER)
    ward_k_best = ward_at_k(k_best_silhouette) if k_best_silhouette != K_PAPER else ward_k7

    print(f"\n[{label}] Ward k={K_PAPER} (el que reporta el paper actual): "
          f"silhouette={ward_k7['silhouette']:.3f}, DB={ward_k7['davies_bouldin']:.3f}, "
          f"CH={ward_k7['calinski_harabasz']:.1f}, tamaños={ward_k7['sizes']}")
    if k_best_silhouette != K_PAPER:
        print(f"[{label}] Ward k={k_best_silhouette} (óptimo real por silhouette de ESTA versión): "
              f"silhouette={ward_k_best['silhouette']:.3f}, DB={ward_k_best['davies_bouldin']:.3f}, "
              f"CH={ward_k_best['calinski_harabasz']:.1f}, tamaños={ward_k_best['sizes']}")

    # --- PCA 2D + % varianza explicada (el paper actual NO reporta esto) ---
    pca = PCA(n_components=2, random_state=SEED)
    X2 = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_
    print(f"[{label}] PCA 2D: PC1={var_explained[0] * 100:.1f}% varianza, "
          f"PC2={var_explained[1] * 100:.1f}% varianza, total={var_explained.sum() * 100:.1f}%")

    # --- Perfiles: medias por cluster (Ward k=7, para comparar con el paper) ---
    df_prof = df_use.copy()
    df_prof["cluster"] = ward_k7["labels"]
    cluster_means = df_prof.groupby("cluster").mean().round(3)

    return {
        "sweep": sweep_df,
        "k_best_silhouette": k_best_silhouette,
        "k_best_davies_bouldin": k_best_db,
        "ward_k7": ward_k7,
        "ward_k_best": ward_k_best,
        "pca_variance_explained": var_explained.tolist(),
        "cluster_means_k7": cluster_means,
        "X2": X2,
        "scaler": scaler,
    }


print("\n" + "=" * 90 + "\n(A) CLUSTERING OLD -- targets + 11 features parcialmente contaminadas por fuga\n" + "=" * 90)
result_old = run_clustering_pipeline(df_old, "OLD")

print("\n" + "=" * 90 + "\n(B) CLUSTERING CLEAN -- targets + top-11 features por SHAP real del modelo sin fuga\n" + "=" * 90)
result_clean = run_clustering_pipeline(df_clean, "CLEAN")

# ---------------------------------------------------------------------
# Comparación directa OLD vs CLEAN
# ---------------------------------------------------------------------
print("\n" + "=" * 90 + "\nCOMPARACIÓN DIRECTA OLD vs CLEAN\n" + "=" * 90)

ari_k7 = adjusted_rand_score(result_old["ward_k7"]["labels"], result_clean["ward_k7"]["labels"])
print(f"Adjusted Rand Index (Ward k=7, OLD vs CLEAN): {ari_k7:.3f} "
      f"(1.0 = misma partición exacta; ~0 = tan distinta como asignación aleatoria)")

print(f"\n{'Métrica':<30s}{'OLD':>15s}{'CLEAN':>15s}")
print(f"{'k óptimo (Silhouette)':<30s}{result_old['k_best_silhouette']:>15d}{result_clean['k_best_silhouette']:>15d}")
print(f"{'k óptimo (Davies-Bouldin)':<30s}{result_old['k_best_davies_bouldin']:>15d}{result_clean['k_best_davies_bouldin']:>15d}")
print(f"{'Silhouette @ k=7':<30s}{result_old['ward_k7']['silhouette']:>15.3f}{result_clean['ward_k7']['silhouette']:>15.3f}")
print(f"{'Davies-Bouldin @ k=7':<30s}{result_old['ward_k7']['davies_bouldin']:>15.3f}{result_clean['ward_k7']['davies_bouldin']:>15.3f}")
print(f"{'Calinski-Harabasz @ k=7':<30s}{result_old['ward_k7']['calinski_harabasz']:>15.1f}{result_clean['ward_k7']['calinski_harabasz']:>15.1f}")
print(f"{'PCA var. explicada (PC1+PC2)':<30s}{sum(result_old['pca_variance_explained'])*100:>14.1f}%{sum(result_clean['pca_variance_explained'])*100:>14.1f}%")

# ---------------------------------------------------------------------
# Guardado
# ---------------------------------------------------------------------
def serialize_result(r):
    return {
        "sweep": r["sweep"].to_dict(orient="records"),
        "k_best_silhouette": r["k_best_silhouette"],
        "k_best_davies_bouldin": r["k_best_davies_bouldin"],
        "ward_k7": {k: v for k, v in r["ward_k7"].items() if k != "labels"},
        "ward_k_best": {k: v for k, v in r["ward_k_best"].items() if k != "labels"},
        "pca_variance_explained": r["pca_variance_explained"],
        "cluster_means_k7": r["cluster_means_k7"].reset_index().to_dict(orient="records"),
    }


output = {
    "old_shap_feats": OLD_SHAP_FEATS,
    "old_leaky_features_in_shap_feats": old_leaky_in_shap_feats,
    "clean_shap_feats_top11": CLEAN_SHAP_FEATS,
    "old": serialize_result(result_old),
    "clean": serialize_result(result_clean),
    "ari_ward_k7_old_vs_clean": float(ari_k7),
}
with open(os.path.join(OUT_DIR, "06_clustering_clean_results.json"), "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nGuardado: {os.path.join(OUT_DIR, '06_clustering_clean_results.json')}")

# Guardar también las medias por cluster en CSV, más fáciles de inspeccionar
result_old["cluster_means_k7"].to_csv(os.path.join(OUT_DIR, "06_cluster_means_OLD_k7.csv"))
result_clean["cluster_means_k7"].to_csv(os.path.join(OUT_DIR, "06_cluster_means_CLEAN_k7.csv"))

# PCA scatter plots (opcional, si matplotlib está disponible)
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for label, r in [("OLD", result_old), ("CLEAN", result_clean)]:
        plt.figure(figsize=(7, 6))
        labels_plot = r["ward_k7"]["labels"]
        for lab in np.unique(labels_plot):
            idx = labels_plot == lab
            plt.scatter(r["X2"][idx, 0], r["X2"][idx, 1], s=12, label=f"Cluster {lab}")
        var_exp = r["pca_variance_explained"]
        plt.xlabel(f"PC1 ({var_exp[0] * 100:.1f}% var.)")
        plt.ylabel(f"PC2 ({var_exp[1] * 100:.1f}% var.)")
        suffix = "leakage-free" if label == "CLEAN" else "diagnostic"
        plt.title(f"Ward (k = 7) PCA projection ({suffix})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"06_pca_scatter_{label}.png"), dpi=150)
        plt.close()
    print(f"Guardado: out/06_pca_scatter_OLD.png, out/06_pca_scatter_CLEAN.png")
except Exception as e:
    print(f"[plot] omitido: {e}")
