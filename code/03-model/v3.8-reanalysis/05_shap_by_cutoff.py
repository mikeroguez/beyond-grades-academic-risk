"""
v3.8 reanálisis — Etapa 3: SHAP sobre el modelo DNN tuneado de UN corte a
la vez (25/50/75%), misma metodología que
`code/03-model/v3.6-reanalysis/09_shap_analysis.py`: KernelExplainer,
background = KMeans (80 clusters) ajustado SOLO con el pool de
entrenamiento ≤2023 de ese corte, explicando el holdout temporal 2024 de
ese mismo corte (nunca visto por el modelo final).

Uso:
    CUTOFF=50 python 05_shap_by_cutoff.py
    CUTOFF=75 python 05_shap_by_cutoff.py
    CUTOFF=25 python 05_shap_by_cutoff.py

Interés particular de Miguel: qué features dominan `missing_assignments`
en cada corte, ahora que por fin tiene señal (Etapa 2) -- si son las de
recencia/ritmo (`days_since_last_engagement`, `submission_pace_per_week`,
`exam_attempts_pace_per_week`) o las de posición relativa
(`relative_position_*`), eso cerraría la narrativa completa de WP1 para
ese target.
"""
import json
import os

if os.environ.get("V38_TF_CPU_ONLY") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from tensorflow import keras
import tensorflow as tf

if os.environ.get("V38_TF_CPU_ONLY") == "1":
    try:
        tf.config.set_visible_devices([], "GPU")
        print("[TensorFlow] GPU deshabilitada por V38_TF_CPU_ONLY=1")
    except RuntimeError as exc:
        print(f"[TensorFlow] No se pudo deshabilitar GPU despues de inicializar runtime: {exc}")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("V38_REANALYSIS_OUT_DIR", os.path.join(HERE, "out"))
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42

CUTOFF = os.environ.get("CUTOFF")
if CUTOFF not in ("25", "50", "75"):
    raise ValueError(f"Define CUTOFF=25|50|75 (recibido: {CUTOFF!r})")

K_BG = int(os.environ.get("SHAP_K_BG", 80))
N_SAMPLE_MAX = int(os.environ.get("SHAP_N_SAMPLE_MAX", 800))
MODEL_SUFFIX = os.environ.get("SHAP_MODEL_SUFFIX", "")

TARGETS = ["avg_assignment_score", "missing_assignments", "assignment_procrast_rate", "avg_exam_score", "exam_accuracy"]

df = pd.read_csv(os.path.join(OUT_DIR, f"cutoff_{CUTOFF}_clean_dataset.csv"))
with open(os.path.join(OUT_DIR, "03_selected_features_by_cutoff.json")) as f:
    selected_by_cutoff = json.load(f)
CLEAN_FEATURES = selected_by_cutoff[CUTOFF]

train_pool = df[df["split_bucket"] == "train_le2023"].reset_index(drop=True)
holdout_2024 = df[df["split_bucket"] == "holdout_2024"].reset_index(drop=True)

N_SAMPLE = min(N_SAMPLE_MAX, len(holdout_2024))
print(f"[corte {CUTOFF}%] Explicando {N_SAMPLE} filas del holdout 2024 (de {len(holdout_2024)} disponibles), "
      f"{len(CLEAN_FEATURES)} features")

model_path = os.path.join(OUT_DIR, f"04_final_tuned_model_cutoff{CUTOFF}{MODEL_SUFFIX}.keras")
imputer_path = os.path.join(OUT_DIR, f"04_final_imputer_cutoff{CUTOFF}{MODEL_SUFFIX}.joblib")
scaler_path = os.path.join(OUT_DIR, f"04_final_scaler_cutoff{CUTOFF}{MODEL_SUFFIX}.joblib")
for p in (model_path, imputer_path, scaler_path):
    if not os.path.exists(p):
        raise FileNotFoundError(f"No existe {p} -- corre primero 04_dnn_hp_search_by_cutoff.py con CUTOFF={CUTOFF}.")

model = keras.models.load_model(model_path)
imputer = joblib.load(imputer_path)
scaler = joblib.load(scaler_path)

# Mismo recorte post-escalado que el bug fix de 04_dnn_hp_search_by_cutoff.py
# (el modelo se entrenó con esto -- hay que explicarlo con el mismo
# preprocesamiento que vio en entrenamiento, si no las predicciones del
# submodelo no coincidirían con las reales).
CLIP_ABS = float(os.environ.get("DNN_CLIP_ABS", 10.0))


def transform(X_raw):
    X = scaler.transform(imputer.transform(X_raw[CLEAN_FEATURES]))
    return np.clip(X, -CLIP_ABS, CLIP_ABS)


X_train_scaled = transform(train_pool)
X_holdout_scaled = transform(holdout_2024)

if N_SAMPLE < len(holdout_2024):
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(holdout_2024), size=N_SAMPLE, replace=False)
else:
    sample_idx = np.arange(len(holdout_2024))
X_explain = X_holdout_scaled[sample_idx]

print(f"[corte {CUTOFF}%] Background K-Means: resumiendo {X_train_scaled.shape[0]} filas de train en {K_BG} clusters...")
km = KMeans(n_clusters=min(K_BG, X_train_scaled.shape[0]), n_init=10, random_state=SEED).fit(X_train_scaled)
bg_summary = km.cluster_centers_

shap_results = {}
feature_names = CLEAN_FEATURES

for target_name in TARGETS:
    print(f"\n[corte {CUTOFF}%] === SHAP para target: {target_name} ===")
    submodel = keras.Model(inputs=model.input, outputs=model.get_layer(target_name).output)

    def f_t(z, _submodel=submodel):
        return _submodel.predict(z, verbose=0).ravel()

    explainer = shap.KernelExplainer(f_t, bg_summary)
    shap_values = explainer.shap_values(X_explain, silent=True)
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    np.save(os.path.join(OUT_DIR, f"05_shap_values_{target_name}_cutoff{CUTOFF}{MODEL_SUFFIX}.npy"), shap_values)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
    shap_results[target_name] = ranking.to_dict()

    print(f"[corte {CUTOFF}%] Top 10 features por |SHAP| medio ({target_name}):")
    print(ranking.head(10).to_string())

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False)
        plt.title(f"SHAP summary — {target_name} (corte {CUTOFF}%, holdout 2024, n={N_SAMPLE})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"05_shap_summary_{target_name}_cutoff{CUTOFF}{MODEL_SUFFIX}.png"), dpi=150)
        plt.close()
        print(f"[corte {CUTOFF}%] Guardado plot: 05_shap_summary_{target_name}_cutoff{CUTOFF}{MODEL_SUFFIX}.png")
    except Exception as e:
        print(f"[corte {CUTOFF}%][SHAP plot] omitido para {target_name}: {e}")

    # Guardado incremental por target (no solo al final del corte), para
    # que si el proceso se corta a medio camino, lo ya completado quede en disco.
    with open(os.path.join(OUT_DIR, f"05_shap_summary_cutoff{CUTOFF}{MODEL_SUFFIX}.json"), "w") as f:
        json.dump(
            {
                "cutoff": CUTOFF,
                "n_sample_explained": int(N_SAMPLE),
                "n_holdout_available": int(len(holdout_2024)),
                "k_background_clusters": K_BG,
                "features": CLEAN_FEATURES,
                "per_target_mean_abs_shap": shap_results,
            },
            f,
            indent=2,
        )

print(f"\n[corte {CUTOFF}%] Guardado: out/05_shap_summary_cutoff{CUTOFF}{MODEL_SUFFIX}.json")
