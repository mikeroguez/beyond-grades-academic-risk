"""
WP1 — Interpretabilidad SHAP sobre el modelo LIMPIO (sin fuga), correctamente
ubicado en el pipeline (a diferencia del SHAP de `v3.4.ipynb`, que se corría
sobre el modelo CON fuga y con preprocesamiento ajustado sobre todo el
dataset). Pedido explícito de Miguel vía coordinador, 18 jul 2026.

Dos errores metodológicos del SHAP original que este script corrige:
  1. `v3.4.ipynb` explica un modelo entrenado con las 26 features originales,
     8 de las cuales son fugas confirmadas (Sección 2 de METODOLOGIA-V3.md)
     -- la Fig. 6 del paper ("SHAP summary plot for missing_assignments...
     total, ungraded, and submitted assignments are key predictors") es en
     realidad una demostración visual de la fuga, no un hallazgo de
     interpretabilidad genuino.
  2. El `RobustScaler` de `v3.4.ipynb` se ajusta sobre `X_train_raw` DESPUÉS
     del split -- eso está bien -- pero como se documentó en la Sección 5 de
     METODOLOGIA-V3.md, los valores que entran a ese split YA vienen
     escalados una vez con estadísticos de TODO el dataset (fuga de
     preprocesamiento río arriba, en `03-feature-selection/02_escala_y_renombra.py`).

Este script usa el modelo final ya entrenado por `08_dnn_hp_search.py`
(`08_final_tuned_model.keras`, ajustado dentro de <=2023 con una validación
interna agrupada para EarlyStopping y los hiperparámetros óptimos de la
búsqueda de 40 trials) junto con su imputer y scaler
(`08_final_imputer.joblib`, `08_final_scaler.joblib`, ajustados ÚNICAMENTE
con datos de desarrollo <=2023) -- ninguno de los dos se re-ajusta aquí.

Decisión metodológica explícita (pedida por Miguel): ¿qué datos explicar?
  - Opción A: holdout temporal 2024 (n=81, 3 componentes) -- datos que el
    modelo final NUNCA vio, ni para entrenar ni para ajustar imputer/scaler.
    Coincide exactamente con qué conjunto explicaba `v3.4.ipynb` (su propio
    test set, nunca visto por el modelo que se explica).
  - Opción B: predicciones out-of-fold de la CV agrupada (4149 filas, 5
    modelos distintos, uno por fold) -- más robusto estadísticamente, pero
    NO son "un modelo", son 5 modelos con los mismos hiperparámetros pero
    pesos distintos; agregar sus SHAP values mezclaría explicaciones de
    modelos distintos.
  Se elige la OPCIÓN A (holdout 2024) como análisis principal, porque
  mantiene la correspondencia 1:1 "un modelo, explicado sobre datos que ese
  modelo nunca vio" -- exactamente la lógica de `v3.4.ipynb`. Se documenta
  explícitamente la limitación de tamaño de muestra (n=81 vs. N_SAMPLE=800
  del original) en vez de forzar un sustituto.

Configuración fiel al original (`v3.4.ipynb`, sección 12, líneas ~393-420):
  K_BG = 80 (clusters de KMeans para resumir el background)
  N_SAMPLE = min(800, n_disponible)  -- aquí, min(800, 81) = 81 (TODO el holdout)
  Un `shap.KernelExplainer` por target, sobre el sub-modelo que aísla esa
  cabeza (`keras.Model(inputs=model.input, outputs=model.get_layer(name).output)`)
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from tensorflow import keras

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
SEED = 42

MODEL_SUFFIX = os.environ.get("SHAP_MODEL_SUFFIX", "")  # "" para el modelo real, "_smoketest" para validar el script
K_BG = int(os.environ.get("SHAP_K_BG", 80))
N_SAMPLE_MAX = int(os.environ.get("SHAP_N_SAMPLE_MAX", 800))

df = pd.read_csv(os.path.join(OUT_DIR, "clean_dataset.csv"))
with open(os.path.join(OUT_DIR, "feature_lists.json")) as f:
    feat_lists = json.load(f)
CLEAN_FEATURES = feat_lists["clean_features"]
TARGETS = feat_lists["targets"]

train_pool = df[df["split_bucket"] == "train_le2023"].reset_index(drop=True)
holdout_2024 = df[df["split_bucket"] == "holdout_2024"].reset_index(drop=True)

N_SAMPLE = min(N_SAMPLE_MAX, len(holdout_2024))
print(f"Explicando {N_SAMPLE} filas del holdout 2024 (de {len(holdout_2024)} disponibles; "
      f"N_SAMPLE original de v3.4.ipynb era min(800, n_test), aquí min(800,{len(holdout_2024)})={N_SAMPLE})")

model_path = os.path.join(OUT_DIR, f"08_final_tuned_model{MODEL_SUFFIX}.keras")
imputer_path = os.path.join(OUT_DIR, f"08_final_imputer{MODEL_SUFFIX}.joblib")
scaler_path = os.path.join(OUT_DIR, f"08_final_scaler{MODEL_SUFFIX}.joblib")
for p in (model_path, imputer_path, scaler_path):
    if not os.path.exists(p):
        raise FileNotFoundError(f"No existe {p} -- corre primero 08_dnn_hp_search.py (con el mismo MODEL_SUFFIX).")

model = keras.models.load_model(model_path)
imputer = joblib.load(imputer_path)
scaler = joblib.load(scaler_path)

# Reconstruir exactamente los mismos arrays escalados que usó 08_dnn_hp_search.py
# (imputer/scaler ya ajustados SOLO con <=2023 -- aquí solo se aplica transform)
X_train_scaled = scaler.transform(imputer.transform(train_pool[CLEAN_FEATURES]))
X_holdout_scaled = scaler.transform(imputer.transform(holdout_2024[CLEAN_FEATURES]))

if N_SAMPLE < len(holdout_2024):
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(holdout_2024), size=N_SAMPLE, replace=False)
else:
    sample_idx = np.arange(len(holdout_2024))
X_explain = X_holdout_scaled[sample_idx]

print(f"Background K-Means: resumiendo {X_train_scaled.shape[0]} filas de entrenamiento en {K_BG} clusters...")
km = KMeans(n_clusters=K_BG, n_init=10, random_state=SEED).fit(X_train_scaled)
bg_summary = km.cluster_centers_

shap_results = {}
feature_names = CLEAN_FEATURES

for target_name in TARGETS:
    print(f"\n=== SHAP para target: {target_name} ===")
    submodel = keras.Model(inputs=model.input, outputs=model.get_layer(target_name).output)

    def f_t(z, _submodel=submodel):
        return _submodel.predict(z, verbose=0).ravel()

    explainer = shap.KernelExplainer(f_t, bg_summary)
    shap_values = explainer.shap_values(X_explain, silent=True)
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:  # algunas versiones devuelven (n, features, 1)
        shap_values = shap_values[:, :, 0]

    np.save(os.path.join(OUT_DIR, f"09_shap_values_{target_name}{MODEL_SUFFIX}.npy"), shap_values)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
    shap_results[target_name] = ranking.to_dict()

    print(f"Top 10 features por |SHAP| medio ({target_name}):")
    print(ranking.head(10).to_string())

    # Plot opcional (si matplotlib está disponible) -- no crítico, se intenta
    # y se sigue de largo si falla.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False)
        plt.title(f"SHAP summary — {target_name} (modelo limpio, sin fuga, holdout 2024, n={N_SAMPLE})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"09_shap_summary_{target_name}{MODEL_SUFFIX}.png"), dpi=150)
        plt.close()
        print(f"Guardado plot: out/09_shap_summary_{target_name}{MODEL_SUFFIX}.png")
    except Exception as e:
        print(f"[SHAP plot] omitido para {target_name}: {e}")

with open(os.path.join(OUT_DIR, f"09_shap_summary{MODEL_SUFFIX}.json"), "w") as f:
    json.dump(
        {
            "model_suffix": MODEL_SUFFIX,
            "n_sample_explained": int(N_SAMPLE),
            "n_holdout_available": int(len(holdout_2024)),
            "k_background_clusters": K_BG,
            "per_target_mean_abs_shap": shap_results,
        },
        f,
        indent=2,
    )
print(f"\nGuardado: out/09_shap_summary{MODEL_SUFFIX}.json")
