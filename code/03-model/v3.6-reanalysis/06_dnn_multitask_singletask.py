"""
WP1 — DNN multi-tarea (arquitectura v3.4/v3.5, simplificada) + ablación
single-task (Reviewer 1 #25) — ESCRITO Y LISTO, **NO EJECUTADO** en esta
sesión.

Por qué no se ejecutó (evidencia, no una excusa vaga): en el entorno conda
local `mexihc-v36` (Python 3.11 x86_64 corriendo vía Rosetta 2 en un Mac
Apple Silicon, host arm64) se instaló `tensorflow==2.16.2` sin error
(`pip install tensorflow` terminó con "Successfully installed..."), pero
`import tensorflow` se queda COLGADO indefinidamente: el proceso entra en
estado `UNE` (uninterruptible sleep) y ni siquiera responde a `kill -9`.
Se probó:
  1. Import simple (`import tensorflow as tf`) — colgado >20 min.
  2. Con `CUDA_VISIBLE_DEVICES=-1` y `TF_CPP_MIN_LOG_LEVEL=3` — mismo
     resultado.
  3. Con el sandbox de la herramienta de shell deshabilitado
     (`dangerouslyDisableSandbox`) — mismo resultado, descarta que sea un
     problema de permisos del sandbox y apunta a una incompatibilidad real
     Rosetta/arm64 de esta build de TensorFlow en esta máquina.

Recomendación para Miguel (ver METODOLOGIA-V3.md): crear un entorno conda
NATIVO arm64 (`CONDA_SUBDIR=osx-arm64 conda create -n mexihc-v36-arm64
python=3.11`) e instalar `tensorflow-macos`+`tensorflow-metal` ahí, o correr
específicamente este script en Google Colab (que sí tiene TF funcional) con
`code/v3.6/data/Material/dataset_final_3_x.csv` subido o montado por Drive.

Este script reproduce, con las correcciones anti-fuga y de split ya
validadas en `03_build_clean_dataset.py`/`05_baselines_grouped_temporal.py`:
  - las features limpias definidas en `out/feature_lists.json` (15 en la
    configuración primaria sin fuga)
  - el mismo split agrupado-por-componente + temporal (train <=2023 / test
    2024), con validación interna agrupada para early stopping
  - una versión de la arquitectura de v3.4/v3.5 (tronco compartido + cabezas
    por target) pero con HIPERPARÁMETROS FIJOS razonables en vez de una
    búsqueda Bayesiana de 40 trials (`keras_tuner.BayesianOptimization`) —
    esa búsqueda completa es demasiado costosa para correr localmente sin
    saber aún si el import de TF funciona; se puede reactivar cambiando
    `USE_TUNER=True` una vez que TF funcione.
  - la ablación pedida: la MISMA arquitectura de tronco+cabeza, pero
    entrenando 5 modelos independientes de una sola salida (single-task)
    vs. 1 modelo con las 5 cabezas (multi-task), mismo split, mismos datos.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

USE_TUNER = False  # cambiar a True si se dispone de tiempo/cómputo y TF funciona

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
SEED = 42
N_FOLDS = 5
INNER_VAL_SIZE = float(os.environ.get("DNN_INNER_VAL_SIZE", 0.15))

# =====================================================================
# 0) Import de TF -- si esto cuelga, es la señal de que el bloqueo de
#    entorno sigue presente; correr este script con un timeout de proceso
#    (ej. `gtimeout 120 python 06_dnn_multitask_singletask.py`) para no
#    quedar colgado indefinidamente como pasó en esta sesión.
# =====================================================================
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

np.random.seed(SEED)
tf.random.set_seed(SEED)

df = pd.read_csv(os.path.join(OUT_DIR, "clean_dataset.csv"))
with open(os.path.join(OUT_DIR, "feature_lists.json")) as f:
    feat_lists = json.load(f)
CLEAN_FEATURES = feat_lists["clean_features"]
TARGETS = feat_lists["targets"]

train_pool = df[df["split_bucket"] == "train_le2023"].reset_index(drop=True)
holdout_2024 = df[df["split_bucket"] == "holdout_2024"].reset_index(drop=True)

ACT_BY_TARGET = {t: "sigmoid" for t in TARGETS}  # los 5 targets están en [0,1] (ver dataset_final_3_x.csv)
LOSS_BY_TARGET = {
    "avg_assignment_score": "mse",
    "missing_assignments": keras.losses.Huber(),
    "assignment_procrast_rate": "mse",
    "avg_exam_score": "mse",
    "exam_accuracy": "mse",
}


def build_multitask_model(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-3))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = []
    for name in TARGETS:
        h = layers.Dense(64, activation="relu")(x)
        out = layers.Dense(1, activation=ACT_BY_TARGET[name], name=name)(h)
        outputs.append(out)
    model = keras.Model(inp, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=LOSS_BY_TARGET,
        loss_weights={k: 1.0 for k in TARGETS},
        metrics={name: ["mae"] for name in TARGETS},
    )
    return model


def build_singletask_model(input_dim, target_name):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-3))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    h = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation=ACT_BY_TARGET[target_name], name=target_name)(h)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=LOSS_BY_TARGET[target_name],
        metrics=["mae"],
    )
    return model


def to_target_dict(y_df):
    return {name: y_df[[name]].values for name in TARGETS}


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def metrics_per_target(y_true_df, y_pred_mat):
    rows = []
    for i, t in enumerate(TARGETS):
        yt = y_true_df[t].to_numpy()
        yp = y_pred_mat[:, i]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rows.append({"target": t, "R2": r2, "MAE": float(np.mean(np.abs(yt - yp))), "RMSE": _rmse(yt, yp)})
    return pd.DataFrame(rows)


def preprocess(X_train_raw, X_val_raw):
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw[CLEAN_FEATURES]))
    X_val = scaler.transform(imputer.transform(X_val_raw[CLEAN_FEATURES]))
    return X_train, X_val


def fit_preprocess(X_train_raw, X_val_raw):
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw[CLEAN_FEATURES]))
    X_val = scaler.transform(imputer.transform(X_val_raw[CLEAN_FEATURES]))
    return X_train, X_val, imputer, scaler


def transform_with(imputer, scaler, X_raw):
    return scaler.transform(imputer.transform(X_raw[CLEAN_FEATURES]))


def train_and_eval_multitask(X_tr, y_tr_df, X_es_val, y_es_val_df, X_eval, y_eval_df):
    model = build_multitask_model(X_tr.shape[1])
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)]
    model.fit(X_tr, to_target_dict(y_tr_df), validation_data=(X_es_val, to_target_dict(y_es_val_df)),
              epochs=150, batch_size=64, verbose=0, callbacks=cb)
    y_pred_list = model.predict(X_eval, verbose=0)
    out_names = [n.split("/")[0] for n in model.output_names]
    name_to_idx = {n: i for i, n in enumerate(out_names)}
    y_pred_mat = np.column_stack([y_pred_list[name_to_idx[n]].ravel() for n in TARGETS])
    return metrics_per_target(y_eval_df, y_pred_mat), y_pred_mat


def train_and_eval_singletask(X_tr, y_tr_df, X_es_val, y_es_val_df, X_eval, y_eval_df):
    preds = np.zeros((X_eval.shape[0], len(TARGETS)))
    for i, t in enumerate(TARGETS):
        model = build_singletask_model(X_tr.shape[1], t)
        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)]
        model.fit(X_tr, y_tr_df[[t]].values, validation_data=(X_es_val, y_es_val_df[[t]].values),
                  epochs=150, batch_size=64, verbose=0, callbacks=cb)
        preds[:, i] = model.predict(X_eval, verbose=0).ravel()
    return metrics_per_target(y_eval_df, preds), preds


if __name__ == "__main__":
    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = train_pool["component_id"]

    results = {"multitask_cv": [], "singletask_cv": []}
    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(train_pool, groups=groups)):
        outer_train_raw = train_pool.iloc[tr_idx].reset_index(drop=True)
        outer_test_raw = train_pool.iloc[va_idx].reset_index(drop=True)
        inner_gss = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_SIZE, random_state=SEED + fold_i)
        inner_tr_idx, inner_val_idx = next(
            inner_gss.split(outer_train_raw, groups=outer_train_raw["component_id"])
        )
        inner_train_raw = outer_train_raw.iloc[inner_tr_idx]
        inner_val_raw = outer_train_raw.iloc[inner_val_idx]
        X_inner_train, X_inner_val, imputer_fold, scaler_fold = fit_preprocess(inner_train_raw, inner_val_raw)
        X_outer_test = transform_with(imputer_fold, scaler_fold, outer_test_raw)

        m_mt, _ = train_and_eval_multitask(
            X_inner_train, inner_train_raw[TARGETS], X_inner_val, inner_val_raw[TARGETS],
            X_outer_test, outer_test_raw[TARGETS]
        )
        m_mt["fold"] = fold_i
        results["multitask_cv"].append(m_mt)

        m_st, _ = train_and_eval_singletask(
            X_inner_train, inner_train_raw[TARGETS], X_inner_val, inner_val_raw[TARGETS],
            X_outer_test, outer_test_raw[TARGETS]
        )
        m_st["fold"] = fold_i
        results["singletask_cv"].append(m_st)
        print(f"fold {fold_i} listo")

    # Modelo final con validación interna agrupada dentro de <=2023; holdout
    # 2024 se reserva exclusivamente para evaluación.
    gss_final = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_SIZE, random_state=SEED)
    fin_tr_idx, fin_val_idx = next(gss_final.split(train_pool, groups=train_pool["component_id"]))
    final_train_raw = train_pool.iloc[fin_tr_idx]
    final_val_raw = train_pool.iloc[fin_val_idx]
    X_fin_tr, X_fin_val, imputer_final, scaler_final = fit_preprocess(final_train_raw, final_val_raw)
    X_holdout = transform_with(imputer_final, scaler_final, holdout_2024)
    m_mt_final, _ = train_and_eval_multitask(
        X_fin_tr, final_train_raw[TARGETS], X_fin_val, final_val_raw[TARGETS],
        X_holdout, holdout_2024[TARGETS]
    )
    m_st_final, _ = train_and_eval_singletask(
        X_fin_tr, final_train_raw[TARGETS], X_fin_val, final_val_raw[TARGETS],
        X_holdout, holdout_2024[TARGETS]
    )

    summary = {
        "multitask_cv_mean": pd.concat(results["multitask_cv"]).groupby("target")[["R2", "MAE", "RMSE"]].mean().to_dict(orient="index"),
        "singletask_cv_mean": pd.concat(results["singletask_cv"]).groupby("target")[["R2", "MAE", "RMSE"]].mean().to_dict(orient="index"),
        "multitask_holdout_2024": m_mt_final.to_dict(orient="records"),
        "singletask_holdout_2024": m_st_final.to_dict(orient="records"),
    }
    with open(os.path.join(OUT_DIR, "06_dnn_ablation_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Guardado: out/06_dnn_ablation_results.json")
