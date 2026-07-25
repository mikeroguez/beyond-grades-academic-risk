"""
WP1 — Búsqueda de hiperparámetros completa para el DNN multi-tarea, fiel al
notebook original (`code/03-model/v3.4.ipynb`), pero sobre el dataset LIMPIO
(sin fuga, Sección 2 de METODOLOGIA-V3.md) y con el protocolo de split
agrupado-por-componente + holdout temporal ya establecido (Secciones 3-4).

Pedido explícito de Miguel (vía coordinador, 18 jul 2026): la corrida de
`06_dnn_multitask_singletask.py` de la sesión anterior usó hiperparámetros
FIJOS (256-128, dropout 0.3, lr 1e-3) para poder ejecutar algo rápido dado
el bloqueo de TensorFlow que existía en ese momento. Con ese bloqueo ya
resuelto (venv arm64 nativo, `~/.venvs/mexihc-v36-arm64`), Miguel quiere
fidelidad completa con el paper: la MISMA búsqueda Bayesiana de 40 trials
que usó `v3.4.ipynb`.

Configuración verificada línea por línea contra `v3.4.ipynb`
(`jupyter nbconvert --to script v3.4.ipynb --stdout`, líneas 139-181):

    tuner = kt.BayesianOptimization(
        hypermodel=build_model, objective="val_loss",
        max_trials=40, num_initial_points=10, ...
    )
    early_stop_tuning = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    )
    tuner.search(X_train, ..., epochs=40, batch_size=64,
                 validation_data=(X_val, ...), callbacks=[early_stop_tuning])

    build_model(hp) rangos EXACTOS:
        trunk_units_1: Int(128, 512, step=64)
        trunk_l2_1: Choice([1e-2, 5e-3])
        trunk_dropout_1: Float(0.3, 0.5, step=0.1)
        trunk_depth_extra: Int(0, 2)
          trunk_units_{i+2}: Int(64, 256, step=64)
          trunk_l2_{i+2}: Choice([5e-3, 1e-3])
          trunk_dropout_{i+2}: Float(0.2, 0.4, step=0.1)
        {target}_head_units: Int(32, 128, step=32)  (una por target)
        learning_rate: Choice([1e-3, 5e-4, 1e-4])

    Entrenamiento FINAL (post-búsqueda, distinto de la búsqueda):
        epochs=150, batch_size=64,
        EarlyStopping(patience=12, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=6, min_lr=1e-6)

Diferencia deliberada frente al notebook original (documentada, no oculta):
v3.4.ipynb hacía la búsqueda de hiperparámetros sobre un split ALEATORIO
agrupado solo por `uid_hash` (GroupShuffleSplit 80/20 dos veces, para
train/val/test). Aquí, para no repetir la fuga de agrupamiento que motivó
esta ronda de revisión (Reviewer 1 #7-8), la búsqueda usa un
GroupShuffleSplit sobre `component_id` (componentes conexas
estudiante+curso, Sección 4) dentro del pool `train_le2023` -- un solo
split train/val (no la CV de 5 folds) porque así es exactamente como
`v3.4.ipynb` hace la búsqueda (un solo split, no CV dentro de la búsqueda).
La evaluación POSTERIOR (Sección 2 de este script) sí usa la CV agrupada de
5 folds + el holdout temporal 2024, igual que en `06_dnn_multitask_singletask.py`,
para que el resultado sea comparable número a número con el de anoche.

Guarda: mejores hiperparámetros, historial de trials, modelo final
reentrenado en todo <=2023 (para que 09_shap_analysis.py lo reutilice sin
tener que re-entrenar), y las métricas CV + holdout con esos hiperparámetros.
"""
import json
import os
import time

import keras_tuner as kt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
TUNER_DIR = os.path.join(OUT_DIR, "tuner_results_v36")

# Valores fieles a v3.4.ipynb (ver docstring). Se pueden sobreescribir por
# variable de entorno SOLO para un smoke test rápido antes de la corrida
# completa -- de no definirse, corren los valores fieles al original.
MAX_TRIALS = int(os.environ.get("DNN_SEARCH_MAX_TRIALS", 40))
NUM_INITIAL_POINTS = int(os.environ.get("DNN_SEARCH_INITIAL_POINTS", 10))
SEARCH_EPOCHS = int(os.environ.get("DNN_SEARCH_EPOCHS", 40))
SEARCH_PATIENCE = int(os.environ.get("DNN_SEARCH_PATIENCE", 6))
FINAL_EPOCHS = int(os.environ.get("DNN_FINAL_EPOCHS", 150))
FINAL_PATIENCE = int(os.environ.get("DNN_FINAL_PATIENCE", 12))
N_FOLDS = int(os.environ.get("DNN_N_FOLDS", 5))
RESULTS_SUFFIX = os.environ.get("DNN_RESULTS_SUFFIX", "")
SKIP_HP_SEARCH = os.environ.get("DNN_SKIP_HP_SEARCH", "0") == "1"
INNER_VAL_SIZE = float(os.environ.get("DNN_INNER_VAL_SIZE", 0.15))

df = pd.read_csv(os.path.join(OUT_DIR, "clean_dataset.csv"))
with open(os.path.join(OUT_DIR, "feature_lists.json")) as f:
    feat_lists = json.load(f)
CLEAN_FEATURES = feat_lists["clean_features"]
TARGETS = feat_lists["targets"]

train_pool = df[df["split_bucket"] == "train_le2023"].reset_index(drop=True)
holdout_2024 = df[df["split_bucket"] == "holdout_2024"].reset_index(drop=True)

ACT_BY_TARGET = {t: "sigmoid" for t in TARGETS}
LOSS_BY_TARGET = {
    "avg_assignment_score": "mse",
    "missing_assignments": keras.losses.Huber(),
    "assignment_procrast_rate": "mse",
    "avg_exam_score": "mse",
    "exam_accuracy": "mse",
}
LOSS_WEIGHTS = {k: 1.0 for k in TARGETS}


def build_model(hp: "kt.HyperParameters"):
    inp = layers.Input(shape=(len(CLEAN_FEATURES),))
    x = layers.Dense(
        hp.Int("trunk_units_1", 128, 512, step=64),
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(hp.Choice("trunk_l2_1", [1e-2, 5e-3])),
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(hp.Float("trunk_dropout_1", 0.3, 0.5, step=0.1))(x)
    for i in range(hp.Int("trunk_depth_extra", 0, 2)):
        x = layers.Dense(
            hp.Int(f"trunk_units_{i + 2}", 64, 256, step=64),
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(hp.Choice(f"trunk_l2_{i + 2}", [5e-3, 1e-3])),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(hp.Float(f"trunk_dropout_{i + 2}", 0.2, 0.4, step=0.1))(x)
    outputs = []
    for name in TARGETS:
        h = layers.Dense(hp.Int(f"{name}_head_units", 32, 128, step=32), activation="relu")(x)
        out = layers.Dense(1, activation=ACT_BY_TARGET[name], name=name)(h)
        outputs.append(out)
    model = keras.Model(inp, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4])),
        loss=LOSS_BY_TARGET,
        loss_weights=LOSS_WEIGHTS,
        metrics={name: ["mae"] for name in TARGETS},
    )
    return model


def to_target_dict(y_df):
    return {name: y_df[[name]].values for name in TARGETS}


def preprocess(X_train_raw, X_val_raw):
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw[CLEAN_FEATURES]))
    X_val = scaler.transform(imputer.transform(X_val_raw[CLEAN_FEATURES]))
    return X_train, X_val, imputer, scaler


def transform_with(imputer, scaler, X_raw):
    return scaler.transform(imputer.transform(X_raw[CLEAN_FEATURES]))


def hyperparameters_from_values(values):
    hp = kt.HyperParameters()
    hp.values.update(values)
    return hp


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


def predict_matrix(model, X):
    y_pred_list = model.predict(X, verbose=0)
    out_names = [n.split("/")[0] for n in model.output_names]
    name_to_idx = {n: i for i, n in enumerate(out_names)}
    return np.column_stack([y_pred_list[name_to_idx[n]].ravel() for n in TARGETS])


def save_final_history_artifacts(history, suffix=""):
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))
    csv_path = os.path.join(OUT_DIR, "08_final_training_history" + suffix + ".csv")
    png_path = os.path.join(OUT_DIR, "08_final_learning_curve" + suffix + ".png")
    json_path = os.path.join(OUT_DIR, "08_final_learning_curve_source" + suffix + ".json")
    history_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8.5, 4.8), dpi=300)
    plt.plot(history_df["epoch"], history_df["loss"], label="Training loss", color="#1f77b4", linewidth=2.2)
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation loss", color="#d62728", linewidth=2.2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.22, linewidth=0.8)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    with open(json_path, "w") as f:
        json.dump(
            {
                "curve_type": "final_training",
                "seed": SEED,
                "final_epochs_configured": FINAL_EPOCHS,
                "final_patience": FINAL_PATIENCE,
                "epochs_observed": int(len(history_df)),
                "output_png": png_path,
                "output_csv": csv_path,
            },
            f,
            indent=2,
        )
    return {"csv": csv_path, "png": png_path, "json": json_path}


if __name__ == "__main__":
    t_start = time.time()

    # =================================================================
    # 1) BÚSQUEDA DE HIPERPARÁMETROS -- un solo split train/val agrupado
    #    por componente dentro de <=2023 (fiel a v3.4.ipynb: un solo
    #    split, no CV, para la búsqueda)
    # =================================================================
    if SKIP_HP_SEARCH:
        best_hp_path = os.path.join(OUT_DIR, "08_best_hyperparameters" + RESULTS_SUFFIX + ".json")
        with open(best_hp_path) as f:
            best_hp = hyperparameters_from_values(json.load(f))
        search_duration = 0.0
        print(f"[HP search] omitida por DNN_SKIP_HP_SEARCH=1; usando {best_hp_path}")
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
        search_tr_idx, search_val_idx = next(gss.split(train_pool, groups=train_pool["component_id"]))
        search_train = train_pool.iloc[search_tr_idx]
        search_val = train_pool.iloc[search_val_idx]
        print(f"[HP search] train={len(search_train)} filas / {search_train['component_id'].nunique()} componentes | "
              f"val={len(search_val)} filas / {search_val['component_id'].nunique()} componentes")

        X_search_train, X_search_val, _, _ = preprocess(search_train, search_val)
        y_search_train, y_search_val = search_train[TARGETS], search_val[TARGETS]

        os.makedirs(TUNER_DIR, exist_ok=True)
        tuner = kt.BayesianOptimization(
            hypermodel=build_model,
            objective="val_loss",
            max_trials=MAX_TRIALS,
            num_initial_points=NUM_INITIAL_POINTS,
            directory=TUNER_DIR,
            project_name=f"dnn_multitask_bayes_v36{RESULTS_SUFFIX}",
            overwrite=True,
            seed=SEED,
        )
        early_stop_tuning = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=SEARCH_PATIENCE, restore_best_weights=True, verbose=0
        )

        print(f"[HP search] iniciando búsqueda Bayesiana: {MAX_TRIALS} trials, "
              f"{NUM_INITIAL_POINTS} puntos iniciales, hasta {SEARCH_EPOCHS} épocas/trial "
              f"(EarlyStopping patience={SEARCH_PATIENCE})")
        t0 = time.time()
        tuner.search(
            X_search_train, to_target_dict(y_search_train),
            epochs=SEARCH_EPOCHS, batch_size=64,
            validation_data=(X_search_val, to_target_dict(y_search_val)),
            callbacks=[early_stop_tuning], verbose=2,
        )
        search_duration = time.time() - t0
        print(f"[HP search] Duración total de la búsqueda: {search_duration / 60:.1f} min "
              f"({search_duration / MAX_TRIALS:.1f} s/trial en promedio)")

        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Historial de todos los trials (val_loss final de cada uno), para que
        # quede trazable qué tan plana o picuda fue la superficie de búsqueda.
        trial_history = []
        for trial_id, trial in tuner.oracle.trials.items():
            trial_history.append({
                "trial_id": trial_id,
                "status": str(trial.status),
                "score": trial.score,
                "hyperparameters": trial.hyperparameters.values,
            })
        with open(os.path.join(OUT_DIR, "08_tuner_trial_history" + RESULTS_SUFFIX + ".json"), "w") as f:
            json.dump(trial_history, f, indent=2, default=str)
        with open(os.path.join(OUT_DIR, "08_best_hyperparameters" + RESULTS_SUFFIX + ".json"), "w") as f:
            json.dump(best_hp.values, f, indent=2)
        print(f"Guardado: out/08_tuner_trial_history{RESULTS_SUFFIX}.json ({len(trial_history)} trials), "
              f"out/08_best_hyperparameters{RESULTS_SUFFIX}.json")

    print("\n🔍 Mejores hiperparámetros usados:")
    for k in best_hp.values.keys():
        print(f"  - {k}: {best_hp.get(k)}")

    # =================================================================
    # 2) EVALUACIÓN:
    #    CV agrupada de 5 folds dentro de <=2023 con validación interna
    #    agrupada para EarlyStopping/ReduceLROnPlateau, y modelo final con
    #    validación interna <=2023 evaluado en el holdout temporal 2024.
    # =================================================================
    def train_final_multitask(X_tr, y_tr_df, X_va, y_va_df):
        model = build_model(best_hp)
        cb = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=FINAL_PATIENCE, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=0),
        ]
        history = model.fit(
            X_tr, to_target_dict(y_tr_df),
            validation_data=(X_va, to_target_dict(y_va_df)),
            epochs=FINAL_EPOCHS, batch_size=64, verbose=0, callbacks=cb,
        )
        return model, history

    print("\n[Eval] CV agrupada (5 folds) con inner-validation agrupada dentro de <=2023, con best_hp")
    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = train_pool["component_id"]
    fold_metrics = []
    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(train_pool, groups=groups)):
        t_fold = time.time()
        outer_train_raw = train_pool.iloc[tr_idx].reset_index(drop=True)
        outer_test_raw = train_pool.iloc[va_idx].reset_index(drop=True)
        inner_gss = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_SIZE, random_state=SEED + fold_i)
        inner_tr_idx, inner_val_idx = next(
            inner_gss.split(outer_train_raw, groups=outer_train_raw["component_id"])
        )
        inner_train_raw = outer_train_raw.iloc[inner_tr_idx]
        inner_val_raw = outer_train_raw.iloc[inner_val_idx]
        y_inner_train = inner_train_raw[TARGETS]
        y_inner_val = inner_val_raw[TARGETS]
        y_outer_test = outer_test_raw[TARGETS]
        X_inner_train, X_inner_val, imputer_fold, scaler_fold = preprocess(inner_train_raw, inner_val_raw)
        X_outer_test = transform_with(imputer_fold, scaler_fold, outer_test_raw)

        model, _ = train_final_multitask(X_inner_train, y_inner_train, X_inner_val, y_inner_val)
        y_pred = predict_matrix(model, X_outer_test)
        m = metrics_per_target(y_outer_test, y_pred)
        m["fold"] = fold_i
        fold_metrics.append(m)
        print(f"  fold {fold_i} listo en {(time.time() - t_fold) / 60:.1f} min")
        keras.backend.clear_session()

    cv_all = pd.concat(fold_metrics, ignore_index=True)
    cv_mean = cv_all.groupby("target")[["R2", "MAE", "RMSE"]].mean().reindex(TARGETS)
    print("\n--- DNN (tuned, 40 trials) CV agrupada <=2023, media sobre 5 folds ---")
    print(cv_mean.to_string())
    cv_global = {
        "R2": float(cv_mean["R2"].mean()),
        "MAE": float(cv_mean["MAE"].mean()),
        "RMSE": float(cv_mean["RMSE"].mean()),
    }
    print("Global (aritmético):", cv_global)

    print("\n[Eval] Modelo final: early stopping con validación interna <=2023, evaluación en holdout 2024")
    gss_final = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    fin_tr_idx, fin_val_idx = next(gss_final.split(train_pool, groups=train_pool["component_id"]))
    final_train_raw = train_pool.iloc[fin_tr_idx]
    final_val_raw = train_pool.iloc[fin_val_idx]
    X_fin_tr, X_fin_val, imputer_final, scaler_final = preprocess(final_train_raw, final_val_raw)
    X_holdout = transform_with(imputer_final, scaler_final, holdout_2024)
    y_fin_tr = final_train_raw[TARGETS]
    y_fin_val = final_val_raw[TARGETS]

    final_model, final_history = train_final_multitask(X_fin_tr, y_fin_tr, X_fin_val, y_fin_val)
    history_artifacts = save_final_history_artifacts(final_history, RESULTS_SUFFIX)
    y_pred_holdout = predict_matrix(final_model, X_holdout)
    m_holdout = metrics_per_target(holdout_2024[TARGETS], y_pred_holdout)
    print("\n--- DNN (tuned, 40 trials) holdout 2024 (n=81, 3 componentes) ---")
    print(m_holdout.to_string(index=False))
    holdout_global = {
        "R2": float(m_holdout["R2"].mean()),
        "MAE": float(m_holdout["MAE"].mean()),
        "RMSE": float(m_holdout["RMSE"].mean()),
    }
    print("Global (aritmético):", holdout_global)

    # Guardar el modelo final + preprocesadores para que 09_shap_analysis.py
    # los reutilice sin reentrenar.
    final_model.save(os.path.join(OUT_DIR, "08_final_tuned_model" + RESULTS_SUFFIX + ".keras"))
    import joblib
    joblib.dump(imputer_final, os.path.join(OUT_DIR, "08_final_imputer" + RESULTS_SUFFIX + ".joblib"))
    joblib.dump(scaler_final, os.path.join(OUT_DIR, "08_final_scaler" + RESULTS_SUFFIX + ".joblib"))

    results = {
        "search_duration_seconds": search_duration,
        "max_trials": MAX_TRIALS,
        "hyperparameter_search_reused": SKIP_HP_SEARCH,
        "best_hyperparameters": best_hp.values,
        "cv_grouped_le2023": {
            "per_target_mean_over_folds": cv_mean.reset_index().rename(columns={"index": "target"}).to_dict(orient="records"),
            "global_arithmetic_mean": cv_global,
        },
        "holdout_2024": {
            "per_target": m_holdout.to_dict(orient="records"),
            "global_arithmetic_mean": holdout_global,
        },
    }
    with open(os.path.join(OUT_DIR, "08_dnn_tuned_results" + RESULTS_SUFFIX + ".json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGuardado: out/08_dnn_tuned_results{RESULTS_SUFFIX}.json, out/08_final_tuned_model{RESULTS_SUFFIX}.keras, "
          f"out/08_final_imputer{RESULTS_SUFFIX}.joblib, out/08_final_scaler{RESULTS_SUFFIX}.joblib")
    print("Historia final guardada:", history_artifacts)
    print(f"\nTiempo total del script: {(time.time() - t_start) / 60:.1f} min")
