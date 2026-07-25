"""
v3.8 reanálisis — Etapa 3: búsqueda de hiperparámetros de 40 trials (fiel
a `v3.4.ipynb`, misma configuración que
`code/03-model/v3.6-reanalysis/08_dnn_hp_search.py`) + evaluación CV
agrupada ≤2023 + holdout 2024, para UN corte temporal a la vez
(25/50/75%), sobre el dataset enriquecido de la Etapa 2 (features
reseleccionadas por Pearson+LassoCV específicas de ese corte).

Uso:
    CUTOFF=50 python 04_dnn_hp_search_by_cutoff.py
    CUTOFF=75 python 04_dnn_hp_search_by_cutoff.py
    CUTOFF=25 python 04_dnn_hp_search_by_cutoff.py

Pedido explícito de Miguel (vía coordinador, 19 jul 2026): búsqueda
INDEPENDIENTE por corte -- no se reutilizan hiperparámetros de un corte en
otro, la señal disponible es distinta en cada uno.

Todo lo demás es idéntico en espíritu a `08_dnn_hp_search.py` de
v3.6-reanalysis (arquitectura, rangos de hiperparámetros, protocolo de
entrenamiento/evaluación) -- ver ese script para el detalle línea por
línea contra `v3.4.ipynb`. Aquí solo cambia: (a) el dataset de entrada
(`out/cutoff_X_clean_dataset.csv`, features variables por corte, 18-21),
y (b) el guardado incremental por corte (para que si el proceso se corta
a medio camino, lo ya completado quede en disco).

⚠️ BUG REAL encontrado y corregido (19 jul 2026, reportado por Miguel tras
revisar `04_dnn_tuned_results_cutoff50.json`): la primera corrida del
corte 50% dio R²≈0 en los 5 targets a la vez (avg_assignment_score=-0.028,
missing_assignments=-0.078, assignment_procrast_rate=-0.049,
avg_exam_score=-0.006, exam_accuracy=-0.003) -- la firma de un modelo que
colapsa a predecir la media, no de "la DNN es peor que RF" (RandomForest,
mismo dataset, mismo preprocesamiento de imputación/escalado, dio
R²=0.395 global con targets individuales razonables). Causa raíz
diagnosticada y confirmada empíricamente: `forum_time_range` tiene 79% de
valores faltantes (896/4230 no-nulos) -- tras la imputación por mediana,
más del 50% del pool de entrenamiento queda con el mismo valor imputado,
lo que colapsa su rango intercuartílico (Q1=Q3=0) dentro del fold de
entrenamiento. `sklearn.preprocessing.RobustScaler` maneja IQR=0
reemplazando `scale_` por 1.0 (para evitar división por cero) -- pero eso
significa que el "escalado" para esa columna se vuelve un no-op, y los
valores genuinos no-imputados (hasta 99,703 minutos ≈ 69 días) entran a la
red SIN reducir de escala, con una magnitud ~370 veces mayor que la
siguiente feature más extrema (`assignment_grade_trend_slope`, hasta
~266) y varios órdenes de magnitud por encima de cualquier otra
(típicamente entre -10 y 10). Ese único valor extremo por fila satura las
sumas ponderadas de la primera capa densa (compartida entre los 5
targets), lo que explica que el colapso sea simultáneo en las 5 salidas
(comparten tronco) y no en un target aislado. RandomForest no sufre esto
porque sus splits dependen del ORDEN de los valores, no de su magnitud.
También explica la duración anómala de la búsqueda (183 min vs. ~114 min
en v3.6 con más datos): con activaciones/gradientes inestables, muchos
trials probablemente tardaron más en estabilizar antes de que
EarlyStopping actuara.

**Corrección aplicada:** `preprocess()` ahora recorta (`np.clip`) las
features YA escaladas a un rango [-CLIP_ABS, CLIP_ABS] (por defecto ±10,
generoso frente al resto de las features, que rara vez superan ±5) antes
de entrar a la red. Ajustado SOLO con estadísticos de la partición de
entrenamiento (el propio `RobustScaler`), el clip es una operación fija
sin parámetros nuevos que aprender de los datos, así que no reintroduce
ningún tipo de fuga. No se tocó `02_etapa2_ablation.py` (RandomForest) --
ese resultado (R²=0.395 en el corte 50%) sigue siendo válido tal cual,
esta corrección es específica de la sensibilidad de las redes neuronales
a la magnitud de las features, no de los datos en sí.
"""
import json
import os
import time

if os.environ.get("V38_TF_CPU_ONLY") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras_tuner as kt
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
if os.environ.get("V38_TF_CPU_ONLY") == "1":
    try:
        tf.config.set_visible_devices([], "GPU")
        print("[TensorFlow] GPU deshabilitada por V38_TF_CPU_ONLY=1")
    except RuntimeError as exc:
        print(f"[TensorFlow] No se pudo deshabilitar GPU despues de inicializar runtime: {exc}")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("V38_REANALYSIS_OUT_DIR", os.path.join(HERE, "out"))
os.makedirs(OUT_DIR, exist_ok=True)

CUTOFF = os.environ.get("CUTOFF")
if CUTOFF not in ("25", "50", "75"):
    raise ValueError(f"Define CUTOFF=25|50|75 como variable de entorno (recibido: {CUTOFF!r})")
CUTOFF = int(CUTOFF)

TUNER_DIR = os.path.join(OUT_DIR, f"tuner_results_cutoff_{CUTOFF}")  # excluido vía .gitignore de este directorio

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
# Corrección del bug de escala descrito arriba: recorte post-RobustScaler.
CLIP_ABS = float(os.environ.get("DNN_CLIP_ABS", 10.0))

TARGETS = ["avg_assignment_score", "missing_assignments", "assignment_procrast_rate", "avg_exam_score", "exam_accuracy"]

df = pd.read_csv(os.path.join(OUT_DIR, f"cutoff_{CUTOFF}_clean_dataset.csv"))
with open(os.path.join(OUT_DIR, "03_selected_features_by_cutoff.json")) as f:
    selected_by_cutoff = json.load(f)
CLEAN_FEATURES = selected_by_cutoff[str(CUTOFF)]
print(f"[corte {CUTOFF}%] dataset: {df.shape}, {len(CLEAN_FEATURES)} features: {CLEAN_FEATURES}")

train_pool = df[df["split_bucket"] == "train_le2023"].reset_index(drop=True)
holdout_2024 = df[df["split_bucket"] == "holdout_2024"].reset_index(drop=True)
print(f"[corte {CUTOFF}%] train_le2023: {len(train_pool)} filas / {train_pool['component_id'].nunique()} componentes | "
      f"holdout_2024: {len(holdout_2024)} filas / {holdout_2024['component_id'].nunique()} componentes")

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


def preprocess(X_train_raw, X_val_raw, verbose=False):
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw[CLEAN_FEATURES]))
    X_val = scaler.transform(imputer.transform(X_val_raw[CLEAN_FEATURES]))

    # ⚠️ Corrección del bug de escala (ver docstring del módulo): algunas
    # features muy dispersas (ej. forum_time_range, ~79% NaN) colapsan su
    # IQR de entrenamiento a 0 -> RobustScaler cae en su fallback
    # scale_=1.0 -> los valores genuinos no-imputados entran a la red sin
    # reducir de escala (hasta ~98000 en unidades ya "escaladas"). Se
    # recorta DESPUÉS de escalar, con estadísticos ya ajustados solo en
    # train -- no aprende nada nuevo de los datos, solo acota el rango.
    n_clipped_train = int(np.sum(np.abs(X_train) > CLIP_ABS))
    n_clipped_val = int(np.sum(np.abs(X_val) > CLIP_ABS))
    if verbose and (n_clipped_train or n_clipped_val):
        max_before = float(np.max(np.abs(X_train)))
        print(f"    [preprocess] recortando a ±{CLIP_ABS}: {n_clipped_train} valores en train, "
              f"{n_clipped_val} en val (máximo absoluto antes del recorte: {max_before:.1f})")
    X_train = np.clip(X_train, -CLIP_ABS, CLIP_ABS)
    X_val = np.clip(X_val, -CLIP_ABS, CLIP_ABS)

    return X_train, X_val, imputer, scaler


def transform_with(imputer, scaler, X_raw):
    X = scaler.transform(imputer.transform(X_raw[CLEAN_FEATURES]))
    return np.clip(X, -CLIP_ABS, CLIP_ABS)


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


if __name__ == "__main__":
    t_start = time.time()

    # =================================================================
    # 1) BÚSQUEDA -- un solo split train/val agrupado por componente
    # =================================================================
    if SKIP_HP_SEARCH:
        best_hp_path = os.path.join(OUT_DIR, f"04_best_hyperparameters_cutoff{CUTOFF}{RESULTS_SUFFIX}.json")
        with open(best_hp_path) as f:
            best_hp = hyperparameters_from_values(json.load(f))
        search_duration = 0.0
        print(f"[corte {CUTOFF}%][HP search] omitida por DNN_SKIP_HP_SEARCH=1; usando {best_hp_path}")
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
        search_tr_idx, search_val_idx = next(gss.split(train_pool, groups=train_pool["component_id"]))
        search_train = train_pool.iloc[search_tr_idx]
        search_val = train_pool.iloc[search_val_idx]
        print(f"[corte {CUTOFF}%][HP search] train={len(search_train)} filas / {search_train['component_id'].nunique()} "
              f"componentes | val={len(search_val)} filas / {search_val['component_id'].nunique()} componentes")

        X_search_train, X_search_val, _, _ = preprocess(search_train, search_val, verbose=True)
        y_search_train, y_search_val = search_train[TARGETS], search_val[TARGETS]

        os.makedirs(TUNER_DIR, exist_ok=True)
        tuner = kt.BayesianOptimization(
            hypermodel=build_model,
            objective="val_loss",
            max_trials=MAX_TRIALS,
            num_initial_points=NUM_INITIAL_POINTS,
            directory=TUNER_DIR,
            project_name=f"dnn_multitask_bayes_cutoff{CUTOFF}{RESULTS_SUFFIX}",
            overwrite=True,
            seed=SEED,
        )
        early_stop_tuning = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=SEARCH_PATIENCE, restore_best_weights=True, verbose=0
        )

        print(f"[corte {CUTOFF}%][HP search] iniciando: {MAX_TRIALS} trials, {NUM_INITIAL_POINTS} puntos iniciales, "
              f"hasta {SEARCH_EPOCHS} épocas/trial (patience={SEARCH_PATIENCE})")
        t0 = time.time()
        tuner.search(
            X_search_train, to_target_dict(y_search_train),
            epochs=SEARCH_EPOCHS, batch_size=64,
            validation_data=(X_search_val, to_target_dict(y_search_val)),
            callbacks=[early_stop_tuning], verbose=2,
        )
        search_duration = time.time() - t0
        print(f"[corte {CUTOFF}%][HP search] Duración: {search_duration / 60:.1f} min "
              f"({search_duration / MAX_TRIALS:.1f} s/trial en promedio)")
        keras.backend.clear_session()

        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

        trial_history = []
        for trial_id, trial in tuner.oracle.trials.items():
            trial_history.append({
                "trial_id": trial_id, "status": str(trial.status), "score": trial.score,
                "hyperparameters": trial.hyperparameters.values,
            })
        with open(os.path.join(OUT_DIR, f"04_tuner_trial_history_cutoff{CUTOFF}{RESULTS_SUFFIX}.json"), "w") as f:
            json.dump(trial_history, f, indent=2, default=str)
        with open(os.path.join(OUT_DIR, f"04_best_hyperparameters_cutoff{CUTOFF}{RESULTS_SUFFIX}.json"), "w") as f:
            json.dump(best_hp.values, f, indent=2)
        print(f"Guardado (corte {CUTOFF}%): 04_tuner_trial_history_cutoff{CUTOFF}{RESULTS_SUFFIX}.json "
              f"({len(trial_history)} trials), 04_best_hyperparameters_cutoff{CUTOFF}{RESULTS_SUFFIX}.json")

    print(f"\n[corte {CUTOFF}%] Mejores hiperparámetros usados:")
    for k in best_hp.values.keys():
        print(f"  - {k}: {best_hp.get(k)}")

    # =================================================================
    # 2) EVALUACIÓN: CV agrupada 5-fold con validación interna agrupada
    #    dentro de <=2023 + holdout 2024
    # =================================================================
    def train_final_multitask(X_tr, y_tr_df, X_va, y_va_df):
        keras.backend.clear_session()
        model = build_model(best_hp)
        cb = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=FINAL_PATIENCE, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=0),
        ]
        model.fit(
            X_tr, to_target_dict(y_tr_df),
            validation_data=(X_va, to_target_dict(y_va_df)),
            epochs=FINAL_EPOCHS, batch_size=64, verbose=0, callbacks=cb,
        )
        return model

    print(f"\n[corte {CUTOFF}%][Eval] CV agrupada (5 folds) con inner-validation agrupada dentro de <=2023, con best_hp")
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
        X_inner_train, X_inner_val, imputer_fold, scaler_fold = preprocess(
            inner_train_raw, inner_val_raw, verbose=(fold_i == 0)
        )
        X_outer_test = transform_with(imputer_fold, scaler_fold, outer_test_raw)

        model = train_final_multitask(X_inner_train, inner_train_raw[TARGETS], X_inner_val, inner_val_raw[TARGETS])
        y_pred = predict_matrix(model, X_outer_test)
        m = metrics_per_target(outer_test_raw[TARGETS], y_pred)
        m["fold"] = fold_i
        fold_metrics.append(m)
        print(f"  [corte {CUTOFF}%] fold {fold_i} listo en {(time.time() - t_fold) / 60:.1f} min")
        keras.backend.clear_session()

    cv_all = pd.concat(fold_metrics, ignore_index=True)
    cv_mean = cv_all.groupby("target")[["R2", "MAE", "RMSE"]].mean().reindex(TARGETS)
    print(f"\n--- corte {CUTOFF}% -- DNN (tuned) CV agrupada <=2023, media sobre 5 folds ---")
    print(cv_mean.to_string())
    cv_global = {
        "R2": float(cv_mean["R2"].mean()), "MAE": float(cv_mean["MAE"].mean()), "RMSE": float(cv_mean["RMSE"].mean()),
    }
    print("Global (aritmético):", cv_global)

    # Guardado INCREMENTAL de la CV, antes de entrenar el modelo final --
    # si el proceso se corta en el paso siguiente, esto ya queda en disco.
    partial_results = {
        "cutoff": CUTOFF,
        "features": CLEAN_FEATURES,
        "search_duration_seconds": search_duration,
        "max_trials": MAX_TRIALS,
        "hyperparameter_search_reused": SKIP_HP_SEARCH,
        "best_hyperparameters": best_hp.values,
        "cv_grouped_le2023": {
            "per_target_mean_over_folds": cv_mean.reset_index().rename(columns={"index": "target"}).to_dict(orient="records"),
            "global_arithmetic_mean": cv_global,
        },
        "holdout_2024": None,
    }
    with open(os.path.join(OUT_DIR, f"04_dnn_tuned_results_cutoff{CUTOFF}{RESULTS_SUFFIX}.json"), "w") as f:
        json.dump(partial_results, f, indent=2)
    print(f"[corte {CUTOFF}%] Guardado parcial (CV completa, holdout pendiente): "
          f"04_dnn_tuned_results_cutoff{CUTOFF}{RESULTS_SUFFIX}.json")

    print(f"\n[corte {CUTOFF}%][Eval] Modelo final: early stopping con validación interna <=2023, evaluación en holdout 2024")
    gss_final = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    fin_tr_idx, fin_val_idx = next(gss_final.split(train_pool, groups=train_pool["component_id"]))
    final_train_raw = train_pool.iloc[fin_tr_idx]
    final_val_raw = train_pool.iloc[fin_val_idx]
    X_fin_tr, X_fin_val, imputer_final, scaler_final = preprocess(final_train_raw, final_val_raw, verbose=True)
    X_holdout = transform_with(imputer_final, scaler_final, holdout_2024)
    y_fin_tr = final_train_raw[TARGETS]
    y_fin_val = final_val_raw[TARGETS]

    final_model = train_final_multitask(X_fin_tr, y_fin_tr, X_fin_val, y_fin_val)
    y_pred_holdout = predict_matrix(final_model, X_holdout)
    m_holdout = metrics_per_target(holdout_2024[TARGETS], y_pred_holdout)
    print(f"\n--- corte {CUTOFF}% -- DNN (tuned) holdout 2024 (n={len(holdout_2024)}) ---")
    print(m_holdout.to_string(index=False))
    holdout_global = {
        "R2": float(m_holdout["R2"].mean()), "MAE": float(m_holdout["MAE"].mean()), "RMSE": float(m_holdout["RMSE"].mean()),
    }
    print("Global (aritmético):", holdout_global)

    final_model.save(os.path.join(OUT_DIR, f"04_final_tuned_model_cutoff{CUTOFF}{RESULTS_SUFFIX}.keras"))
    import joblib
    joblib.dump(imputer_final, os.path.join(OUT_DIR, f"04_final_imputer_cutoff{CUTOFF}{RESULTS_SUFFIX}.joblib"))
    joblib.dump(scaler_final, os.path.join(OUT_DIR, f"04_final_scaler_cutoff{CUTOFF}{RESULTS_SUFFIX}.joblib"))

    partial_results["holdout_2024"] = {
        "per_target": m_holdout.to_dict(orient="records"),
        "global_arithmetic_mean": holdout_global,
    }
    with open(os.path.join(OUT_DIR, f"04_dnn_tuned_results_cutoff{CUTOFF}{RESULTS_SUFFIX}.json"), "w") as f:
        json.dump(partial_results, f, indent=2)
    print(f"\n[corte {CUTOFF}%] Guardado COMPLETO: 04_dnn_tuned_results_cutoff{CUTOFF}{RESULTS_SUFFIX}.json, "
          f"04_final_tuned_model_cutoff{CUTOFF}{RESULTS_SUFFIX}.keras, "
          f"04_final_imputer_cutoff{CUTOFF}{RESULTS_SUFFIX}.joblib, 04_final_scaler_cutoff{CUTOFF}{RESULTS_SUFFIX}.joblib")
    print(f"\n[corte {CUTOFF}%] Tiempo total del script: {(time.time() - t_start) / 60:.1f} min")
