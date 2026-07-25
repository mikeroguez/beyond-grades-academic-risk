"""
Generate Figure 5 from the final DNN training protocol.

The original 08_dnn_hp_search.py did not persist Keras' epoch-wise History
object for the final model. This script therefore re-runs only the final
training step, using the already selected hyperparameters, the same <=2023
training pool, the same grouped internal validation split, and the same random
seed. It does not run the 40-trial Bayesian search, cross-validation, SHAP, or
clustering, and it does not overwrite the saved model or metrics.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
SOURCE_SCRIPT = HERE / "08_dnn_hp_search.py"


class FixedHyperParameters:
    def __init__(self, values: dict):
        self.values = values

    def Int(self, name, *args, **kwargs):
        return self.values[name]

    def Choice(self, name, *args, **kwargs):
        return self.values[name]

    def Float(self, name, *args, **kwargs):
        return self.values[name]


def load_dnn_module():
    spec = importlib.util.spec_from_file_location("dnn_hp_search_v36", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plot_history(history_df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(8.5, 4.8), dpi=300)
    plt.plot(history_df["epoch"], history_df["loss"], label="Training loss", color="#1f77b4", linewidth=2.2)
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation loss", color="#d62728", linewidth=2.2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.22, linewidth=0.8)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def main() -> int:
    best_hp_path = OUT_DIR / "08_best_hyperparameters.json"
    if not best_hp_path.exists():
        raise FileNotFoundError(f"Missing best hyperparameters: {best_hp_path}")

    with best_hp_path.open(encoding="utf-8") as f:
        best_hp_values = json.load(f)

    mod = load_dnn_module()
    tf.keras.utils.set_random_seed(mod.SEED)

    train_pool = mod.train_pool
    holdout_2024 = mod.holdout_2024

    X_tr_full, _, _, _ = mod.preprocess(train_pool, holdout_2024)
    gss_final = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=mod.SEED)
    fin_tr_idx, fin_val_idx = next(gss_final.split(train_pool, groups=train_pool["component_id"]))
    X_fin_tr = X_tr_full[fin_tr_idx]
    X_fin_val = X_tr_full[fin_val_idx]
    y_fin_tr = train_pool.iloc[fin_tr_idx][mod.TARGETS]
    y_fin_val = train_pool.iloc[fin_val_idx][mod.TARGETS]

    model = mod.build_model(FixedHyperParameters(best_hp_values))
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=mod.FINAL_PATIENCE, restore_best_weights=True, verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=0
        ),
    ]
    history = model.fit(
        X_fin_tr,
        mod.to_target_dict(y_fin_tr),
        validation_data=(X_fin_val, mod.to_target_dict(y_fin_val)),
        epochs=mod.FINAL_EPOCHS,
        batch_size=64,
        verbose=0,
        callbacks=callbacks,
    )

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))

    out_csv = OUT_DIR / "08_final_training_history.csv"
    out_png = OUT_DIR / "08_final_learning_curve.png"
    out_json = OUT_DIR / "08_final_learning_curve_source.json"
    history_df.to_csv(out_csv, index=False)
    plot_history(history_df, out_png)
    out_json.write_text(
        json.dumps(
            {
                "source_script": str(SOURCE_SCRIPT),
                "source_hyperparameters": str(best_hp_path),
                "curve_type": "final_training_rerun",
                "seed": mod.SEED,
                "final_epochs_configured": mod.FINAL_EPOCHS,
                "final_patience": mod.FINAL_PATIENCE,
                "epochs_observed": int(len(history_df)),
                "train_rows": int(len(fin_tr_idx)),
                "validation_rows": int(len(fin_val_idx)),
                "validation_split": "GroupShuffleSplit(test_size=0.15, random_state=42) on component_id within train_le2023",
                "outputs_not_overwritten": [
                    "08_dnn_tuned_results.json",
                    "08_final_tuned_model.keras",
                    "08_final_imputer.joblib",
                    "08_final_scaler.joblib",
                ],
                "output_png": str(out_png),
                "output_csv": str(out_csv),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {out_png}")
    print(f"Saved {out_csv}")
    print(f"Epochs observed: {len(history_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
