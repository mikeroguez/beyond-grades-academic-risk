"""
WP1 — Paso 4: Selección de variables ajustada SOLO con la partición de
entrenamiento (Reviewer 1 #18-19).

El script original (`code/v3.6/03-feature-selection/04_seleccion_pearson_lassocv.py`)
tenía dos problemas de orden, documentados en AUDITORIA-LEAKAGE.md:
(a) el umbral óptimo de
Pearson se buscaba con validación cruzada sobre TODO el dataset (`X_clean`,
`y` completos, no `X_train`/`y_train`), y (b) el `MinMaxScaler`/`RobustScaler`
de `02_escala_y_renombra.py` se ajustaba sobre el dataset completo antes de
cualquier split.

Este script re-ejecuta un chequeo de selección de variables análogo
(LassoCV por target), pero ajustado ÚNICAMENTE con las filas del pool de
entrenamiento temporal (`split_bucket == train_le2023`), sobre el conjunto
de 18 features ya limpias de fuga (`out/clean_dataset.csv`,
`out/feature_lists.json`). No se usa NUNCA el holdout 2024 ni las filas de
validación de los folds internos para elegir qué variables entran.

Dado que el pool de candidatos ya se redujo de 26 a 18 por el filtro
anti-fuga (paso 3), este chequeo es más una auditoría de "¿sobra alguna
variable sin señal?" que una selección agresiva -- no se usa para podar el
set principal de 18 (ver METODOLOGIA-V3.md, decisión documentada), pero deja
registro de qué variables tienen coeficiente Lasso distinto de cero para al
menos un target, en caso de que Miguel quiera una versión más compacta.

Output: out/04_feature_selection_train_only.json
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")

df = pd.read_csv(os.path.join(OUT_DIR, "clean_dataset.csv"))
with open(os.path.join(OUT_DIR, "feature_lists.json")) as f:
    feat_lists = json.load(f)

CLEAN_FEATURES = feat_lists["clean_features"]
TARGETS = feat_lists["targets"]

train = df[df["split_bucket"] == "train_le2023"].copy()
print(f"Pool de entrenamiento (≤2023): {train.shape[0]} filas, {train['component_id'].nunique()} componentes")

groups = train["component_id"]

results = {}
for target in TARGETS:
    X = train[CLEAN_FEATURES]
    y = train[target]

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=CLEAN_FEATURES, index=X.index)

    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=CLEAN_FEATURES, index=X.index)

    gkf = GroupKFold(n_splits=5)
    lasso = LassoCV(cv=list(gkf.split(X_scaled, y, groups=groups)), random_state=42, n_jobs=-1, max_iter=10000)
    lasso.fit(X_scaled, y)

    coefs = pd.Series(lasso.coef_, index=CLEAN_FEATURES)
    nonzero = coefs[coefs.abs() > 1e-6].sort_values(key=np.abs, ascending=False)

    results[target] = {
        "alpha": float(lasso.alpha_),
        "n_features_nonzero": int((coefs.abs() > 1e-6).sum()),
        "nonzero_features": nonzero.to_dict(),
    }
    print(f"\n[{target}] alpha={lasso.alpha_:.5f} | features con coef!=0: {(coefs.abs()>1e-6).sum()}/{len(CLEAN_FEATURES)}")
    print(nonzero)

# Unión de features usadas por al menos un target -> candidato a set "compacto"
union_used = sorted(set().union(*[set(v["nonzero_features"].keys()) for v in results.values()]))
never_used = sorted(set(CLEAN_FEATURES) - set(union_used))
print(f"\nFeatures usadas (coef!=0) por AL MENOS un target: {len(union_used)}/{len(CLEAN_FEATURES)}")
print(f"Features con coef=0 en LOS 5 targets simultáneamente (candidatas a poda): {never_used}")

with open(os.path.join(OUT_DIR, "04_feature_selection_train_only.json"), "w") as f:
    json.dump({"per_target": results, "union_used": union_used, "never_used_any_target": never_used}, f, indent=2)
print(f"\nGuardado: {os.path.join(OUT_DIR, '04_feature_selection_train_only.json')}")
print("\nDecisión de diseño: se CONSERVAN las 18 features limpias para el modelo "
      "principal (ver METODOLOGIA-V3.md) -- este chequeo es informativo, no se usa "
      "para podar automáticamente, porque Lasso con L1 puede anular features "
      "correlacionadas de forma inestable entre folds; una poda automática aquí "
      "arriesgaría eliminar señal real por colinealidad, no por falta de relación "
      "con el target.")
