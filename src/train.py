"""Model factory and training helpers for the 3-class outcome task.

Models range from trivial baselines to gradient boosting. All classifiers
expose ``predict_proba`` so they can be scored with log loss / Brier and used
to produce calibrated-ish outcome probabilities for 2026.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, features

# XGBoost / LightGBM are optional: they require the OpenMP runtime
# (``libomp``) which is not present on every machine. When unavailable we fall
# back to scikit-learn's HistGradientBoostingClassifier, a strong gradient
# booster with no external system dependency.
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:  # pragma: no cover
    _HAS_LGBM = False


def build_models() -> dict[str, object]:
    """Return a dict {name: estimator} of all candidate models.

    The dict is ordered from simplest to most complex so that reports read
    naturally (baseline first).
    """
    models: dict[str, object] = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "prior_baseline": DummyClassifier(strategy="stratified",
                                          random_state=config.RANDOM_SEED),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=15,
            random_state=config.RANDOM_SEED, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300,
            l2_regularization=1.0, min_samples_leaf=20,
            random_state=config.RANDOM_SEED),
    }
    if _HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="multi:softprob", num_class=3,
            random_state=config.RANDOM_SEED, n_jobs=-1, eval_metric="mlogloss",
        )
    if _HAS_LGBM:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=config.RANDOM_SEED, n_jobs=-1, verbose=-1)
    return models


def fit_model(model, train_df: pd.DataFrame):
    """Fit a model on a modeling dataframe (selecting features + target)."""
    X = features.get_feature_matrix(train_df)
    y = train_df["target"].to_numpy()
    model.fit(X, y)
    return model


def predict_proba(model, df: pd.DataFrame) -> np.ndarray:
    """Return an (n, 3) probability matrix aligned to OUTCOME_CLASSES order.

    Some sklearn models drop classes not present in training; we re-expand to
    the full 3-class space so downstream metrics are always well-defined.
    """
    X = features.get_feature_matrix(df)
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", range(proba.shape[1])))
    full = np.zeros((len(df), 3))
    for j, cls in enumerate(classes):
        full[:, int(cls)] = proba[:, j]
    # renormalize defensively
    row_sums = full.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return full / row_sums
