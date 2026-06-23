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


def recency_weights(years, halflife_years: float | None,
                    reference_year: float | None = None) -> np.ndarray | None:
    """Exponential-decay sample weights based on match year.

    A match ``halflife_years`` older than ``reference_year`` (default: the most
    recent year in the data) gets half the weight of a current match. Returns
    ``None`` when ``halflife_years`` is ``None`` (i.e. equal weighting).
    """
    if halflife_years is None:
        return None
    years = np.asarray(years, dtype=float)
    ref = float(years.max()) if reference_year is None else float(reference_year)
    age = ref - years
    return np.power(0.5, age / float(halflife_years))


def fit_model(model, train_df: pd.DataFrame,
              recency_halflife: float | None = "default"):
    """Fit a model on a modeling dataframe (selecting features + target).

    Parameters
    ----------
    recency_halflife:
        * ``"default"`` (the sentinel) → use ``config.RECENCY_HALFLIFE_YEARS``.
        * a number → exponential half-life in years for sample weighting.
        * ``None`` → equal weights (no recency weighting).

    Sample weights are passed only to estimators that support them. For a
    scikit-learn ``Pipeline`` the weight is routed to the final ``clf`` step.
    """
    X = features.get_feature_matrix(train_df)
    y = train_df["target"].to_numpy()

    if recency_halflife == "default":
        recency_halflife = config.RECENCY_HALFLIFE_YEARS
    weights = recency_weights(train_df["year"].to_numpy(), recency_halflife)

    if weights is None or not _supports_sample_weight(model):
        model.fit(X, y)
    elif isinstance(model, Pipeline):
        model.fit(X, y, clf__sample_weight=weights)
    else:
        model.fit(X, y, sample_weight=weights)
    return model


def _supports_sample_weight(model) -> bool:
    """Whether ``model`` (or a Pipeline's final step) accepts sample_weight."""
    import inspect

    est = model.steps[-1][1] if isinstance(model, Pipeline) else model
    try:
        return "sample_weight" in inspect.signature(est.fit).parameters
    except (ValueError, TypeError):
        return False


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
