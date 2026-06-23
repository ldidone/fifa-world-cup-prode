"""Metrics and time-based (tournament) validation.

Why not a random split?
-----------------------
Match data is time-ordered and our features (Elo, cumulative stats) are built
from the past. A random split would let the model "peek" at future tournaments
through the shared rating state and inflate scores. The honest evaluation is a
*temporal backtest*: to predict tournament Y, train only on matches before Y.

Metric guidance
---------------
* **Log loss** is the primary metric: it rewards well-calibrated probabilities,
  which is exactly what we want for a 3-way outcome with an irreducible draw.
* **Brier score** (multiclass) is a complementary proper scoring rule.
* **Accuracy** is intuitive but misleading here (draws are hard and a model can
  look "good" by never predicting them).
* **Macro-F1** checks that the minority *draw* class is not ignored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss)

from . import config, train


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean multiclass Brier score (sum of squared errors over classes)."""
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    y_pred = proba.argmax(axis=1)
    labels = [0, 1, 2]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels,
                             average="macro", zero_division=0),
        "log_loss": log_loss(y_true, proba, labels=labels),
        "brier": multiclass_brier(y_true, proba),
        "n": int(len(y_true)),
    }


def evaluate_against_actual(preds: pd.DataFrame,
                            results: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Score pre-tournament predictions against actual played results.

    Both frames are joined on ``match_number``. ``preds`` must contain
    ``p_team_a_win`` / ``p_draw`` / ``p_team_b_win`` in the same team_a/team_b
    orientation as ``results`` (which holds ``score_a`` / ``score_b``).

    Returns (metrics dict, per-match detail dataframe). Only matches present in
    both frames (i.e. already played) are scored.
    """
    merged = preds.merge(
        results[["match_number", "score_a", "score_b"]],
        on="match_number", how="inner",
    )
    if merged.empty:
        return {}, merged

    actual = np.where(merged["score_a"] > merged["score_b"], 0,
                      np.where(merged["score_a"] == merged["score_b"], 1, 2))
    proba = merged[["p_team_a_win", "p_draw", "p_team_b_win"]].to_numpy()

    merged = merged.copy()
    merged["actual_outcome"] = [config.INT_TO_OUTCOME[i] for i in actual]
    merged["predicted_outcome"] = [config.INT_TO_OUTCOME[i] for i in proba.argmax(1)]
    merged["correct"] = merged["actual_outcome"] == merged["predicted_outcome"]
    merged["p_assigned_to_actual"] = proba[np.arange(len(actual)), actual].round(4)

    metrics = compute_metrics(actual, proba)
    return metrics, merged


def confusion(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, proba.argmax(axis=1), labels=[0, 1, 2])
    return pd.DataFrame(cm, index=[f"true_{c}" for c in config.OUTCOME_CLASSES],
                        columns=[f"pred_{c}" for c in config.OUTCOME_CLASSES])


# --------------------------------------------------------------------------- #
# Temporal backtest
# --------------------------------------------------------------------------- #
def temporal_backtest(
    df: pd.DataFrame,
    model_factory,
    test_years: list[int] | None = None,
    min_train_years: int = 4,
    recency_halflife: float | None = "default",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward validation by tournament year.

    For each year in ``test_years`` a *fresh* model is trained on all matches
    from strictly-earlier years and evaluated on that year's matches.

    Parameters
    ----------
    df:
        Full modeling dataset (from :func:`features.build_modeling_dataset`).
    model_factory:
        Zero-arg callable returning a *new, unfitted* estimator.
    test_years:
        Tournament years to evaluate (default: ``config.BACKTEST_YEARS``).
    min_train_years:
        Minimum number of distinct prior tournaments required to evaluate a
        test year (skips early years with too little history).

    Returns
    -------
    (per_year, overall) : two DataFrames of metrics.
    """
    if test_years is None:
        test_years = config.BACKTEST_YEARS

    all_years = sorted(df["year"].unique())
    rows = []
    pooled_true: list[int] = []
    pooled_proba: list[np.ndarray] = []

    for ty in test_years:
        prior_years = [y for y in all_years if y < ty]
        if len(prior_years) < min_train_years:
            continue
        train_df = df[df["year"] < ty]
        test_df = df[df["year"] == ty]
        if len(test_df) == 0:
            continue

        model = train.fit_model(model_factory(), train_df,
                                recency_halflife=recency_halflife)
        proba = train.predict_proba(model, test_df)
        y_true = test_df["target"].to_numpy()

        metrics = compute_metrics(y_true, proba)
        metrics = {"year": ty, **metrics}
        rows.append(metrics)
        pooled_true.append(y_true)
        pooled_proba.append(proba)

    per_year = pd.DataFrame(rows)
    if pooled_true:
        yt = np.concatenate(pooled_true)
        pr = np.concatenate(pooled_proba)
        overall = pd.DataFrame([{"scope": "pooled", **compute_metrics(yt, pr)}])
    else:
        overall = pd.DataFrame()
    return per_year, overall


def compare_models(
    df: pd.DataFrame, test_years: list[int] | None = None,
    recency_halflife: float | None = "default",
) -> pd.DataFrame:
    """Run the temporal backtest for every model and return pooled metrics."""
    results = []
    for name, _ in train.build_models().items():
        # rebuild a fresh estimator each fold via a factory closure
        factory = (lambda n=name: train.build_models()[n])
        _, overall = temporal_backtest(df, factory, test_years=test_years,
                                       recency_halflife=recency_halflife)
        if not overall.empty:
            row = overall.iloc[0].to_dict()
            row["model"] = name
            results.append(row)
    cols = ["model", "n", "accuracy", "macro_f1", "log_loss", "brier"]
    out = pd.DataFrame(results)
    return out[cols].sort_values("log_loss").reset_index(drop=True)
