"""Test the model against the ACTUAL FIFA World Cup 2026 results so far.

This script answers two questions raised after comparing predictions with the
real tournament:

1. *Does recency-weighted training help?* (down-weighting very old World Cups).
   We compare the temporal backtest and the 2026-actuals score for the model
   trained with vs. without recency weighting.

2. *How good were the pre-tournament predictions?* We score the model's
   pre-tournament probabilities against the matches actually played (loaded
   from ``data/raw/results_2026.csv``).

It also demonstrates refreshing Elo with the played results to re-predict the
*remaining* fixtures.

Run:  python scripts/evaluate_2026.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, evaluate, features, poisson, predict, train
from scripts.build_fixture_2026 import HOSTS

MODEL = "random_forest"


def _predict_fixture(clf, pois, hist):
    fixture = pd.read_csv(config.FIXTURE_2026_PATH)
    return predict.predict_matches(fixture, clf, pois, hist, HOSTS)


def main() -> None:
    np.random.seed(config.RANDOM_SEED)
    matches = data.load_men_matches()
    model_df = features.build_modeling_dataset(matches)
    results = pd.read_csv(config.RESULTS_2026_PATH)
    print(f"Loaded {len(results)} played 2026 matches (through "
          f"{results['date'].max()}).\n")

    # ------------------------------------------------------------------ #
    # 1. Backtest: recency weighting OFF vs ON
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("TEMPORAL BACKTEST (2010-2022): effect of recency weighting")
    print("=" * 70)
    factory = lambda: train.build_models()[MODEL]
    _, off = evaluate.temporal_backtest(model_df, factory, recency_halflife=None)
    rows = [{"setting": "no recency (equal weights)", **off.iloc[0].to_dict()}]
    for hl in (8.0, 16.0, 24.0):
        _, ov = evaluate.temporal_backtest(model_df, factory, recency_halflife=hl)
        rows.append({"setting": f"recency half-life = {hl:g} yrs",
                     **ov.iloc[0].to_dict()})
    bt = pd.DataFrame(rows)[["setting", "accuracy", "macro_f1", "log_loss", "brier"]]
    print(bt.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 2. Score pre-tournament predictions vs ACTUAL results
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("PRE-TOURNAMENT PREDICTIONS vs ACTUAL 2026 RESULTS (played matches)")
    print("=" * 70)
    hist_pre = features.fit_history(matches)  # historical only (no leakage)
    pois = poisson.PoissonGoalModel().fit(model_df)

    summary = []
    detail_by_setting = {}
    for label, hl in [("no recency", None), ("recency 16y", 16.0)]:
        clf = train.fit_model(train.build_models()[MODEL], model_df,
                              recency_halflife=hl)
        preds = _predict_fixture(clf, pois, hist_pre)
        metrics, detail = evaluate.evaluate_against_actual(preds, results)
        metrics = {"setting": label, **metrics}
        summary.append(metrics)
        detail_by_setting[label] = detail
    print(pd.DataFrame(summary)[
        ["setting", "n", "accuracy", "macro_f1", "log_loss", "brier"]
    ].to_string(index=False))

    # Per-match detail for the recency model
    det = detail_by_setting["recency 16y"][
        ["team_a", "team_b", "p_team_a_win", "p_draw", "p_team_b_win",
         "score_a", "score_b", "actual_outcome", "predicted_outcome",
         "correct", "p_assigned_to_actual"]
    ]
    print("\nPer-match detail (recency 16y model):")
    print(det.to_string(index=False))

    # Naive bookmaker-free baseline: always predict the class proportions
    base_rate = model_df["target"].value_counts(normalize=True).reindex(
        [0, 1, 2]).to_numpy()
    actual = np.where(results["score_a"] > results["score_b"], 0,
                      np.where(results["score_a"] == results["score_b"], 1, 2))
    base_proba = np.tile(base_rate, (len(actual), 1))
    base_metrics = evaluate.compute_metrics(actual, base_proba)
    print(f"\nReference — class-prior baseline on the same matches: "
          f"acc={base_metrics['accuracy']:.3f}, "
          f"log_loss={base_metrics['log_loss']:.3f}")

    # ------------------------------------------------------------------ #
    # 3. Refresh Elo with played results -> predict REMAINING fixtures
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("UPDATED PREDICTIONS FOR UPCOMING FIXTURES (Elo refreshed w/ results)")
    print("=" * 70)
    clf = train.fit_model(train.build_models()[MODEL], model_df, recency_halflife=16.0)
    hist_live = predict.fit_history_with_played()
    preds_live = _predict_fixture(clf, pois, hist_live)
    played_ids = set(results["match_number"])
    upcoming = preds_live[~preds_live["match_number"].isin(played_ids)]
    out_cols = ["match_number", "group", "team_a", "team_b",
                "p_team_a_win", "p_draw", "p_team_b_win",
                "predicted_outcome_label", "predicted_score"]
    preds_live.to_csv(config.PROCESSED_DIR / "predictions_2026_updated.csv",
                      index=False)
    print(f"(saved full refreshed predictions to "
          f"data/processed/predictions_2026_updated.csv)\n")
    print(upcoming[out_cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
