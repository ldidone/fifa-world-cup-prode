"""End-to-end pipeline: validate models, train, predict 2026, export artifacts.

Run:  python scripts/run_pipeline.py [--model xgboost] [--sims 2000]

Outputs (under reports/ and data/processed/):
  * model_comparison.csv         - temporal backtest metrics per model
  * backtest_<model>.csv         - per-tournament metrics for the chosen model
  * confusion_<model>.csv        - pooled confusion matrix
  * predictions_2026_groups.csv  - per-match group-stage predictions
  * tournament_simulation_2026.csv - per-team advancement / title odds
  * modeling_dataset.csv         - the full leakage-free modeling table
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, evaluate, features, predict, train
from scripts.build_fixture_2026 import GROUPS, HOSTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="auto",
                    help="final model name, or 'auto' to pick the best by "
                         "backtest log loss (see src.train.build_models)")
    ap.add_argument("--sims", type=int, default=2000,
                    help="Monte-Carlo simulations for tournament odds")
    args = ap.parse_args()

    np.random.seed(config.RANDOM_SEED)

    print("==> Building leakage-free modeling dataset ...")
    matches = data.load_men_matches()
    model_df = features.build_modeling_dataset(matches)
    model_df.to_csv(config.PROCESSED_DIR / "modeling_dataset.csv", index=False)
    print(f"    {len(model_df)} matches, {len(features.FEATURE_COLUMNS)} features")

    print("\n==> Temporal backtest: comparing all models ...")
    comparison = evaluate.compare_models(model_df)
    comparison.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    print(comparison.to_string(index=False))

    # 'auto' -> best non-baseline model by log loss (comparison is sorted).
    if args.model in train.build_models():
        chosen = args.model
    else:
        non_baseline = comparison[~comparison["model"].str.contains("baseline")]
        chosen = non_baseline.iloc[0]["model"]
    print(f"\n==> Detailed backtest for chosen model: {chosen}")
    per_year, overall = evaluate.temporal_backtest(
        model_df, lambda: train.build_models()[chosen])
    per_year.to_csv(config.REPORTS_DIR / f"backtest_{chosen}.csv", index=False)
    print(per_year.to_string(index=False))
    print("pooled:", overall.to_dict("records"))

    # pooled confusion matrix for the chosen model
    pooled_true, pooled_proba = [], []
    years = sorted(model_df["year"].unique())
    for ty in config.BACKTEST_YEARS:
        tr = model_df[model_df["year"] < ty]
        te = model_df[model_df["year"] == ty]
        if len(te) == 0 or len(tr) == 0:
            continue
        mdl = train.fit_model(train.build_models()[chosen], tr)
        pr = train.predict_proba(mdl, te)
        pooled_true.append(te["target"].to_numpy())
        pooled_proba.append(pr)
    if pooled_true:
        cm = evaluate.confusion(np.concatenate(pooled_true),
                                np.concatenate(pooled_proba))
        cm.to_csv(config.REPORTS_DIR / f"confusion_{chosen}.csv")
        print("\nConfusion matrix (pooled backtest):\n", cm)

    print("\n==> Fitting final model on ALL history ...")
    clf, pois, hist, clf_name = predict.fit_full(chosen)
    joblib.dump(clf, config.MODELS_DIR / f"outcome_{clf_name}.joblib")
    joblib.dump(pois, config.MODELS_DIR / "poisson_goal_model.joblib")
    print(f"    saved outcome model ({clf_name}) + Poisson goal model")

    print("\n==> Predicting 2026 group-stage matches ...")
    fixture = pd.read_csv(config.FIXTURE_2026_PATH)
    _, unmatched = predict.resolve_team_ids(
        sorted(set(fixture["team_a"]) | set(fixture["team_b"])))
    if unmatched:
        print(f"    NOTE: {len(unmatched)} teams have no men's WC history "
              f"(treated as debutants @Elo {int(config.ELO_BASE)}): {unmatched}")
    preds = predict.predict_matches(fixture, clf, pois, hist, HOSTS)
    cols = ["match_number", "group", "team_a", "team_b", "team_a_elo",
            "team_b_elo", "p_team_a_win", "p_draw", "p_team_b_win",
            "predicted_outcome_label", "predicted_score",
            "exp_goals_a", "exp_goals_b"]
    preds[cols].to_csv(config.PROCESSED_DIR / "predictions_2026_groups.csv",
                       index=False)
    print(preds[cols].head(10).to_string(index=False))

    print(f"\n==> Monte-Carlo tournament simulation ({args.sims} sims) ...")
    sim = predict.simulate_tournament(GROUPS, clf, pois, hist, HOSTS,
                                      n_sims=args.sims)
    sim.to_csv(config.PROCESSED_DIR / "tournament_simulation_2026.csv",
               index=False)
    print(sim.head(12).to_string(index=False))

    print("\nDone. Artifacts written to reports/ and data/processed/.")


if __name__ == "__main__":
    main()
