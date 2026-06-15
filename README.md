# FIFA World Cup 2026 Match Outcome Prediction

Machine learning project that predicts the result (win / draw / loss) and score
of every match in the FIFA World Cup 2026, trained exclusively on historical
World Cup data (1930–2022).

## Quick start

```bash
# 1. Create & activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (compare models, train, predict 2026, simulate)
python scripts/run_pipeline.py          # ~2 min on a laptop
```

Key outputs:

| File | Description |
|---|---|
| `data/processed/predictions_2026_groups.csv` | Per-match outcome probabilities, predicted winner, score |
| `data/processed/tournament_simulation_2026.csv` | Each team's Monte-Carlo advancement & title probability |
| `reports/model_comparison.csv` | Temporal backtest metrics for all models |
| `models/*.joblib` | Serialised best classifier + Poisson goal model |

## Project structure

```
├── data/
│   ├── raw/              # 27 CSVs from jfjelstul/worldcup + 2026 fixture
│   └── processed/        # modeling dataset, predictions, simulation results
├── notebooks/
│   ├── 01_data_exploration_eda.ipynb      # data inspection & EDA
│   ├── 02_modeling_and_predictions.ipynb  # models, validation, 2026 forecast (final)
│   └── 03_modeling_with_xgboost.ipynb    # extended: XGBoost deep dive & hyperparam sweep
├── src/                  # reusable Python package
│   ├── config.py         # paths, constants, Elo hyperparameters
│   ├── data.py           # data loading & cleaning
│   ├── elo.py            # chronological Elo rating engine
│   ├── features.py       # leakage-free feature engineering & modeling dataset
│   ├── train.py          # model factory (baselines → gradient boosting)
│   ├── evaluate.py       # metrics + temporal (tournament-year) backtest
│   ├── poisson.py        # Poisson goal model for score prediction
│   └── predict.py        # 2026 fixture prediction & tournament simulation
├── scripts/
│   ├── build_fixture_2026.py   # editable 48-team group draw → fixture CSV
│   ├── run_pipeline.py         # headless end-to-end pipeline
│   └── inspect_data.py         # one-off data inspection
├── models/               # serialised model artifacts (.joblib)
├── reports/              # CSV metrics + figures/
│   └── figures/          # PNGs from notebooks
└── requirements.txt
```

## Data source

All historical data comes from
[jfjelstul/worldcup](https://github.com/jfjelstul/worldcup)
(CC-BY-4.0). It contains 27 CSV files covering every FIFA World Cup from 1930
to 2022 (men's & women's). This project uses only the **men's** tournaments
(964 matches, 22 World Cups) to avoid mixing two separate competitions that
share team identifiers.

The 2026 fixture (`data/raw/fixtures_2026.csv`) is a representative 48-team
group draw that can be edited in `scripts/build_fixture_2026.py` to match the
official draw as it is finalised.

## Modeling approach

### Target

Three-class outcome from **team_a's** perspective:
`team_a_win` / `draw` / `team_b_win`.

Defined from the regulation+ET goal score. Penalty shoot-outs count as draws
(the correct, well-defined 1×2 result).

### Neutral-venue symmetrisation

World Cup "home"/"away" labels only reflect listing order — stronger teams are
systematically listed first (~55% listed-home win rate on neutral pitches). To
prevent this from leaking into the model, each match is emitted once with
`team_a`/`team_b` assigned by a seeded coin flip, and all features are
*antisymmetric differences* (team_a − team_b). This yields a balanced
~39/22/39% target distribution.

### Features (13, all leakage-free)

| Feature | Type | Description |
|---|---|---|
| `elo_diff` | strength | Elo rating difference (team_a − team_b) |
| `matches_played_diff` | experience | WC appearances difference |
| `win_rate_diff` | quality | historical win rate difference |
| `gf_avg_diff` | attack | goals-for per match difference |
| `ga_avg_diff` | defence | goals-against per match difference |
| `form_points_diff` | recency | recent-form points (last 5 matches) difference |
| `experience_diff` | experience | distinct WC tournaments played difference |
| `host_advantage` | context | +1 if team_a is host, −1 if team_b, else 0 |
| `h2h_played` | head-to-head | number of prior meetings |
| `h2h_a_win_rate` | head-to-head | team_a win rate in past H2H |
| `h2h_goal_diff_avg` | head-to-head | average goal difference in past H2H |
| `same_confederation` | context | 1 if both teams share a confederation |
| `knockout_stage` | context | 1 if knockout stage, 0 if group |

Every feature is computed by walking matches chronologically and recording
each team's state *before* the current match — no future information ever leaks.

### Models

All 7 candidate models are evaluated via temporal backtest (256 test matches
across the 2010–2022 World Cups):

| Model | Log loss | Accuracy | Macro F1 | Brier | Notes |
|---|---|---|---|---|---|
| **Random Forest** | **1.006** | **55.5%** | **0.417** | **0.598** | **best by log loss — selected** |
| Logistic Regression | 1.041 | 49.6% | 0.389 | 0.614 | linear, well-calibrated |
| XGBoost | 1.116 | 47.7% | 0.424 | 0.656 | gradient boosting (requires `libomp` on macOS) |
| LightGBM | 1.135 | 45.3% | 0.400 | 0.666 | gradient boosting (requires `libomp` on macOS) |
| HistGradientBoosting | 1.217 | 45.7% | 0.412 | 0.695 | sklearn built-in, no `libomp` needed |
| Prior baseline | 21.68 | 39.8% | 0.357 | 1.203 | predicts by training-set class proportions |
| Majority baseline | 22.25 | 38.3% | 0.185 | 1.234 | always predicts the mode class |
| Poisson goal model | 1.013 | 52.7% | 0.396 | 0.604 | predicts exact scores via Poisson regression |

On this small dataset (~960 matches), Random Forest outperforms all gradient
boosting methods. The boosters (XGBoost, LightGBM, HistGB) tend to overfit with
default hyperparameters; a detailed sensitivity analysis is available in
`notebooks/03_modeling_with_xgboost.ipynb`.

> **Note:** XGBoost and LightGBM require the OpenMP runtime (`libomp`) on macOS.
> Install it with `brew install libomp`. When unavailable, the pipeline
> automatically falls back to sklearn's `HistGradientBoostingClassifier`.

### Validation strategy

**Temporal backtest by tournament year.** For each of the last 4 men's World
Cups (2010, 2014, 2018, 2022), the model is trained on all prior matches and
evaluated on that tournament. This mirrors the real prediction task.

**Primary metric: log loss.** Accuracy is misleading for a 3-class problem
where draws (~22%) are inherently hard to predict as the *most likely* outcome.
Log loss rewards well-calibrated probabilities across all three classes.

### 2026 predictions (highlights)

Based on the [official FIFA draw](https://www.fifa.com/es/tournaments/mens/worldcup/canadamexicousa2026/articles/calendario-fixture-mundial-2026-partidos-fechas)
(5 December 2025, Washington D.C.).

Title contenders (Monte-Carlo simulation, 3000 runs):

| Team | Elo | P(Champion) |
|---|---|---|
| Brazil | 1724 | ~20.1% |
| France | 1775 | ~12.7% |
| Argentina | 1713 | ~12.2% |
| Germany | 1741 | ~8.5% |
| Netherlands | 1804 | ~8.0% |
| England | 1640 | ~6.1% |
| Spain | 1631 | ~5.7% |

## Limitations & future improvements

### Known limitations

1. **World Cup data only.** Elo ratings update only at WC matches (every 4
   years). Real team strength changes continuously through qualifiers,
   friendlies, and continental cups. This is the single largest accuracy gap.
2. **No player-level information.** Squad changes, injuries, and individual
   form are invisible.
3. **Defunct teams.** West Germany, USSR, Yugoslavia hold high Elo that their
   modern successors do not inherit (the dataset uses separate `team_id`s).
4. **48-team format is new.** 2026 uses 12 groups of 4 teams and a 32-team
   knockout — structurally different from any prior World Cup. The simulation
   bracket rules are an approximation.
5. **Draws are inherently hard.** No results-only model reliably picks a draw
   as the single most likely outcome; the best models assign ~20–25% draw
   probability, which is realistic.

### High-value improvements (designed to be addable)

The code is structured so that these can be plugged in incrementally:

* **International match data** (qualifiers, friendlies, continental cups) —
  doubles the dataset and makes Elo ratings continuously updated.
* **Official FIFA / Elo rankings** — much finer-grained strength estimates.
* **Squad-level features** — market value, average age, injury count.
* **Betting-market implied probabilities** — as calibration targets or features.
* **Dixon-Coles correction** — improves the Poisson model by adjusting for the
  over-frequency of low-scoring draws (0-0, 1-1).

## Reproducing the analysis

```bash
# Re-execute the notebooks (generates all figures & CSVs)
cd notebooks
jupyter nbconvert --execute --inplace 01_data_exploration_eda.ipynb
jupyter nbconvert --execute --inplace 02_modeling_and_predictions.ipynb
jupyter nbconvert --execute --inplace 03_modeling_with_xgboost.ipynb  # optional XGBoost deep dive

# Or run the headless pipeline
python scripts/run_pipeline.py --model random_forest --sims 5000
```

## License

Code: MIT. Data: [jfjelstul/worldcup](https://github.com/jfjelstul/worldcup)
is CC-BY-4.0.
