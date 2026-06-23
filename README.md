# FIFA World Cup 2026 Match Outcome Prediction

Machine learning project that predicts the result (win / draw / loss) and score
of every match in the FIFA World Cup 2026. It is trained on historical World Cup
data (1930–2022) and augmented with an external **World Football Elo** history
(eloratings.net) that tracks team strength across *all* international football.

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
| `data/processed/predictions_2026_updated.csv` | Predictions for remaining fixtures with Elo refreshed by played results |
| `data/processed/tournament_simulation_2026.csv` | Each team's Monte-Carlo advancement & title probability |
| `reports/model_comparison.csv` | Temporal backtest metrics for all models |
| `models/*.joblib` | Serialised best classifier + Poisson goal model |

To validate against the matches already played in 2026:

```bash
python scripts/evaluate_2026.py   # scores predictions vs data/raw/results_2026.csv
```

## Project structure

```
├── data/
│   ├── raw/              # jfjelstul/worldcup CSVs + 2026 fixture + eloratings.csv
│   └── processed/        # modeling dataset, predictions, simulation results
├── notebooks/
│   ├── 01_data_exploration_eda.ipynb      # data inspection & EDA
│   ├── 02_modeling_and_predictions.ipynb  # models, validation, 2026 forecast (final)
│   └── 03_modeling_with_xgboost.ipynb    # extended: XGBoost deep dive & hyperparam sweep
├── src/                  # reusable Python package
│   ├── config.py         # paths, constants, Elo hyperparameters
│   ├── data.py           # data loading & cleaning
│   ├── elo.py            # chronological Elo rating engine
│   ├── external_elo.py   # external World Football Elo (leak-free as-of lookups)
│   ├── features.py       # leakage-free feature engineering & modeling dataset
│   ├── train.py          # model factory (baselines → gradient boosting)
│   ├── evaluate.py       # metrics + temporal (tournament-year) backtest
│   ├── poisson.py        # Poisson goal model for score prediction
│   └── predict.py        # 2026 fixture prediction & tournament simulation
├── scripts/
│   ├── build_fixture_2026.py   # official 48-team draw → fixture CSV
│   ├── run_pipeline.py         # headless end-to-end pipeline
│   ├── evaluate_2026.py        # test predictions vs actual 2026 results
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

**External Elo history** (`data/raw/eloratings.csv`, from
[eloratings.net](https://www.eloratings.net)) provides per-team Elo snapshots
updated across all international football (qualifiers, friendlies, continental
cups), with a clean snapshot dated **2025-12-13** — right before the 2026
tournament. Unlike the bundled WC-only Elo, it reflects current form and is read
**strictly as-of the date before each match**, so it never leaks (see
`src/external_elo.py`).

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

### Features (14, all leakage-free)

| Feature | Type | Description |
|---|---|---|
| `elo_diff` | strength | WC-only Elo rating difference (team_a − team_b) |
| `ext_elo_diff` | strength | External World Football Elo difference, as-of the day before the match (team_a − team_b) |
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

### External Elo (eloratings.net)

The bundled `elo_diff` is updated *only* at World Cup matches, so between
tournaments it goes stale and never sees the qualifiers, friendlies and
continental cups that actually drive team strength. We therefore add
`ext_elo_diff` from an external World Football Elo history. For each historical
match we look up each team's rating from the snapshot **strictly before** the
match date (leak-free); for 2026 we use the latest pre-tournament snapshot
(2025-12-13). Coverage is 100% for the backtest years (2010–2022) and the 2026
field. Adding it improves **every** model in the temporal backtest:

| Model | Log loss (WC-only Elo) | Log loss (+ external Elo) | Accuracy | Macro F1 |
|---|---|---|---|---|
| Random Forest | 0.998 | **0.991** | 54.3% → 55.9% | 0.411 → 0.432 |
| XGBoost | 1.117 | 1.096 | 47.3% → 51.2% | 0.421 → 0.461 |
| LightGBM | 1.136 | 1.143 | 47.7% → 50.0% | 0.426 → 0.445 |

### Recency weighting

Older World Cups are less representative of the modern game, so training
applies an **exponential time decay** to sample weights: a match `H` years
older than the most recent one counts half as much (`config.RECENCY_HALFLIFE_YEARS`,
default 16). The temporal backtest shows this is a small but real improvement.
Set it to `None` to disable. Note this is a *soft*
version of "train on recent World Cups" — strictly dropping old data shrinks an
already-small dataset and does not help.

### Models

All 7 candidate models are evaluated via temporal backtest (256 test matches
across the 2010–2022 World Cups):

| Model | Log loss | Accuracy | Macro F1 | Brier | Notes |
|---|---|---|---|---|---|
| **Random Forest** | **0.991** | **55.9%** | **0.432** | **0.588** | **best by log loss — selected** |
| Logistic Regression | 1.020 | 53.1% | 0.431 | 0.596 | linear, well-calibrated |
| XGBoost | 1.096 | 51.2% | 0.461 | 0.640 | gradient boosting (requires `libomp` on macOS) |
| LightGBM | 1.143 | 50.0% | 0.445 | 0.663 | gradient boosting (requires `libomp` on macOS) |
| HistGradientBoosting | 1.189 | 48.0% | 0.416 | 0.677 | sklearn built-in, no `libomp` needed |
| Prior baseline | 21.68 | 39.8% | 0.361 | 1.203 | predicts by training-set class proportions |
| Majority baseline | 21.82 | 39.5% | 0.189 | 1.211 | always predicts the mode class |
| Poisson goal model | 1.013 | 52.7% | 0.396 | 0.604 | predicts exact scores via Poisson regression (uses goal rates, not Elo) |

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

Title contenders (Monte-Carlo simulation, 3000 runs, model with external Elo):

| Team | P(Champion) | P(Advance) |
|---|---|---|
| Brazil | ~14.4% | 92.9% |
| Argentina | ~13.4% | 93.6% |
| France | ~13.0% | 96.3% |
| Spain | ~9.2% | 89.1% |
| England | ~6.7% | 91.6% |
| Germany | ~6.6% | 88.6% |
| Netherlands | ~6.1% | 93.4% |
| Portugal | ~5.2% | 90.1% |

### Validation against the live 2026 tournament

`scripts/evaluate_2026.py` scores the pre-tournament predictions against the
matches actually played (stored in `data/raw/results_2026.csv`; append rows as
more are played). Through the first 10 group games (11–14 June):

| Metric | Model | Class-prior baseline |
|---|---|---|
| Accuracy | 4/10 (40%) | 1/10 (10%) |
| Log loss | ~1.13 | ~1.18 |

**What the model got right:** Mexico, South Korea, Scotland over Haiti, and
Germany (7-1) — the clear favourites by Elo landed.

**What it missed:** four of the six misses were **draws** (Canada–Bosnia,
Qatar–Switzerland, Brazil–Morocco, Netherlands–Japan), plus two upsets (USA 4-1
Paraguay and Australia 2-0 Türkiye). This is exactly the documented weakness: a
results-only model rarely predicts a draw as the single most likely outcome, and
the opening round was unusually draw-heavy (4 of 10 games).

**On the "train on recent World Cups" hypothesis:** recency weighting was added
and tested; it helps only *marginally* and does not change the 2026-actuals
accuracy. The bigger win came from fixing **stale strength estimates**: adding
the external World Football Elo (`ext_elo_diff`) — which is current as of
December 2025 and reflects qualifiers/friendlies — improves every model in the
backtest and modestly improves the 2026-actuals log loss (1.192 → 1.180). The
remaining dominant limitation is draw prediction, not the age of the training
tournaments (the final model already trains on all years through 2022).

As matches are played, `evaluate_2026.py` also refreshes Elo with the results
and re-predicts the remaining fixtures (`predictions_2026_updated.csv`).

## Limitations & future improvements

### Known limitations

1. **Limited match history.** The cumulative form/H2H features still update only
   at WC matches (every 4 years). This is partly mitigated by `ext_elo_diff`
   (external Elo updated across all international football), but the per-team
   goal/form/H2H features remain WC-only.
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
  would let the *goal/form/H2H* features (not just Elo) update continuously,
  the same way `ext_elo_diff` already does for strength.
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

# Validate against actual 2026 results played so far
python scripts/evaluate_2026.py
```

## License

Code: MIT. Data: [jfjelstul/worldcup](https://github.com/jfjelstul/worldcup)
is CC-BY-4.0.
