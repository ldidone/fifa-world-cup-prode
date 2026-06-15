"""Central configuration: filesystem paths and global modeling constants.

Keeping these in one place makes the pipeline reproducible and easy to tweak
without hunting through the codebase.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 2026 fixture lives in raw (hand-built from the official FIFA schedule).
FIXTURE_2026_PATH = RAW_DIR / "fixtures_2026.csv"

# --------------------------------------------------------------------------- #
# Target encoding (from team_a's perspective)
# --------------------------------------------------------------------------- #
# We model the *regulation* (90' + extra time) goal result. Matches decided on
# penalties count as draws here, because in open play they ended level. This is
# the standard, well-defined target for a 1X2 / goal model.
OUTCOME_CLASSES = ["team_a_win", "draw", "team_b_win"]
OUTCOME_TO_INT = {c: i for i, c in enumerate(OUTCOME_CLASSES)}
INT_TO_OUTCOME = {i: c for c, i in OUTCOME_TO_INT.items()}

# --------------------------------------------------------------------------- #
# Elo configuration
# --------------------------------------------------------------------------- #
ELO_BASE = 1500.0          # rating assigned to a team's first ever appearance
ELO_K = 40.0               # base K-factor (World-Football-Elo style)
ELO_HOME_ADVANTAGE = 0.0   # neutral venues -> no generic home bonus
# Optional reversion toward the mean applied at the start of each tournament to
# account for squad turnover across 4-year gaps. 0.0 == fully persistent Elo.
ELO_TOURNAMENT_REVERSION = 0.0

# --------------------------------------------------------------------------- #
# Feature-engineering configuration
# --------------------------------------------------------------------------- #
FORM_WINDOW = 5            # number of most-recent matches used for "recent form"

# Reproducibility
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
# Tournaments used as held-out test sets in the time-based backtest. They are
# the most recent men's World Cups; each is predicted using only prior data.
BACKTEST_YEARS = [2010, 2014, 2018, 2022]
