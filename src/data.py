"""Loading and cleaning of the jfjelstul/worldcup CSV dataset.

The raw dataset (https://github.com/jfjelstul/worldcup) ships one CSV per
entity. For the match-outcome task the relevant tables are:

* ``matches``         - one row per match (scores, stage, teams, date)  [CORE]
* ``tournaments``     - one row per tournament (host, winner, #teams)
* ``teams``           - team metadata incl. confederation               [JOIN]
* ``team_appearances``- one row per team per match (long form of matches)

This module exposes thin, cached loaders plus a ``load_men_matches`` helper
that returns a cleaned, analysis-ready match table for the **men's** World Cup
(women's tournaments are excluded so that team-strength estimates are not
corrupted by mixing two different competitions that share ``team_id``s).
"""
from __future__ import annotations

import functools

import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Low-level loaders
# --------------------------------------------------------------------------- #
def load_csv(name: str) -> pd.DataFrame:
    """Load a raw CSV by stem (e.g. ``"matches"``)."""
    path = config.RAW_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the data-download step first "
            f"(see README / scripts)."
        )
    return pd.read_csv(path)


@functools.lru_cache(maxsize=None)
def _cached_csv(name: str) -> pd.DataFrame:
    return load_csv(name)


def load_teams() -> pd.DataFrame:
    """Team metadata with a tidy confederation code per team."""
    teams = _cached_csv("teams").copy()
    return teams[
        ["team_id", "team_name", "team_code", "confederation_id",
         "confederation_name", "confederation_code", "region_name"]
    ]


def load_tournaments() -> pd.DataFrame:
    return _cached_csv("tournaments").copy()


# --------------------------------------------------------------------------- #
# Cleaned match table
# --------------------------------------------------------------------------- #
def load_men_matches(drop_replays: bool = True) -> pd.DataFrame:
    """Return a cleaned, chronologically sorted men's World Cup match table.

    Parameters
    ----------
    drop_replays:
        If True, drop the handful of historical replayed matches' *original*
        (replayed) leg, keeping only the decisive replay, to avoid counting a
        single fixture twice. (There are only 4 such matches in the dataset.)

    Notes
    -----
    * Only men's tournaments are kept.
    * Goal columns are integers and contain no missing values in this dataset.
    * A ``year`` column is added for convenient temporal splitting.
    * The ``home_team`` / ``away_team`` labels only reflect *listing order*
      (World Cup venues are neutral). Downstream code treats matches as neutral
      and randomizes orientation, so this label must not be used as a feature.
    """
    m = _cached_csv("matches").copy()

    # Keep men's World Cup only.
    m = m[m["tournament_name"].str.contains("Men's", regex=False)].copy()

    if drop_replays:
        # 'replay' == 1 marks the decisive rematch; 'replayed' == 1 marks the
        # original drawn match that was annulled and replayed. Drop the latter.
        m = m[m["replayed"] == 0].copy()

    m["match_date"] = pd.to_datetime(m["match_date"])
    m["year"] = m["match_date"].dt.year

    # Sort chronologically; use match_id as a stable tiebreaker for same-day
    # matches so feature generation is deterministic.
    m = m.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    keep = [
        "tournament_id", "tournament_name", "year", "match_id", "match_name",
        "stage_name", "group_name", "group_stage", "knockout_stage",
        "match_date", "stadium_name", "city_name", "country_name",
        "home_team_id", "home_team_name", "home_team_code",
        "away_team_id", "away_team_name", "away_team_code",
        "home_team_score", "away_team_score",
        "extra_time", "penalty_shootout", "result",
    ]
    return m[keep].reset_index(drop=True)


def host_country_map() -> dict[int, set[str]]:
    """Map tournament ``year`` -> set of host country names.

    Handles multi-host tournaments (e.g. 2002 "Korea, Japan") by splitting on
    commas. Useful for building the host-country indicator feature.
    """
    t = load_tournaments()
    out: dict[int, set[str]] = {}
    for _, row in t.iterrows():
        hosts = {h.strip() for h in str(row["host_country"]).split(",")}
        out[int(row["year"])] = hosts
    return out


def data_summary() -> pd.DataFrame:
    """Quick one-row-per-file summary of all raw CSVs (shape only).

    Handy for the exploration notebook / data-quality report.
    """
    rows = []
    for path in sorted(config.RAW_DIR.glob("*.csv")):
        df = pd.read_csv(path)
        rows.append(
            {
                "file": path.name,
                "rows": len(df),
                "cols": df.shape[1],
                "n_missing": int(df.isna().sum().sum()),
            }
        )
    return pd.DataFrame(rows)
