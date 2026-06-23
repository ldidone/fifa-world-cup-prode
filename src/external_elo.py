"""External World Football Elo ratings (eloratings.net) integration.

The bundled WC-only Elo (``src/elo.py``) is updated *only* at World Cup
matches, so between tournaments it goes stale and it never sees qualifiers,
friendlies or continental cups. This module loads an external Elo history
(``data/raw/eloratings.csv``) that is updated across all international football,
and exposes **leakage-free as-of lookups**: the rating of a team strictly
*before* a given date.

CSV schema: ``date, team, rating, change`` — one row per team per rating
snapshot (several per year). Team names use non-breaking spaces and a few
spellings that differ from our dataset, both handled here.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from . import config

# Our team name (dataset / 2026 fixture spelling) -> external Elo spelling.
# Applied after normalising non-breaking spaces, so only genuine differences
# need listing here.
_NAME_ALIASES = {
    "Czech Republic": "Czechia",
    "DR Congo": "Democratic Republic of Congo",
    "Zaire": "Democratic Republic of Congo",
    "Republic of Ireland": "Ireland",
    "Curacao": "Curaçao",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Turkiye": "Turkey",
}


def _normalize(name: str) -> str:
    """Normalise a team name to the external-Elo key space."""
    if name is None:
        return ""
    name = str(name).replace("\xa0", " ").strip()
    return _NAME_ALIASES.get(name, name)


class ExternalElo:
    """As-of accessor over the external Elo time series."""

    def __init__(self, df: pd.DataFrame):
        # Per-team sorted (dates, ratings) for fast as-of lookup.
        self._dates: dict[str, np.ndarray] = {}
        self._ratings: dict[str, np.ndarray] = {}
        for team, g in df.sort_values("date").groupby("team"):
            self._dates[team] = g["date"].to_numpy(dtype="datetime64[ns]")
            self._ratings[team] = g["rating"].to_numpy(dtype=float)
        self._global_mean = float(df["rating"].mean())

    # -- construction ----------------------------------------------------- #
    @classmethod
    def from_csv(cls, path=None) -> "ExternalElo":
        path = path or config.ELO_RATINGS_PATH
        raw = pd.read_csv(path)
        raw["team"] = raw["team"].map(lambda s: str(s).replace("\xa0", " ").strip())
        raw["date"] = pd.to_datetime(raw["date"], format="mixed", errors="coerce")
        raw = raw.dropna(subset=["date"])
        return cls(raw)

    # -- queries ---------------------------------------------------------- #
    def rating_asof(self, team: str, date, strict: bool = True) -> float | None:
        """Most recent rating for ``team`` before ``date``.

        ``strict`` uses ``< date`` (no same-day leakage); set False for ``<=``.
        Returns ``None`` when no prior rating exists (unknown team / too early).
        """
        key = _normalize(team)
        dates = self._dates.get(key)
        if dates is None:
            return None
        d = np.datetime64(pd.Timestamp(date))
        idx = np.searchsorted(dates, d, side="left" if strict else "right")
        if idx == 0:
            return None
        return float(self._ratings[key][idx - 1])

    def latest_rating(self, team: str, cutoff=None) -> float | None:
        """Latest rating at or before ``cutoff`` (default: end of series)."""
        if cutoff is None:
            key = _normalize(team)
            r = self._ratings.get(key)
            return float(r[-1]) if r is not None and len(r) else None
        return self.rating_asof(team, cutoff, strict=False)

    def has(self, team: str) -> bool:
        return _normalize(team) in self._dates


@functools.lru_cache(maxsize=1)
def load_external_elo() -> ExternalElo:
    """Cached singleton loader."""
    return ExternalElo.from_csv()
