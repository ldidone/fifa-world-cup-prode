"""A chronological Elo rating engine for international football.

Elo is the single most informative feature for football match prediction that
can be derived purely from historical results. Ratings are updated match by
match in date order, so the rating *before* a given match only depends on
earlier matches -> no data leakage by construction.

The update rule follows the "World Football Elo" family:

    expected_a = 1 / (1 + 10 ** ((R_b - R_a) / 400))
    R_a' = R_a + K * G * (S_a - expected_a)

where ``S_a`` is the actual score (1 win / 0.5 draw / 0 loss) and ``G`` is a
goal-difference multiplier that increases the update for larger winning
margins.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import config


def expected_score(rating_a: float, rating_b: float,
                   home_advantage: float = 0.0) -> float:
    """Expected score (win probability + half draw prob) of A vs B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - (rating_a + home_advantage)) / 400.0))


def _goal_diff_multiplier(goal_diff: int) -> float:
    """Margin-of-victory multiplier (World Football Elo convention)."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


@dataclass
class EloEngine:
    """Stateful Elo tracker.

    Call :meth:`pre_match` to read both teams' current ratings (the features),
    then :meth:`update` with the realised result to advance the state.
    """

    base: float = config.ELO_BASE
    k: float = config.ELO_K
    home_advantage: float = config.ELO_HOME_ADVANTAGE
    tournament_reversion: float = config.ELO_TOURNAMENT_REVERSION

    ratings: dict[str, float] = field(default_factory=dict)
    _last_year_seen: int | None = field(default=None, repr=False)

    def rating(self, team_id: str) -> float:
        return self.ratings.get(team_id, self.base)

    def maybe_revert(self, year: int) -> None:
        """Optionally regress all ratings toward the mean at a new tournament.

        Accounts for squad turnover across 4-year gaps. Controlled by
        ``tournament_reversion`` (0 == disabled / fully persistent Elo).
        """
        if self.tournament_reversion <= 0:
            return
        if self._last_year_seen is not None and year != self._last_year_seen:
            f = self.tournament_reversion
            self.ratings = {
                t: r + f * (self.base - r) for t, r in self.ratings.items()
            }
        self._last_year_seen = year

    def pre_match(self, team_a: str, team_b: str) -> tuple[float, float]:
        """Return current ratings of (team_a, team_b) before the match."""
        return self.rating(team_a), self.rating(team_b)

    def update(self, team_a: str, team_b: str,
               goals_a: int, goals_b: int) -> None:
        """Advance ratings using a finished match's regulation score."""
        ra, rb = self.rating(team_a), self.rating(team_b)
        exp_a = expected_score(ra, rb, self.home_advantage)

        if goals_a > goals_b:
            score_a = 1.0
        elif goals_a == goals_b:
            score_a = 0.5
        else:
            score_a = 0.0

        mult = _goal_diff_multiplier(goals_a - goals_b)
        delta = self.k * mult * (score_a - exp_a)
        self.ratings[team_a] = ra + delta
        self.ratings[team_b] = rb - delta


def compute_pre_match_elo(matches: pd.DataFrame, **engine_kwargs) -> pd.DataFrame:
    """Attach pre-match Elo ratings to every match (leakage-free).

    Iterates over ``matches`` (assumed chronologically sorted) and, for each,
    records both teams' ratings *before* updating with the result.

    Returns a copy of ``matches`` with added columns:
    ``home_elo_pre``, ``away_elo_pre``, ``home_elo_post``, ``away_elo_post``.
    """
    engine = EloEngine(**engine_kwargs)
    home_pre, away_pre, home_post, away_post = [], [], [], []

    for row in matches.itertuples(index=False):
        engine.maybe_revert(int(row.year))
        ha = row.home_team_id
        aw = row.away_team_id
        ra, rb = engine.pre_match(ha, aw)
        home_pre.append(ra)
        away_pre.append(rb)
        engine.update(ha, aw, int(row.home_team_score), int(row.away_team_score))
        home_post.append(engine.rating(ha))
        away_post.append(engine.rating(aw))

    out = matches.copy()
    out["home_elo_pre"] = home_pre
    out["away_elo_pre"] = away_pre
    out["home_elo_post"] = home_post
    out["away_elo_post"] = away_post
    return out


def final_ratings(matches: pd.DataFrame, **engine_kwargs) -> dict[str, float]:
    """Return the Elo rating of every team after the last match in ``matches``.

    Used to seed predictions for the upcoming (2026) tournament with each
    team's most recent strength estimate.
    """
    engine = EloEngine(**engine_kwargs)
    for row in matches.itertuples(index=False):
        engine.maybe_revert(int(row.year))
        engine.update(
            row.home_team_id, row.away_team_id,
            int(row.home_team_score), int(row.away_team_score),
        )
    return dict(engine.ratings)
