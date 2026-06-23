"""Leakage-free feature engineering and modeling-dataset construction.

Design principles
-----------------
1. **No leakage.** Every feature for a match is computed from matches that
   finished *strictly before* it. We do this with a single chronological pass
   that records each team's running state *before* applying the match result.

2. **Neutral teams.** World Cup venues are neutral, and the dataset's
   home/away labels merely reflect listing order (which is biased toward
   stronger/seeded teams). We therefore build a *symmetric* dataset: each match
   becomes one row with ``team_a`` / ``team_b`` assigned by a seeded coin flip,
   and the model consumes antisymmetric *difference* features (a minus b) plus a
   few neutral context features. This removes listing-order leakage.

3. **Difference features.** elo_diff, win_rate_diff, etc. are naturally
   antisymmetric, which is exactly the right inductive bias for "who wins".

The public entry point is :func:`build_modeling_dataset`.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from . import config, data, elo
from .external_elo import ExternalElo, load_external_elo


# --------------------------------------------------------------------------- #
# Per-team running state
# --------------------------------------------------------------------------- #
class _TeamState:
    """Accumulates a single team's history as we walk forward in time."""

    __slots__ = ("played", "wins", "draws", "losses", "gf", "ga",
                 "tournaments", "form")

    def __init__(self, form_window: int):
        self.played = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.gf = 0          # cumulative goals for
        self.ga = 0          # cumulative goals against
        self.tournaments: set[str] = set()
        self.form: deque[int] = deque(maxlen=form_window)  # points: 3/1/0

    # --- pre-match readouts (features) ------------------------------------ #
    def win_rate(self) -> float:
        return self.wins / self.played if self.played else np.nan

    def gf_avg(self) -> float:
        return self.gf / self.played if self.played else np.nan

    def ga_avg(self) -> float:
        return self.ga / self.played if self.played else np.nan

    def form_points(self) -> float:
        return float(np.mean(self.form)) if self.form else np.nan

    def experience(self) -> int:
        return len(self.tournaments)

    # --- post-match update ------------------------------------------------ #
    def update(self, gf: int, ga: int, tournament_id: str) -> None:
        self.played += 1
        self.gf += gf
        self.ga += ga
        self.tournaments.add(tournament_id)
        if gf > ga:
            self.wins += 1
            self.form.append(3)
        elif gf == ga:
            self.draws += 1
            self.form.append(1)
        else:
            self.losses += 1
            self.form.append(0)


# --------------------------------------------------------------------------- #
# Step 1: per-side pre-match features (home / away listing order)
# --------------------------------------------------------------------------- #
def build_team_match_features(
    matches: pd.DataFrame, form_window: int = config.FORM_WINDOW,
    ext_elo: ExternalElo | None = None,
    **elo_kwargs,
) -> pd.DataFrame:
    """Attach pre-match features for both listed sides of every match.

    Returns a copy of ``matches`` with ``home_*`` / ``away_*`` pre-match
    feature columns plus head-to-head and host context. Still in listing-order
    space (symmetrization happens later).

    ``ext_elo`` (defaults to the bundled eloratings.net history) supplies a
    leakage-free external Elo for each side, read strictly *before* the match
    date. Pass ``ext_elo=False`` to disable.
    """
    if ext_elo is None:
        ext_elo = load_external_elo()
    elif ext_elo is False:
        ext_elo = None

    m = elo.compute_pre_match_elo(matches, **elo_kwargs)

    states: dict[str, _TeamState] = defaultdict(lambda: _TeamState(form_window))
    # Head-to-head ledger keyed by ordered pair (teamX, teamY): cumulative
    # (wins_of_X_over_Y, draws, goals_X, goals_Y, played).
    h2h: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])

    hosts = data.host_country_map()

    cols: dict[str, list] = defaultdict(list)

    for row in m.itertuples(index=False):
        ha, aw = row.home_team_id, row.away_team_id
        sa, sb = states[ha], states[aw]

        # --- team-level pre-match features -------------------------------- #
        cols["home_matches_played"].append(sa.played)
        cols["away_matches_played"].append(sb.played)
        cols["home_win_rate"].append(sa.win_rate())
        cols["away_win_rate"].append(sb.win_rate())
        cols["home_gf_avg"].append(sa.gf_avg())
        cols["away_gf_avg"].append(sb.gf_avg())
        cols["home_ga_avg"].append(sa.ga_avg())
        cols["away_ga_avg"].append(sb.ga_avg())
        cols["home_form_points"].append(sa.form_points())
        cols["away_form_points"].append(sb.form_points())
        cols["home_experience"].append(sa.experience())
        cols["away_experience"].append(sb.experience())

        # --- head-to-head (home perspective) ------------------------------ #
        rec = h2h[(ha, aw)]
        h2h_played = rec[4]
        cols["h2h_played"].append(h2h_played)
        cols["h2h_home_win_rate"].append(rec[0] / h2h_played if h2h_played else np.nan)
        cols["h2h_goal_diff_avg"].append(
            (rec[2] - rec[3]) / h2h_played if h2h_played else np.nan
        )

        # --- host indicator ----------------------------------------------- #
        host_set = hosts.get(int(row.year), set())
        cols["home_is_host"].append(int(row.home_team_name in host_set))
        cols["away_is_host"].append(int(row.away_team_name in host_set))

        # --- external Elo (leakage-free: strictly before match date) ------- #
        if ext_elo is not None:
            cols["home_ext_elo"].append(
                ext_elo.rating_asof(row.home_team_name, row.match_date))
            cols["away_ext_elo"].append(
                ext_elo.rating_asof(row.away_team_name, row.match_date))

        # --- advance state with realised result --------------------------- #
        gh, ga_ = int(row.home_team_score), int(row.away_team_score)
        sa.update(gh, ga_, row.tournament_id)
        sb.update(ga_, gh, row.tournament_id)

        # Update both ordered head-to-head records. Each record is
        # [wins_of_first_over_second, draws, goals_first, goals_second, played].
        rec_ha = h2h[(ha, aw)]   # home perspective
        rec_aw = h2h[(aw, ha)]   # away perspective
        rec_ha[4] += 1
        rec_aw[4] += 1
        rec_ha[2] += gh
        rec_ha[3] += ga_
        rec_aw[2] += ga_
        rec_aw[3] += gh
        if gh > ga_:
            rec_ha[0] += 1
        elif gh == ga_:
            rec_ha[1] += 1
            rec_aw[1] += 1
        else:
            rec_aw[0] += 1

    feat = m.copy()
    for c, v in cols.items():
        feat[c] = v
    return feat


# --------------------------------------------------------------------------- #
# Step 2: confederation join
# --------------------------------------------------------------------------- #
def _attach_confederation(feat: pd.DataFrame) -> pd.DataFrame:
    teams = data.load_teams()[["team_id", "confederation_code"]]
    feat = feat.merge(
        teams.rename(columns={"team_id": "home_team_id",
                              "confederation_code": "home_conf"}),
        on="home_team_id", how="left",
    )
    feat = feat.merge(
        teams.rename(columns={"team_id": "away_team_id",
                              "confederation_code": "away_conf"}),
        on="away_team_id", how="left",
    )
    return feat


# --------------------------------------------------------------------------- #
# Step 3: symmetrization into team_a / team_b + target
# --------------------------------------------------------------------------- #
# Difference features (team_a minus team_b). Antisymmetric: swap a<->b flips sign.
_DIFF_FEATURES = [
    "elo", "matches_played", "win_rate", "gf_avg", "ga_avg",
    "form_points", "experience",
]


def _target_from_goals(goals_a: int, goals_b: int) -> int:
    if goals_a > goals_b:
        return config.OUTCOME_TO_INT["team_a_win"]
    if goals_a == goals_b:
        return config.OUTCOME_TO_INT["draw"]
    return config.OUTCOME_TO_INT["team_b_win"]


def build_modeling_dataset(
    matches: pd.DataFrame | None = None,
    seed: int = config.RANDOM_SEED,
    form_window: int = config.FORM_WINDOW,
    **elo_kwargs,
) -> pd.DataFrame:
    """Build the final, symmetric, leakage-free modeling table.

    One row per match. For each match we randomly decide (seeded) whether the
    listed home side becomes ``team_a`` or ``team_b``; features are emitted as
    ``*_diff`` (team_a - team_b) plus neutral context, and the target is the
    goal-based regulation outcome from team_a's perspective.

    Returned columns include identifiers (year, tournament, stage, names),
    feature columns, ``goals_a`` / ``goals_b`` (regulation), and ``target``.
    """
    if matches is None:
        matches = data.load_men_matches()

    feat = build_team_match_features(matches, form_window=form_window, **elo_kwargs)
    feat = _attach_confederation(feat)
    has_ext = "home_ext_elo" in feat.columns

    rng = np.random.default_rng(seed)
    swap = rng.random(len(feat)) < 0.5  # True -> away listed side becomes team_a

    rows = []
    for i, row in enumerate(feat.itertuples(index=False)):
        if swap[i]:
            a, b = "away", "home"
        else:
            a, b = "home", "away"

        def g(side: str, name: str):
            return getattr(row, f"{side}_{name}")

        goals_a = int(g(a, "team_score"))
        goals_b = int(g(b, "team_score"))

        rec = {
            "match_id": row.match_id,
            "year": row.year,
            "tournament_id": row.tournament_id,
            "tournament_name": row.tournament_name,
            "stage_name": row.stage_name,
            "knockout_stage": int(row.knockout_stage),
            "team_a_id": g(a, "team_id"),
            "team_b_id": g(b, "team_id"),
            "team_a_name": g(a, "team_name"),
            "team_b_name": g(b, "team_name"),
            "team_a_elo_pre": g(a, "elo_pre"),
            "team_b_elo_pre": g(b, "elo_pre"),
            # raw per-team attack/defence rates (used by the Poisson model)
            "team_a_gf_avg": g(a, "gf_avg"),
            "team_a_ga_avg": g(a, "ga_avg"),
            "team_b_gf_avg": g(b, "gf_avg"),
            "team_b_ga_avg": g(b, "ga_avg"),
            "goals_a": goals_a,
            "goals_b": goals_b,
            "penalty_shootout": int(row.penalty_shootout),
            "target": _target_from_goals(goals_a, goals_b),
        }

        # antisymmetric difference features
        rec["elo_diff"] = g(a, "elo_pre") - g(b, "elo_pre")
        if has_ext:
            rec["ext_elo_diff"] = _safe_sub(g(a, "ext_elo"), g(b, "ext_elo"))
        rec["matches_played_diff"] = g(a, "matches_played") - g(b, "matches_played")
        rec["win_rate_diff"] = _safe_sub(g(a, "win_rate"), g(b, "win_rate"))
        rec["gf_avg_diff"] = _safe_sub(g(a, "gf_avg"), g(b, "gf_avg"))
        rec["ga_avg_diff"] = _safe_sub(g(a, "ga_avg"), g(b, "ga_avg"))
        rec["form_points_diff"] = _safe_sub(g(a, "form_points"), g(b, "form_points"))
        rec["experience_diff"] = g(a, "experience") - g(b, "experience")

        # host indicator (signed: +1 if team_a is host, -1 if team_b is host)
        rec["host_advantage"] = g(a, "is_host") - g(b, "is_host")

        # head-to-head, re-oriented to team_a perspective
        h2h_played = row.h2h_played
        rec["h2h_played"] = h2h_played
        if h2h_played and not np.isnan(row.h2h_home_win_rate):
            if a == "home":
                rec["h2h_a_win_rate"] = row.h2h_home_win_rate
                rec["h2h_goal_diff_avg"] = row.h2h_goal_diff_avg
            else:
                rec["h2h_a_win_rate"] = 1.0 - row.h2h_home_win_rate
                rec["h2h_goal_diff_avg"] = -row.h2h_goal_diff_avg
        else:
            rec["h2h_a_win_rate"] = np.nan
            rec["h2h_goal_diff_avg"] = np.nan

        rec["same_confederation"] = int(g(a, "conf") == g(b, "conf"))
        rec["team_a_conf"] = g(a, "conf")
        rec["team_b_conf"] = g(b, "conf")

        rows.append(rec)

    df = pd.DataFrame(rows)
    return df


def _safe_sub(x, y):
    if x is None or y is None or (isinstance(x, float) and np.isnan(x)) or \
       (isinstance(y, float) and np.isnan(y)):
        return np.nan
    return x - y


# --------------------------------------------------------------------------- #
# Feature lists used by the models
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS = [
    "elo_diff",
    "ext_elo_diff",
    "matches_played_diff",
    "win_rate_diff",
    "gf_avg_diff",
    "ga_avg_diff",
    "form_points_diff",
    "experience_diff",
    "host_advantage",
    "h2h_played",
    "h2h_a_win_rate",
    "h2h_goal_diff_avg",
    "same_confederation",
    "knockout_stage",
]


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select model features; impute NaNs (no-history cases) with neutral 0.

    For difference/rate features a missing value means "no prior history",
    which is most naturally encoded as 0 (no known edge either way).
    """
    # reindex so the matrix always has every model feature, even if a source
    # (e.g. external Elo) was disabled for a given build.
    X = df.reindex(columns=FEATURE_COLUMNS)
    return X.fillna(0.0)


# --------------------------------------------------------------------------- #
# Forecasting helpers: extract each team's *latest* state and build a single
# prediction-feature row for an upcoming (e.g. 2026) match.
# --------------------------------------------------------------------------- #
class HistoryState:
    """Snapshot of all teams' accumulated state after the last historical match.

    Holds final Elo ratings, per-team running stats, head-to-head ledger and a
    confederation lookup, so that feature rows for future fixtures can be built
    consistently with the training-time feature definitions.
    """

    def __init__(self, ratings, states, h2h, conf, ext_ratings=None):
        self.ratings = ratings
        self.states = states
        self.h2h = h2h
        self.conf = conf
        # team_id -> external Elo as of the prediction cutoff (None if unknown)
        self.ext_ratings = ext_ratings or {}

    def rating(self, team_id: str) -> float:
        return self.ratings.get(team_id, config.ELO_BASE)

    def ext_rating(self, team_id: str):
        return self.ext_ratings.get(team_id)


def fit_history(matches: pd.DataFrame,
                form_window: int = config.FORM_WINDOW,
                ext_elo: ExternalElo | None = None,
                ext_cutoff=None,
                **elo_kwargs) -> HistoryState:
    """Walk all matches once and return the final accumulated state.

    ``ext_elo`` (defaults to the bundled history) provides each team's external
    Elo as of ``ext_cutoff`` (default: latest available snapshot), used as a
    leakage-free strength prior for forecasting. Pass ``ext_elo=False`` to skip.
    """
    if ext_elo is None:
        ext_elo = load_external_elo()
    elif ext_elo is False:
        ext_elo = None

    eng = elo.EloEngine(**elo_kwargs)
    states: dict[str, _TeamState] = defaultdict(lambda: _TeamState(form_window))
    h2h: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])

    for row in matches.itertuples(index=False):
        eng.maybe_revert(int(row.year))
        ha, aw = row.home_team_id, row.away_team_id
        gh, ga_ = int(row.home_team_score), int(row.away_team_score)
        eng.update(ha, aw, gh, ga_)
        states[ha].update(gh, ga_, row.tournament_id)
        states[aw].update(ga_, gh, row.tournament_id)
        rec_ha = h2h[(ha, aw)]
        rec_aw = h2h[(aw, ha)]
        rec_ha[4] += 1; rec_aw[4] += 1
        rec_ha[2] += gh; rec_ha[3] += ga_
        rec_aw[2] += ga_; rec_aw[3] += gh
        if gh > ga_:
            rec_ha[0] += 1
        elif gh == ga_:
            rec_ha[1] += 1; rec_aw[1] += 1
        else:
            rec_aw[0] += 1

    teams = data.load_teams()
    conf = dict(zip(teams["team_id"], teams["confederation_code"]))

    ext_ratings: dict[str, float] = {}
    if ext_elo is not None:
        id_to_name = dict(zip(teams["team_id"], teams["team_name"]))
        for tid, name in id_to_name.items():
            r = ext_elo.latest_rating(name, ext_cutoff)
            if r is not None:
                ext_ratings[tid] = r

    return HistoryState(dict(eng.ratings), dict(states), dict(h2h), conf,
                        ext_ratings)


def match_feature_row(hist: HistoryState, a_id: str, b_id: str,
                      knockout: int = 0,
                      host_ids: set[str] | None = None) -> dict:
    """Build a single prediction row (features + Elo + goal rates) for a fixture.

    Uses each team's *latest* historical state, so it is the natural extension
    of the training-time feature definitions to an unplayed match.
    """
    host_ids = host_ids or set()
    sa = hist.states.get(a_id)
    sb = hist.states.get(b_id)

    def _stat(s, fn, default=np.nan):
        return fn(s) if s is not None else default

    ea, eb = hist.rating(a_id), hist.rating(b_id)
    a_wr, b_wr = _stat(sa, _TeamState.win_rate), _stat(sb, _TeamState.win_rate)
    a_gf, b_gf = _stat(sa, _TeamState.gf_avg), _stat(sb, _TeamState.gf_avg)
    a_ga, b_ga = _stat(sa, _TeamState.ga_avg), _stat(sb, _TeamState.ga_avg)
    a_fp, b_fp = _stat(sa, _TeamState.form_points), _stat(sb, _TeamState.form_points)
    a_exp = _stat(sa, _TeamState.experience, 0)
    b_exp = _stat(sb, _TeamState.experience, 0)
    a_mp = sa.played if sa is not None else 0
    b_mp = sb.played if sb is not None else 0

    rec = hist.h2h.get((a_id, b_id), [0, 0, 0, 0, 0])
    h2h_played = rec[4]

    row = {
        "team_a_id": a_id, "team_b_id": b_id,
        "team_a_elo_pre": ea, "team_b_elo_pre": eb,
        "team_a_gf_avg": a_gf, "team_a_ga_avg": a_ga,
        "team_b_gf_avg": b_gf, "team_b_ga_avg": b_ga,
        "knockout_stage": int(knockout),
        "elo_diff": ea - eb,
        "ext_elo_diff": _safe_sub(hist.ext_rating(a_id), hist.ext_rating(b_id)),
        "matches_played_diff": a_mp - b_mp,
        "win_rate_diff": _safe_sub(a_wr, b_wr),
        "gf_avg_diff": _safe_sub(a_gf, b_gf),
        "ga_avg_diff": _safe_sub(a_ga, b_ga),
        "form_points_diff": _safe_sub(a_fp, b_fp),
        "experience_diff": a_exp - b_exp,
        "host_advantage": int(a_id in host_ids) - int(b_id in host_ids),
        "h2h_played": h2h_played,
        "h2h_a_win_rate": (rec[0] / h2h_played) if h2h_played else np.nan,
        "h2h_goal_diff_avg": ((rec[2] - rec[3]) / h2h_played) if h2h_played else np.nan,
        "same_confederation": int(hist.conf.get(a_id) == hist.conf.get(b_id)),
    }
    return row
