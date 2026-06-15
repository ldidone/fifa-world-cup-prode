"""Generate FIFA World Cup 2026 predictions from a trained model.

Pipeline
--------
1. Build the leakage-free modeling dataset from all historical men's matches
   and fit the chosen classifier + the Poisson goal model on *all* of it.
2. Snapshot each team's latest historical state (Elo, attack/defence, etc.).
3. For every fixture match, build a feature row and predict:
   * P(team_a win) / P(draw) / P(team_b win)
   * predicted outcome
   * Poisson expected goals + most-likely exact score
4. (Optional) Monte-Carlo simulate the full tournament to estimate each team's
   probability of advancing from the group and lifting the trophy.

Team-name -> dataset team_id matching is done by exact name with a small alias
table; unmatched teams are treated as debutants (base Elo, no history) and
reported so the user can fix the fixture spelling if desired.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data, features, poisson, train

# Minor spelling aliases: fixture name -> dataset team_name.
# Maps alternative / FIFA-style names to the exact spelling in teams.csv.
NAME_ALIASES = {
    "Korea Republic": "South Korea",
    "USA": "United States",
    "Cote d'Ivoire": "Ivory Coast",
    "DR Congo": "Zaire",           # played in 1974 as Zaire
    "Curacao": "Curacao",           # debutant — no historical entry
    "Cape Verde": "Cape Verde",     # debutant — no historical entry
}


def _name_to_id_map() -> dict[str, str]:
    teams = data.load_teams()
    return dict(zip(teams["team_name"], teams["team_id"]))


def resolve_team_ids(names: list[str]) -> tuple[dict[str, str], list[str]]:
    """Map fixture team names to dataset team_ids.

    Returns (name->id map, list of unmatched names). Unmatched names get a
    synthetic id ``"NEW-<name>"`` so they still flow through as debutants.
    """
    name2id = _name_to_id_map()
    resolved, unmatched = {}, []
    for n in names:
        key = NAME_ALIASES.get(n, n)
        if key in name2id:
            resolved[n] = name2id[key]
        else:
            resolved[n] = f"NEW-{n}"
            unmatched.append(n)
    return resolved, unmatched


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def fit_full(model_name: str = "hist_gradient_boosting"):
    """Fit the outcome classifier + Poisson model on all historical data.

    Returns (clf, poisson_model, history_state).
    """
    matches = data.load_men_matches()
    model_df = features.build_modeling_dataset(matches)

    models = train.build_models()
    clf_name = model_name if model_name in models else "logistic_regression"
    clf = train.fit_model(models[clf_name], model_df)

    pois = poisson.PoissonGoalModel().fit(model_df)
    hist = features.fit_history(matches)
    return clf, pois, hist, clf_name


# --------------------------------------------------------------------------- #
# Per-match prediction
# --------------------------------------------------------------------------- #
def predict_matches(fixture: pd.DataFrame, clf, pois,
                    hist: features.HistoryState,
                    host_names: set[str]) -> pd.DataFrame:
    """Predict every row of a fixture dataframe.

    ``fixture`` must contain ``team_a`` and ``team_b`` name columns and may
    contain ``stage`` (``"group"``/``"knockout"``), ``group``, ``match_number``.
    """
    name2id, _ = resolve_team_ids(
        sorted(set(fixture["team_a"]) | set(fixture["team_b"])))
    host_ids = {name2id[n] for n in host_names if n in name2id}

    rows = []
    for r in fixture.itertuples(index=False):
        a_id, b_id = name2id[r.team_a], name2id[r.team_b]
        knockout = int(getattr(r, "stage", "group") != "group")
        feat = features.match_feature_row(hist, a_id, b_id, knockout, host_ids)
        rows.append(feat)
    feat_df = pd.DataFrame(rows)

    proba = train.predict_proba(clf, feat_df)
    scores = pois.predict_scoreline(feat_df)

    out = fixture.reset_index(drop=True).copy()
    out["p_team_a_win"] = proba[:, 0].round(4)
    out["p_draw"] = proba[:, 1].round(4)
    out["p_team_b_win"] = proba[:, 2].round(4)
    out["predicted_outcome"] = [
        config.INT_TO_OUTCOME[i] for i in proba.argmax(axis=1)]
    out["predicted_outcome_label"] = out.apply(
        lambda x: {
            "team_a_win": x["team_a"], "team_b_win": x["team_b"], "draw": "Draw",
        }[x["predicted_outcome"]], axis=1)
    out["team_a_elo"] = feat_df["team_a_elo_pre"].round(0).astype(int)
    out["team_b_elo"] = feat_df["team_b_elo_pre"].round(0).astype(int)
    out = pd.concat([out, scores], axis=1)
    out["predicted_score"] = (out["likely_score_a"].astype(str) + "-"
                              + out["likely_score_b"].astype(str))
    return out


# --------------------------------------------------------------------------- #
# Monte-Carlo tournament simulation
# --------------------------------------------------------------------------- #
def _sample_outcome(p: np.ndarray, rng) -> int:
    return int(rng.choice(3, p=p))


def simulate_tournament(groups: dict[str, list[str]], clf, pois,
                        hist: features.HistoryState, host_names: set[str],
                        n_sims: int = 2000,
                        seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Monte-Carlo estimate of advancement / title probabilities.

    Group stage uses model outcome probabilities (3 pts win / 1 draw). The top
    2 of each group plus the 8 best third-placed teams advance to a 32-team
    knockout, which is then resolved with a simplified seeded bracket using the
    win/lose probabilities (draws split). The bracket crossing is a transparent
    approximation of the official one (the exact crossings depend on the final
    draw), so treat deep-run odds as indicative rather than exact.
    """
    rng = np.random.default_rng(seed)
    name2id, _ = resolve_team_ids([t for ts in groups.values() for t in ts])
    host_ids = {name2id[n] for n in host_names if n in name2id}

    teams = [t for ts in groups.values() for t in ts]
    # Precompute pairwise outcome probabilities for all teams (group + knockout).
    def pair_proba(a, b, knockout):
        feat = features.match_feature_row(
            hist, name2id[a], name2id[b], knockout, host_ids)
        p = train.predict_proba(clf, pd.DataFrame([feat]))[0]
        return p

    group_probs = {}
    ko_probs = {}
    for g, ts in groups.items():
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                group_probs[(ts[i], ts[j])] = pair_proba(ts[i], ts[j], 0)
    # knockout probs computed lazily/cached
    def ko_pair(a, b):
        key = (a, b)
        if key not in ko_probs:
            ko_probs[key] = pair_proba(a, b, 1)
        return ko_probs[key]

    counts = {t: {"advance": 0, "champion": 0} for t in teams}

    for _ in range(n_sims):
        qualifiers = []
        thirds = []
        for g, ts in groups.items():
            pts = {t: 0 for t in ts}
            gd = {t: 0 for t in ts}
            for i in range(len(ts)):
                for j in range(i + 1, len(ts)):
                    a, b = ts[i], ts[j]
                    o = _sample_outcome(group_probs[(a, b)], rng)
                    if o == 0:
                        pts[a] += 3
                    elif o == 1:
                        pts[a] += 1; pts[b] += 1
                    else:
                        pts[b] += 3
            ranked = sorted(ts, key=lambda t: (pts[t], gd[t], rng.random()),
                            reverse=True)
            qualifiers.extend(ranked[:2])
            thirds.append((ranked[2], pts[ranked[2]]))
        # 8 best third-placed teams
        thirds.sort(key=lambda x: (x[1], rng.random()), reverse=True)
        qualifiers.extend([t for t, _ in thirds[:8]])

        for t in qualifiers:
            counts[t]["advance"] += 1

        # Simplified single-elimination bracket over the 32 qualifiers.
        bracket = qualifiers[:]
        rng.shuffle(bracket)
        while len(bracket) > 1:
            nxt = []
            for k in range(0, len(bracket), 2):
                a, b = bracket[k], bracket[k + 1]
                p = ko_pair(a, b)
                p_a = p[0] + p[1] / 2.0  # split draw prob in knockout
                nxt.append(a if rng.random() < p_a else b)
            bracket = nxt
        counts[bracket[0]]["champion"] += 1

    rows = []
    for t in teams:
        rows.append({
            "team": t,
            "elo": int(hist.rating(name2id[t])),
            "p_advance": round(counts[t]["advance"] / n_sims, 4),
            "p_champion": round(counts[t]["champion"] / n_sims, 4),
        })
    return pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)
