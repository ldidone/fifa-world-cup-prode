"""Poisson goal model: predict goals and derive outcome probabilities.

Idea
----
Football goals are well-approximated by a Poisson process. We model the
*expected goals* of the scoring side as a function of:

* its own attacking rate (historical goals-for average),
* the opponent's defensive rate (historical goals-against average),
* the Elo difference (overall strength edge),
* whether the match is a knockout (tends to be lower-scoring).

A single :class:`~sklearn.linear_model.PoissonRegressor` is fitted on a
"scorer-perspective" table where each match contributes two rows (A-scoring and
B-scoring). For prediction we obtain ``lambda_a`` and ``lambda_b`` for a match,
assume the two scorelines are (conditionally) independent Poisson draws, and
build the full score-probability matrix. Summing the appropriate cells gives
P(team_a win) / P(draw) / P(team_b win), and the matrix also yields the most
likely exact score and expected goals.

Independence is a simplification (real scores are mildly correlated, and draws
are slightly under-predicted -- the Dixon-Coles correction addresses this), but
it is transparent, leakage-free, and a solid baseline for goal prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from . import config

_POISSON_FEATURES = ["attack_rate", "opp_defense_rate", "elo_diff", "knockout_stage"]
_MAX_GOALS = 10  # truncate score matrix here (P(>10 goals) is negligible)


def _scorer_table(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build the stacked scorer-perspective design matrix and goal targets."""
    # team_a scoring against team_b
    a = pd.DataFrame({
        "attack_rate": df["team_a_gf_avg"],
        "opp_defense_rate": df["team_b_ga_avg"],
        "elo_diff": df["team_a_elo_pre"] - df["team_b_elo_pre"],
        "knockout_stage": df["knockout_stage"],
        "goals": df["goals_a"],
    })
    # team_b scoring against team_a
    b = pd.DataFrame({
        "attack_rate": df["team_b_gf_avg"],
        "opp_defense_rate": df["team_a_ga_avg"],
        "elo_diff": df["team_b_elo_pre"] - df["team_a_elo_pre"],
        "knockout_stage": df["knockout_stage"],
        "goals": df["goals_b"],
    })
    stacked = pd.concat([a, b], ignore_index=True)
    # Neutral imputation for teams without history (debutants).
    overall_mean = stacked[["attack_rate", "opp_defense_rate"]].mean()
    stacked["attack_rate"] = stacked["attack_rate"].fillna(overall_mean["attack_rate"])
    stacked["opp_defense_rate"] = stacked["opp_defense_rate"].fillna(
        overall_mean["opp_defense_rate"])
    X = stacked[_POISSON_FEATURES].to_numpy()
    y = stacked["goals"].to_numpy()
    return X, y


class PoissonGoalModel:
    """Fit lambda(attacker, defender) and convert to score/outcome probs."""

    def __init__(self, alpha: float = 1e-6):
        self.reg = PoissonRegressor(alpha=alpha, max_iter=1000)
        self._train_means: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "PoissonGoalModel":
        X, y = _scorer_table(df)
        self.reg.fit(X, y)
        self._train_means = {
            "attack_rate": float(np.nanmean(
                pd.concat([df["team_a_gf_avg"], df["team_b_gf_avg"]]))),
            "defense_rate": float(np.nanmean(
                pd.concat([df["team_a_ga_avg"], df["team_b_ga_avg"]]))),
        }
        return self

    def _design(self, attack_rate, opp_defense_rate, elo_diff, knockout):
        ar = self._train_means["attack_rate"] if pd.isna(attack_rate) else attack_rate
        dr = self._train_means["defense_rate"] if pd.isna(opp_defense_rate) else opp_defense_rate
        return np.array([[ar, dr, elo_diff, knockout]])

    def predict_lambdas(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (lambda_a, lambda_b) expected goals for each match row."""
        lam_a, lam_b = [], []
        for r in df.itertuples(index=False):
            xa = self._design(r.team_a_gf_avg, r.team_b_ga_avg,
                              r.team_a_elo_pre - r.team_b_elo_pre, r.knockout_stage)
            xb = self._design(r.team_b_gf_avg, r.team_a_ga_avg,
                              r.team_b_elo_pre - r.team_a_elo_pre, r.knockout_stage)
            lam_a.append(float(self.reg.predict(xa)[0]))
            lam_b.append(float(self.reg.predict(xb)[0]))
        return np.array(lam_a), np.array(lam_b)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Outcome probabilities (n, 3) in OUTCOME_CLASSES order."""
        lam_a, lam_b = self.predict_lambdas(df)
        out = np.zeros((len(df), 3))
        for i in range(len(df)):
            pa, pd_, pb = _outcome_from_lambdas(lam_a[i], lam_b[i])
            out[i] = [pa, pd_, pb]
        return out

    def predict_scoreline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Most-likely exact score + expected goals for each match row."""
        lam_a, lam_b = self.predict_lambdas(df)
        recs = []
        for i in range(len(df)):
            ea, eb, sa, sb = _expected_and_mode(lam_a[i], lam_b[i])
            recs.append({
                "exp_goals_a": ea, "exp_goals_b": eb,
                "likely_score_a": sa, "likely_score_b": sb,
            })
        return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# Score-matrix helpers
# --------------------------------------------------------------------------- #
def _score_matrix(lam_a: float, lam_b: float) -> np.ndarray:
    ka = poisson.pmf(np.arange(_MAX_GOALS + 1), lam_a)
    kb = poisson.pmf(np.arange(_MAX_GOALS + 1), lam_b)
    return np.outer(ka, kb)  # M[i, j] = P(A=i, B=j)


def _outcome_from_lambdas(lam_a: float, lam_b: float) -> tuple[float, float, float]:
    m = _score_matrix(lam_a, lam_b)
    p_a = np.tril(m, -1).sum()   # A goals > B goals
    p_draw = np.trace(m)
    p_b = np.triu(m, 1).sum()
    total = p_a + p_draw + p_b
    return p_a / total, p_draw / total, p_b / total


def _expected_and_mode(lam_a: float, lam_b: float) -> tuple[float, float, int, int]:
    m = _score_matrix(lam_a, lam_b)
    sa, sb = np.unravel_index(np.argmax(m), m.shape)
    return round(lam_a, 2), round(lam_b, 2), int(sa), int(sb)
