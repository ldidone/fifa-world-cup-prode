"""FIFA World Cup 2026 match-outcome prediction package.

Modules
-------
config      : project paths and global constants
data        : loading and cleaning of the raw worldcup CSV dataset
elo         : chronological Elo rating engine
features    : leakage-free feature engineering and modeling-dataset builder
target      : match-outcome target definition helpers
train       : model factory and training utilities
evaluate    : metrics and time-based (tournament) validation
poisson     : Poisson goal model (score / goal prediction)
predict     : generate predictions for the 2026 fixture
"""

__all__ = [
    "config",
    "data",
    "elo",
    "features",
    "target",
    "train",
    "evaluate",
    "poisson",
    "predict",
]
