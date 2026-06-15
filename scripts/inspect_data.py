"""Quick data inspection to understand the worldcup dataset before modeling."""
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)

for f in sorted(RAW.glob("*.csv")):
    df = pd.read_csv(f)
    print(f"\n{'='*70}\n{f.name}  shape={df.shape}")
    print("cols:", list(df.columns))

print("\n\n############ MATCHES DETAIL ############")
m = pd.read_csv(RAW / "matches.csv")
print(m[["tournament_name", "match_date", "stage_name", "home_team_name",
         "away_team_name", "home_team_score", "away_team_score", "result",
         "extra_time", "penalty_shootout"]].head())
print("\nresult value counts:\n", m["result"].value_counts(dropna=False))
print("\nmatches per tournament:\n", m.groupby("tournament_name").size())
print("\nmissing values (cols with any):\n", m.isna().sum()[m.isna().sum() > 0])
print("\nreplayed/replay flags:", m["replayed"].sum(), m["replay"].sum())
print("date range:", m["match_date"].min(), m["match_date"].max())

print("\n\n############ TEAM_APPEARANCES DETAIL ############")
ta = pd.read_csv(RAW / "team_appearances.csv")
print(ta[["tournament_name","match_date","team_name","opponent_name",
          "home_team","away_team","goals_for","goals_against","result"]].head())
print("\nresult vc:\n", ta["result"].value_counts())

print("\n\n############ TOURNAMENTS ############")
t = pd.read_csv(RAW / "tournaments.csv")
print(t[["tournament_name","year","host_country","winner","host_won","count_teams"]])

print("\n\n############ TEAMS / CONFED ############")
teams = pd.read_csv(RAW / "teams.csv")
print(teams["confederation_name"].value_counts())
print("n teams:", teams["team_id"].nunique())
