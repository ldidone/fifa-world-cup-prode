"""Generate the FIFA World Cup 2026 fixture CSV from the official draw.

Source: https://www.fifa.com/es/tournaments/mens/worldcup/canadamexicousa2026/
        articles/calendario-fixture-mundial-2026-partidos-fechas

The draw was held on 5 December 2025 in Washington D.C.
48 teams, 12 groups of 4, 72 group-stage matches + 32 knockout = 104 total.

Team names are spelled to match the historical dataset (`data/raw/teams.csv`)
where possible. Debutant teams (no prior men's WC appearance) get a synthetic
ID and the base Elo of 1500 in the prediction pipeline.

Edit ``GROUPS`` and ``GROUP_MATCHES`` if the draw changes, then re-run:
    python scripts/build_fixture_2026.py
"""
from __future__ import annotations

import csv
from pathlib import Path

# --------------------------------------------------------------------------- #
# Hosts (used for the host-advantage feature)
# --------------------------------------------------------------------------- #
HOSTS = {"United States", "Mexico", "Canada"}

# --------------------------------------------------------------------------- #
# Official groups (FIFA draw, 5 Dec 2025)
# --------------------------------------------------------------------------- #
GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# --------------------------------------------------------------------------- #
# Official match schedule — 72 group-stage matches in chronological order.
# Each tuple: (match_number, date, team_a, team_b, group, venue)
# --------------------------------------------------------------------------- #
GROUP_MATCHES: list[tuple[int, str, str, str, str, str]] = [
    # --- Matchday 1 ---
    (1,  "2026-06-11", "Mexico",         "South Africa",             "A", "Mexico City"),
    (2,  "2026-06-11", "South Korea",    "Czech Republic",           "A", "Guadalajara"),
    (3,  "2026-06-12", "Canada",         "Bosnia and Herzegovina",   "B", "Toronto"),
    (4,  "2026-06-12", "United States",  "Paraguay",                 "D", "Los Angeles"),
    (5,  "2026-06-13", "Qatar",          "Switzerland",              "B", "San Francisco"),
    (6,  "2026-06-13", "Brazil",         "Morocco",                  "C", "New York"),
    (7,  "2026-06-13", "Haiti",          "Scotland",                 "C", "Boston"),
    (8,  "2026-06-13", "Australia",       "Turkey",                  "D", "Vancouver"),
    (9,  "2026-06-14", "Germany",        "Curacao",                  "E", "Houston"),
    (10, "2026-06-14", "Netherlands",    "Japan",                    "F", "Dallas"),
    (11, "2026-06-14", "Ivory Coast",    "Ecuador",                  "E", "Philadelphia"),
    (12, "2026-06-14", "Sweden",         "Tunisia",                  "F", "Monterrey"),
    (13, "2026-06-15", "Spain",          "Cape Verde",               "H", "Atlanta"),
    (14, "2026-06-15", "Belgium",        "Egypt",                    "G", "Seattle"),
    (15, "2026-06-15", "Saudi Arabia",   "Uruguay",                  "H", "Miami"),
    (16, "2026-06-15", "Iran",           "New Zealand",              "G", "Los Angeles"),
    (17, "2026-06-16", "France",         "Senegal",                  "I", "New York"),
    (18, "2026-06-16", "Iraq",           "Norway",                   "I", "Boston"),
    (19, "2026-06-16", "Argentina",      "Algeria",                  "J", "Kansas City"),
    (20, "2026-06-16", "Austria",        "Jordan",                   "J", "San Francisco"),
    (21, "2026-06-17", "Portugal",       "DR Congo",                 "K", "Houston"),
    (22, "2026-06-17", "England",        "Croatia",                  "L", "Dallas"),
    (23, "2026-06-17", "Ghana",          "Panama",                   "L", "Toronto"),
    (24, "2026-06-17", "Uzbekistan",     "Colombia",                 "K", "Mexico City"),
    # --- Matchday 2 ---
    (25, "2026-06-18", "Czech Republic", "South Africa",             "A", "Atlanta"),
    (26, "2026-06-18", "Switzerland",    "Bosnia and Herzegovina",   "B", "Los Angeles"),
    (27, "2026-06-18", "Canada",         "Qatar",                    "B", "Vancouver"),
    (28, "2026-06-18", "Mexico",         "South Korea",              "A", "Guadalajara"),
    (29, "2026-06-19", "United States",  "Australia",                "D", "Seattle"),
    (30, "2026-06-19", "Scotland",       "Morocco",                  "C", "Boston"),
    (31, "2026-06-19", "Brazil",         "Haiti",                    "C", "Philadelphia"),
    (32, "2026-06-19", "Turkey",         "Paraguay",                 "D", "San Francisco"),
    (33, "2026-06-20", "Netherlands",    "Sweden",                   "F", "Houston"),
    (34, "2026-06-20", "Germany",        "Ivory Coast",              "E", "Toronto"),
    (35, "2026-06-20", "Ecuador",        "Curacao",                  "E", "Kansas City"),
    (36, "2026-06-20", "Tunisia",        "Japan",                    "F", "Monterrey"),
    (37, "2026-06-21", "Spain",          "Saudi Arabia",             "H", "Atlanta"),
    (38, "2026-06-21", "Belgium",        "Iran",                     "G", "Los Angeles"),
    (39, "2026-06-21", "Uruguay",        "Cape Verde",               "H", "Miami"),
    (40, "2026-06-21", "New Zealand",    "Egypt",                    "G", "Vancouver"),
    (41, "2026-06-22", "Argentina",      "Austria",                  "J", "Dallas"),
    (42, "2026-06-22", "France",         "Iraq",                     "I", "Philadelphia"),
    (43, "2026-06-22", "Norway",         "Senegal",                  "I", "New York"),
    (44, "2026-06-22", "Jordan",         "Algeria",                  "J", "San Francisco"),
    (45, "2026-06-23", "Portugal",       "Uzbekistan",               "K", "Houston"),
    (46, "2026-06-23", "England",        "Ghana",                    "L", "Boston"),
    (47, "2026-06-23", "Panama",         "Croatia",                  "L", "Toronto"),
    (48, "2026-06-23", "Colombia",       "DR Congo",                 "K", "Guadalajara"),
    # --- Matchday 3 ---
    (49, "2026-06-24", "Switzerland",    "Canada",                   "B", "Vancouver"),
    (50, "2026-06-24", "Bosnia and Herzegovina", "Qatar",            "B", "Seattle"),
    (51, "2026-06-24", "Scotland",       "Brazil",                   "C", "Miami"),
    (52, "2026-06-24", "Morocco",        "Haiti",                    "C", "Atlanta"),
    (53, "2026-06-24", "Czech Republic", "Mexico",                   "A", "Mexico City"),
    (54, "2026-06-24", "South Africa",   "South Korea",              "A", "Monterrey"),
    (55, "2026-06-25", "Curacao",        "Ivory Coast",              "E", "Philadelphia"),
    (56, "2026-06-25", "Ecuador",        "Germany",                  "E", "New York"),
    (57, "2026-06-25", "Japan",          "Sweden",                   "F", "Dallas"),
    (58, "2026-06-25", "Tunisia",        "Netherlands",              "F", "Kansas City"),
    (59, "2026-06-25", "Turkey",         "United States",            "D", "Los Angeles"),
    (60, "2026-06-25", "Paraguay",       "Australia",                "D", "San Francisco"),
    (61, "2026-06-26", "Norway",         "France",                   "I", "Boston"),
    (62, "2026-06-26", "Senegal",        "Iraq",                     "I", "Toronto"),
    (63, "2026-06-26", "Cape Verde",     "Saudi Arabia",             "H", "Houston"),
    (64, "2026-06-26", "Uruguay",        "Spain",                    "H", "Guadalajara"),
    (65, "2026-06-26", "Egypt",          "Iran",                     "G", "Seattle"),
    (66, "2026-06-26", "New Zealand",    "Belgium",                  "G", "Vancouver"),
    (67, "2026-06-27", "Panama",         "England",                  "L", "New York"),
    (68, "2026-06-27", "Croatia",        "Ghana",                    "L", "Philadelphia"),
    (69, "2026-06-27", "Colombia",       "Portugal",                 "K", "Miami"),
    (70, "2026-06-27", "DR Congo",       "Uzbekistan",               "K", "Atlanta"),
    (71, "2026-06-27", "Algeria",        "Austria",                  "J", "Kansas City"),
    (72, "2026-06-27", "Jordan",         "Argentina",                "J", "Dallas"),
]

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "fixtures_2026.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["match_number", "date", "stage", "group",
                         "team_a", "team_b", "venue"],
        )
        writer.writeheader()
        for mn, date, a, b, grp, venue in GROUP_MATCHES:
            writer.writerow({
                "match_number": mn, "date": date, "stage": "group",
                "group": grp, "team_a": a, "team_b": b, "venue": venue,
            })
    teams = sorted({t for ts in GROUPS.values() for t in ts})
    print(f"Wrote {len(GROUP_MATCHES)} group-stage matches to {OUT}")
    print(f"{len(teams)} teams across {len(GROUPS)} groups")


if __name__ == "__main__":
    main()
