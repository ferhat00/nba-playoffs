"""
Hardcoded 2025-26 NBA playoff seedings and realistic synthetic stats.

Used when the live nba_api endpoints are unavailable or rate-limited. Numbers
reflect the 2025-26 regular season arrangement (as seen on nba.com/playoffs/2026)
for the 16 playoff teams, rounded to plausible values. Rotation players are
generated procedurally from star-anchored distributions so every team has a
reasonable depth chart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EAST_SEEDS: List[tuple[int, str]] = [
    (1, "Detroit Pistons"),
    (2, "Boston Celtics"),
    (3, "New York Knicks"),
    (4, "Cleveland Cavaliers"),
    (5, "Toronto Raptors"),
    (6, "Atlanta Hawks"),
    (7, "Philadelphia 76ers"),
    (8, "Orlando Magic"),
]

WEST_SEEDS: List[tuple[int, str]] = [
    (1, "Oklahoma City Thunder"),
    (2, "San Antonio Spurs"),
    (3, "Denver Nuggets"),
    (4, "Los Angeles Lakers"),
    (5, "Houston Rockets"),
    (6, "Minnesota Timberwolves"),
    (7, "Portland Trail Blazers"),
    (8, "Phoenix Suns"),
]


@dataclass(frozen=True)
class TeamSeasonLine:
    team_id: str
    team_name: str
    seed: int
    conference: str
    off_rtg: float
    def_rtg: float
    pace: float
    w: int
    l: int
    three_pt_pct: float
    opp_three_pt_pct: float
    ast_to_tov: float
    oreb_pct: float
    clutch_net_rtg: float

    @property
    def net_rtg(self) -> float:
        return round(self.off_rtg - self.def_rtg, 2)


TEAM_LINES: List[TeamSeasonLine] = [
    # East — (id, name, seed, conf, off, def, pace, w, l, 3p%, opp3p%, ast/to, oreb%, clutch_net)
    TeamSeasonLine("DET", "Detroit Pistons", 1, "East", 119.5, 110.5, 99.0, 60, 22, 37.5, 34.5, 1.90, 27.5, 7.0),
    TeamSeasonLine("BOS", "Boston Celtics", 2, "East", 118.5, 111.5, 98.5, 56, 26, 37.8, 34.5, 1.85, 25.0, 5.5),
    TeamSeasonLine("NYK", "New York Knicks", 3, "East", 117.5, 112.0, 97.5, 53, 29, 36.5, 35.0, 1.80, 28.0, 4.5),
    TeamSeasonLine("CLE", "Cleveland Cavaliers", 4, "East", 117.0, 112.5, 98.5, 52, 30, 37.5, 34.5, 1.88, 27.0, 4.0),
    TeamSeasonLine("TOR", "Toronto Raptors", 5, "East", 114.5, 112.5, 100.0, 46, 36, 36.0, 35.0, 1.70, 26.0, 1.5),
    TeamSeasonLine("ATL", "Atlanta Hawks", 6, "East", 115.0, 113.5, 101.0, 46, 36, 36.5, 35.5, 1.70, 25.5, 0.5),
    TeamSeasonLine("PHI", "Philadelphia 76ers", 7, "East", 113.5, 112.5, 98.5, 45, 37, 36.0, 35.0, 1.75, 26.0, -0.5),
    TeamSeasonLine("ORL", "Orlando Magic", 8, "East", 112.0, 112.0, 97.5, 45, 37, 34.5, 34.5, 1.65, 27.5, 0.0),
    # West
    TeamSeasonLine("OKC", "Oklahoma City Thunder", 1, "West", 119.5, 106.5, 99.0, 64, 18, 37.5, 33.0, 2.00, 29.0, 10.0),
    TeamSeasonLine("SAS", "San Antonio Spurs", 2, "West", 118.0, 109.0, 99.5, 62, 20, 37.0, 34.0, 1.90, 27.0, 7.0),
    TeamSeasonLine("DEN", "Denver Nuggets", 3, "West", 118.0, 114.0, 99.5, 54, 28, 37.0, 35.5, 1.85, 27.0, 5.5),
    TeamSeasonLine("LAL", "Los Angeles Lakers", 4, "West", 117.0, 112.5, 98.5, 53, 29, 36.5, 35.0, 1.85, 26.5, 4.0),
    TeamSeasonLine("HOU", "Houston Rockets", 5, "West", 116.0, 111.5, 99.5, 52, 30, 36.0, 34.5, 1.80, 28.0, 3.5),
    TeamSeasonLine("MIN", "Minnesota Timberwolves", 6, "West", 115.0, 112.5, 98.5, 49, 33, 36.5, 34.5, 1.75, 26.0, 2.0),
    TeamSeasonLine("POR", "Portland Trail Blazers", 7, "West", 113.0, 113.0, 100.5, 42, 40, 35.5, 35.5, 1.65, 26.5, -0.5),
    TeamSeasonLine("PHX", "Phoenix Suns", 8, "West", 114.0, 114.0, 99.0, 45, 37, 36.0, 35.5, 1.70, 25.5, -1.0),
]


# Star players with realistic 2025-26 per-36 lines.
# Fields: (player_name, team_id, pts36, reb36, ast36, usg, bpm, injured)
STAR_PLAYERS: List[tuple[str, str, float, float, float, float, float, bool]] = [
    # East
    ("Cade Cunningham", "DET", 26.5, 6.0, 9.0, 30.5, 5.2, False),
    ("Jalen Duren", "DET", 13.5, 11.0, 2.5, 17.0, 2.5, False),
    ("Ausar Thompson", "DET", 13.0, 7.0, 3.5, 17.0, 2.5, False),
    ("Jayson Tatum", "BOS", 27.5, 8.5, 5.5, 30.0, 6.5, False),
    ("Jaylen Brown", "BOS", 23.0, 6.0, 4.8, 27.5, 3.2, False),
    ("Derrick White", "BOS", 16.0, 4.5, 4.5, 20.0, 3.0, False),
    ("Kristaps Porzingis", "BOS", 19.0, 7.5, 2.0, 24.0, 3.5, False),
    ("Jalen Brunson", "NYK", 26.5, 3.5, 7.3, 30.5, 4.5, False),
    ("Karl-Anthony Towns", "NYK", 24.0, 12.0, 3.0, 27.0, 4.0, False),
    ("OG Anunoby", "NYK", 18.0, 5.0, 2.5, 19.5, 2.8, False),
    ("Mikal Bridges", "NYK", 17.5, 3.5, 3.5, 19.0, 1.8, False),
    ("Donovan Mitchell", "CLE", 26.0, 4.5, 5.1, 29.0, 4.8, False),
    ("Darius Garland", "CLE", 20.5, 3.0, 7.0, 26.0, 3.0, False),
    ("Evan Mobley", "CLE", 19.5, 10.0, 3.5, 21.0, 5.2, False),
    ("Jarrett Allen", "CLE", 14.5, 11.0, 2.0, 16.0, 3.1, False),
    ("Scottie Barnes", "TOR", 20.5, 8.5, 6.0, 24.0, 3.5, False),
    ("RJ Barrett", "TOR", 19.5, 5.5, 4.0, 24.0, 1.5, False),
    ("Immanuel Quickley", "TOR", 17.5, 4.0, 6.0, 22.0, 2.0, False),
    ("Trae Young", "ATL", 23.0, 3.0, 11.0, 29.0, 2.5, False),
    ("Jalen Johnson", "ATL", 18.5, 9.0, 5.0, 22.0, 3.0, False),
    ("Dyson Daniels", "ATL", 13.5, 4.5, 4.5, 15.5, 2.0, False),
    ("Joel Embiid", "PHI", 27.5, 10.0, 4.0, 32.0, 6.0, True),
    ("Tyrese Maxey", "PHI", 25.0, 3.5, 7.0, 28.5, 3.5, False),
    ("Paul George", "PHI", 21.5, 5.5, 4.5, 26.0, 3.0, False),
    ("Paolo Banchero", "ORL", 25.5, 7.5, 5.0, 29.0, 3.8, False),
    ("Franz Wagner", "ORL", 22.5, 5.5, 4.5, 24.5, 3.0, False),
    ("Jalen Suggs", "ORL", 15.5, 4.0, 4.5, 20.0, 2.2, False),
    # West
    ("Shai Gilgeous-Alexander", "OKC", 32.5, 5.0, 6.0, 31.5, 8.5, False),
    ("Jalen Williams", "OKC", 21.5, 5.5, 5.0, 24.0, 3.5, False),
    ("Chet Holmgren", "OKC", 17.0, 8.0, 2.5, 20.5, 4.0, False),
    ("Luguentz Dort", "OKC", 11.0, 4.0, 1.8, 15.0, 1.4, False),
    ("Victor Wembanyama", "SAS", 26.5, 11.5, 4.0, 28.5, 7.5, False),
    ("De'Aaron Fox", "SAS", 23.5, 4.0, 6.5, 27.5, 3.5, False),
    ("Devin Vassell", "SAS", 17.0, 4.0, 3.5, 22.0, 1.8, False),
    ("Nikola Jokic", "DEN", 28.5, 12.5, 10.0, 29.5, 11.5, False),
    ("Jamal Murray", "DEN", 20.5, 4.0, 6.5, 25.5, 1.5, False),
    ("Aaron Gordon", "DEN", 14.0, 6.5, 3.5, 16.0, 2.0, False),
    ("Luka Doncic", "LAL", 30.0, 8.5, 8.5, 32.5, 7.5, False),
    ("LeBron James", "LAL", 23.5, 7.5, 8.5, 28.0, 5.2, False),
    ("Austin Reaves", "LAL", 18.5, 4.5, 5.5, 21.5, 2.5, False),
    ("Alperen Sengun", "HOU", 21.0, 10.0, 5.0, 26.0, 3.5, False),
    ("Jalen Green", "HOU", 21.5, 4.5, 3.5, 26.5, 1.2, False),
    ("Amen Thompson", "HOU", 15.0, 7.5, 4.0, 18.5, 2.8, False),
    ("Fred VanVleet", "HOU", 14.0, 3.8, 6.0, 19.0, 2.0, True),
    ("Anthony Edwards", "MIN", 27.0, 5.5, 5.0, 30.0, 4.5, False),
    ("Julius Randle", "MIN", 21.0, 8.0, 5.0, 25.5, 2.5, False),
    ("Rudy Gobert", "MIN", 12.5, 12.0, 1.5, 15.0, 2.8, False),
    ("Shaedon Sharpe", "POR", 20.0, 5.0, 3.5, 25.0, 1.0, False),
    ("Deni Avdija", "POR", 17.5, 6.5, 4.0, 21.0, 2.0, False),
    ("Donovan Clingan", "POR", 12.0, 10.5, 2.0, 16.0, 2.5, False),
    ("Devin Booker", "PHX", 26.0, 4.5, 6.5, 30.0, 3.5, False),
    ("Kevin Durant", "PHX", 27.0, 6.5, 4.5, 30.0, 5.0, False),
    ("Bradley Beal", "PHX", 18.5, 4.0, 4.0, 23.0, 1.2, False),
]


def _synthesize_rotation(team_id: str, rng: np.random.Generator, n: int = 5) -> List[dict]:
    """Procedurally generate plausible rotation player lines for a team."""
    def _clip(value: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, value)))

    out = []
    for i in range(n):
        out.append(
            {
                "player_id": f"{team_id}_rot_{i}",
                "player_name": f"{team_id} Role Player {i + 1}",
                "team_id": team_id,
                "pts_per36": _clip(rng.normal(10.5, 3.0), 3.0, 20.0),
                "reb_per36": _clip(rng.normal(5.0, 1.8), 1.5, 11.0),
                "ast_per36": _clip(rng.normal(2.5, 1.2), 0.5, 7.0),
                "usg_pct": _clip(rng.normal(15.0, 3.0), 8.0, 24.0),
                "bpm": _clip(rng.normal(-0.5, 1.4), -4.0, 3.0),
                "injured": bool(rng.random() < 0.05),
            }
        )
    return out


def _synthesize_game_log(line: TeamSeasonLine, rng: np.random.Generator, n: int = 20) -> List[dict]:
    """Generate a plausible last-N regular season game log for a team."""
    games = []
    for i in range(n):
        off = float(rng.normal(line.off_rtg, 4.5))
        df = float(rng.normal(line.def_rtg, 4.5))
        pc = float(rng.normal(line.pace, 2.0))
        margin = off - df
        # Slight noise + home court effect drawn from a binary coin.
        home = bool(rng.random() < 0.5)
        if home:
            margin += 3.0
        final_margin = margin + rng.normal(0, 5.5)
        won = final_margin > 0
        games.append(
            {
                "game_idx": i,
                "off_rtg": off,
                "def_rtg": df,
                "pace": pc,
                "home": home,
                "won": bool(won),
                "margin": round(float(final_margin), 1),
            }
        )
    return games


def build_team_stats_df(seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame matching the fetcher's get_team_stats() contract."""
    rng = np.random.default_rng(seed)
    rows = []
    for line in TEAM_LINES:
        rows.append(
            {
                "team_id": line.team_id,
                "team_name": line.team_name,
                "seed": line.seed,
                "conference": line.conference,
                "off_rtg": line.off_rtg,
                "def_rtg": line.def_rtg,
                "pace": line.pace,
                "net_rtg": line.net_rtg,
                "w": line.w,
                "l": line.l,
                "three_pt_pct": line.three_pt_pct,
                "opp_three_pt_pct": line.opp_three_pt_pct,
                "ast_to_tov": line.ast_to_tov,
                "oreb_pct": line.oreb_pct,
                "clutch_net_rtg": line.clutch_net_rtg,
                "last20_game_log": _synthesize_game_log(line, rng),
            }
        )
    df = pd.DataFrame(rows)
    logger.debug("Built fallback team stats: %d teams", len(df))
    return df


def build_player_stats_df(seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame matching the fetcher's get_player_stats() contract."""
    rng = np.random.default_rng(seed + 1)
    rows: List[dict] = []
    # Star players first.
    for name, team_id, pts, reb, ast, usg, bpm, injured in STAR_PLAYERS:
        rows.append(
            {
                "player_id": f"{team_id}_{name.replace(' ', '_').replace('.', '')}",
                "player_name": name,
                "team_id": team_id,
                "pts_per36": float(pts),
                "reb_per36": float(reb),
                "ast_per36": float(ast),
                "usg_pct": float(usg),
                "bpm": float(bpm),
                "injured": bool(injured),
            }
        )
    # Rotation fill.
    for line in TEAM_LINES:
        rows.extend(_synthesize_rotation(line.team_id, rng, n=5))
    df = pd.DataFrame(rows)
    logger.debug("Built fallback player stats: %d players", len(df))
    return df


def get_bracket_structure() -> Dict[str, List[tuple[int, int]]]:
    """Return first-round matchups keyed by conference as (high_seed, low_seed)."""
    return {
        "East": [(1, 8), (4, 5), (3, 6), (2, 7)],
        "West": [(1, 8), (4, 5), (3, 6), (2, 7)],
    }


def get_seed_to_team_id() -> Dict[str, Dict[int, str]]:
    """Map (conference, seed) -> team_id for the current fallback bracket."""
    out: Dict[str, Dict[int, str]] = {"East": {}, "West": {}}
    for line in TEAM_LINES:
        out[line.conference][line.seed] = line.team_id
    return out
