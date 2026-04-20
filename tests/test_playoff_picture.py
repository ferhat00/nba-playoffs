"""Unit tests for the PlayoffPicture-backed seed discovery in fetcher."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nba_playoff_predictor.data import fetcher  # noqa: E402


# NBA team ids pulled straight from fetcher.FULL_NBA_TEAM_ID_MAP.
_ABBR_TO_ID = {v: k for k, v in fetcher.FULL_NBA_TEAM_ID_MAP.items()}


def _matchup_frame(conf: str, pairs: list[tuple[int, str, int, str]]) -> pd.DataFrame:
    rows = []
    for hi_seed, hi_abbr, lo_seed, lo_abbr in pairs:
        rows.append(
            {
                "CONFERENCE": conf,
                "HIGH_SEED_RANK": hi_seed,
                "HIGH_SEED_TEAM": hi_abbr,
                "HIGH_SEED_TEAM_ID": _ABBR_TO_ID[hi_abbr],
                "LOW_SEED_RANK": lo_seed,
                "LOW_SEED_TEAM": lo_abbr,
                "LOW_SEED_TEAM_ID": _ABBR_TO_ID[lo_abbr],
            }
        )
    return pd.DataFrame(rows)


EAST_2026_PAIRS = [
    (1, "DET", 8, "ORL"),
    (2, "BOS", 7, "PHI"),
    (3, "NYK", 6, "ATL"),
    (4, "CLE", 5, "TOR"),
]

WEST_2026_PAIRS = [
    (1, "OKC", 8, "POR"),
    (2, "SAS", 7, "PHX"),
    (3, "DEN", 6, "MIN"),
    (4, "LAL", 5, "HOU"),
]


def _picture_2026() -> dict[str, pd.DataFrame]:
    return {
        "east_matchups": _matchup_frame("East", EAST_2026_PAIRS),
        "west_matchups": _matchup_frame("West", WEST_2026_PAIRS),
    }


def test_parse_playoff_picture_returns_seed_map_and_pairs():
    seed_map, pairs = fetcher._parse_playoff_picture(_picture_2026())

    assert seed_map["East"][1] == "DET"
    assert seed_map["East"][8] == "ORL"
    assert seed_map["East"][4] == "CLE"
    assert seed_map["East"][5] == "TOR"
    assert seed_map["West"][1] == "OKC"
    assert seed_map["West"][2] == "SAS"

    # Each conference has exactly 8 seeds 1..8
    assert sorted(seed_map["East"].keys()) == list(range(1, 9))
    assert sorted(seed_map["West"].keys()) == list(range(1, 9))

    # Pairs preserve the (high, low) seed pairings.
    assert (1, 8) in pairs["East"]
    assert (4, 5) in pairs["East"]
    assert (2, 7) in pairs["West"]


def test_get_playoff_seed_map_prefers_playoff_picture():
    fetcher.reset_seed_cache()
    with patch.object(fetcher, "_nba_api_available", return_value=True), \
         patch.object(fetcher, "_fetch_playoff_picture", return_value=_picture_2026()):
        seed_map, pairs = fetcher.get_playoff_seed_map(season="2025-26", return_pairs=True)

    assert seed_map["East"][1] == "DET"
    assert seed_map["West"][8] == "POR"
    assert pairs is not None
    assert {(1, 8), (2, 7), (3, 6), (4, 5)} == set(pairs["East"])


def test_get_playoff_seed_map_falls_back_to_standings_rank():
    fetcher.reset_seed_cache()

    # Simulate: PlayoffPicture raises, but LeagueDashTeamStats returns a usable frame.
    raw = pd.DataFrame(
        [
            {"TEAM_ID": _ABBR_TO_ID[abbr], "W": wins, "L": 82 - wins}
            for abbr, wins in [
                ("DET", 60), ("BOS", 56), ("NYK", 53), ("CLE", 52),
                ("TOR", 46), ("ATL", 46), ("PHI", 45), ("ORL", 45),
                ("OKC", 64), ("SAS", 62), ("DEN", 54), ("LAL", 53),
                ("HOU", 52), ("MIN", 49), ("PHX", 45), ("POR", 42),
            ]
        ]
    )

    with patch.object(fetcher, "_nba_api_available", return_value=True), \
         patch.object(fetcher, "_fetch_playoff_picture", side_effect=RuntimeError("boom")), \
         patch.object(fetcher, "_fetch_raw_team_stats", return_value=raw):
        seed_map, pairs = fetcher.get_playoff_seed_map(season="2025-26", return_pairs=True)

    assert seed_map["East"][1] == "DET"
    assert seed_map["West"][1] == "OKC"
    assert pairs is None  # Derived bracket has no authoritative pairings.


def test_get_playoff_seed_map_falls_back_to_hardcoded_when_api_unavailable():
    fetcher.reset_seed_cache()
    with patch.object(fetcher, "_nba_api_available", return_value=False):
        seed_map = fetcher.get_playoff_seed_map(season="2025-26")

    # Hardcoded fallback should match the 2025-26 arrangement in fallback_data.
    assert seed_map["East"][1] == "DET"
    assert seed_map["East"][8] == "ORL"
    assert seed_map["West"][1] == "OKC"
