"""
Live data fetchers for the 2024-25 NBA regular season.

Wraps nba_api endpoints with joblib-backed caching, exponential backoff, and a
graceful fallback to the hardcoded 2025 dataset so the pipeline is always
runnable. The public API is intentionally small:

    get_team_stats()    -> pd.DataFrame
    get_player_stats()  -> pd.DataFrame

Both frames match the contracts documented in the project README.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from joblib import Memory

from . import fallback_data

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_memory = Memory(location=str(_CACHE_DIR), verbose=0)

SEASON = "2024-25"
SEASON_TYPE = "Regular Season"

# Official NBA team_id integers for the 16 playoff teams. Used to translate
# between nba_api's numeric ids and the project's string ids (fallback_data).
NBA_TEAM_ID_MAP = {
    1610612739: "CLE",
    1610612738: "BOS",
    1610612752: "NYK",
    1610612754: "IND",
    1610612749: "MIL",
    1610612765: "DET",
    1610612737: "ATL",
    1610612748: "MIA",
    1610612760: "OKC",
    1610612745: "HOU",
    1610612747: "LAL",
    1610612743: "DEN",
    1610612746: "LAC",
    1610612744: "GSW",
    1610612763: "MEM",
    1610612742: "DAL",
}


def _retry_with_backoff(fn: Callable, *args, max_retries: int = 3, base_delay: float = 1.0, **kwargs):
    """Call `fn(*args, **kwargs)` with exponential backoff on transient errors."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # nba_api raises generic exceptions on 429/5xx
            last_exc = exc
            delay = base_delay * (2**attempt)
            logger.warning(
                "nba_api call failed (attempt %d/%d): %s. Retrying in %.1fs.",
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _nba_api_available() -> bool:
    try:
        import nba_api  # noqa: F401

        return True
    except ImportError:
        logger.info("nba_api is not installed; using fallback dataset.")
        return False


@_memory.cache
def _fetch_raw_team_stats() -> pd.DataFrame:
    from nba_api.stats.endpoints import LeagueDashTeamStats

    logger.info("Fetching LeagueDashTeamStats (advanced) for %s...", SEASON)
    adv = _retry_with_backoff(
        LeagueDashTeamStats,
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
        measure_type_detailed_defense="Advanced",
    )
    adv_df = adv.get_data_frames()[0]
    return adv_df


@_memory.cache
def _fetch_raw_player_stats() -> pd.DataFrame:
    from nba_api.stats.endpoints import LeagueDashPlayerStats

    logger.info("Fetching LeagueDashPlayerStats (per36) for %s...", SEASON)
    per36 = _retry_with_backoff(
        LeagueDashPlayerStats,
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
        per_mode_detailed="Per36",
    )
    per36_df = per36.get_data_frames()[0]

    logger.info("Fetching LeagueDashPlayerStats (advanced) for %s...", SEASON)
    adv = _retry_with_backoff(
        LeagueDashPlayerStats,
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
        measure_type_detailed_defense="Advanced",
    )
    adv_df = adv.get_data_frames()[0]

    merged = per36_df.merge(
        adv_df[["PLAYER_ID", "USG_PCT", "NET_RATING"]],
        on="PLAYER_ID",
        how="left",
        suffixes=("", "_adv"),
    )
    return merged


@_memory.cache
def _fetch_team_game_log(nba_team_id: int) -> pd.DataFrame:
    from nba_api.stats.endpoints import TeamGameLog

    log = _retry_with_backoff(
        TeamGameLog,
        team_id=nba_team_id,
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
    )
    return log.get_data_frames()[0]


def _game_log_to_records(raw: pd.DataFrame, pace: float) -> List[dict]:
    """Convert a raw TeamGameLog frame into the 20-game summary structure."""
    if raw is None or raw.empty:
        return []
    recent = raw.head(20).copy()
    # TeamGameLog does not expose per-game ratings directly; approximate.
    records = []
    for i, row in enumerate(recent.itertuples(index=False)):
        pts_for = float(getattr(row, "PTS", np.nan))
        pts_against = float(getattr(row, "PTS", np.nan)) - float(getattr(row, "PLUS_MINUS", 0.0))
        poss_est = max(85.0, pace * 48.0 / 48.0)  # approximate possessions/game
        off_rtg = 100.0 * pts_for / poss_est if poss_est else np.nan
        def_rtg = 100.0 * pts_against / poss_est if poss_est else np.nan
        matchup = str(getattr(row, "MATCHUP", ""))
        home = " vs. " in matchup  # "@" indicates away
        won = str(getattr(row, "WL", "")).upper() == "W"
        records.append(
            {
                "game_idx": i,
                "off_rtg": off_rtg,
                "def_rtg": def_rtg,
                "pace": pace,
                "home": home,
                "won": won,
            }
        )
    return records


def _normalize_team_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Trim the raw LeagueDashTeamStats frame to playoff teams and canonical cols."""
    keep_ids = set(NBA_TEAM_ID_MAP.keys())
    sub = raw[raw["TEAM_ID"].isin(keep_ids)].copy()
    sub["team_id"] = sub["TEAM_ID"].map(NBA_TEAM_ID_MAP)

    seed_map = fallback_data.get_seed_to_team_id()
    reverse_seed = {}
    for conf, seeds in seed_map.items():
        for seed, tid in seeds.items():
            reverse_seed[tid] = (conf, seed)

    sub["conference"] = sub["team_id"].map(lambda t: reverse_seed[t][0])
    sub["seed"] = sub["team_id"].map(lambda t: reverse_seed[t][1])

    out = pd.DataFrame(
        {
            "team_id": sub["team_id"].values,
            "team_name": sub["TEAM_NAME"].values,
            "seed": sub["seed"].values,
            "conference": sub["conference"].values,
            "off_rtg": sub.get("OFF_RATING", pd.Series(dtype=float)).values,
            "def_rtg": sub.get("DEF_RATING", pd.Series(dtype=float)).values,
            "pace": sub.get("PACE", pd.Series(dtype=float)).values,
            "net_rtg": sub.get("NET_RATING", pd.Series(dtype=float)).values,
            "w": sub.get("W", pd.Series(dtype=int)).values,
            "l": sub.get("L", pd.Series(dtype=int)).values,
        }
    )
    return out


def get_team_stats() -> pd.DataFrame:
    """
    Return a DataFrame of regular-season team statistics for the 16 playoff teams.

    Schema:
        team_id, team_name, seed, conference,
        off_rtg, def_rtg, pace, net_rtg, w, l,
        last20_game_log (list of dicts)
    """
    if not _nba_api_available():
        return fallback_data.build_team_stats_df()

    try:
        raw = _fetch_raw_team_stats()
        df = _normalize_team_frame(raw)

        logs: List[List[dict]] = []
        inverse = {v: k for k, v in NBA_TEAM_ID_MAP.items()}
        for _, row in df.iterrows():
            nba_id = inverse[row["team_id"]]
            try:
                raw_log = _fetch_team_game_log(int(nba_id))
                logs.append(_game_log_to_records(raw_log, float(row["pace"])))
            except Exception as exc:
                logger.warning("Game log fetch failed for %s: %s", row["team_id"], exc)
                logs.append([])
        df["last20_game_log"] = logs
        logger.info("Fetched team stats for %d teams via nba_api", len(df))
        return df
    except Exception as exc:
        logger.warning("Falling back to hardcoded team stats: %s", exc)
        return fallback_data.build_team_stats_df()


def _normalize_player_frame(raw: pd.DataFrame) -> pd.DataFrame:
    keep_ids = set(NBA_TEAM_ID_MAP.keys())
    sub = raw[raw["TEAM_ID"].isin(keep_ids)].copy()
    sub["team_id"] = sub["TEAM_ID"].map(NBA_TEAM_ID_MAP)

    # Minutes threshold to avoid end-of-bench noise.
    if "MIN" in sub.columns:
        sub = sub[sub["MIN"] >= 10.0]

    out = pd.DataFrame(
        {
            "player_id": sub["PLAYER_ID"].astype(str).values,
            "player_name": sub.get("PLAYER_NAME", pd.Series(dtype=str)).values,
            "team_id": sub["team_id"].values,
            "pts_per36": sub.get("PTS", pd.Series(dtype=float)).values,
            "reb_per36": sub.get("REB", pd.Series(dtype=float)).values,
            "ast_per36": sub.get("AST", pd.Series(dtype=float)).values,
            "usg_pct": (sub.get("USG_PCT", pd.Series(dtype=float)) * 100.0).values,
            "bpm": sub.get("NET_RATING", pd.Series(dtype=float)).values / 5.0,
            "injured": False,
        }
    )
    return out


def _augment_injuries(df: pd.DataFrame) -> pd.DataFrame:
    """Mark high-usage players injured if an injury signal is available.

    Uses the `PlayerInjurySummary` endpoint when present. Failures degrade to the
    status quo frame (no injury flags set).
    """
    try:
        from nba_api.stats.endpoints import PlayerInjurySummary  # type: ignore

        logger.info("Fetching PlayerInjurySummary...")
        raw = _retry_with_backoff(PlayerInjurySummary)
        inj_df = raw.get_data_frames()[0]
        injured_ids = set(inj_df["PLAYER_ID"].astype(str).tolist())
        df["injured"] = df["player_id"].isin(injured_ids)
        logger.info("Marked %d players as injured", int(df["injured"].sum()))
    except Exception as exc:
        logger.warning("PlayerInjurySummary unavailable (%s); no injuries applied", exc)
    return df


def get_player_stats() -> pd.DataFrame:
    """
    Return a DataFrame of per-36 + usage for players on the 16 playoff teams.

    Schema:
        player_id, team_id, pts_per36, reb_per36, ast_per36,
        usg_pct, bpm, injured (bool)
    """
    if not _nba_api_available():
        return fallback_data.build_player_stats_df()

    try:
        raw = _fetch_raw_player_stats()
        df = _normalize_player_frame(raw)
        df = _augment_injuries(df)
        logger.info("Fetched player stats for %d players via nba_api", len(df))
        return df
    except Exception as exc:
        logger.warning("Falling back to hardcoded player stats: %s", exc)
        return fallback_data.build_player_stats_df()


def clear_cache() -> None:
    """Remove all joblib cache entries."""
    _memory.clear(warn=False)
    logger.info("Cleared nba_api cache at %s", _CACHE_DIR)
