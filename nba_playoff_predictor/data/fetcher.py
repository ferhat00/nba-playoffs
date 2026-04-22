"""
Live data fetchers for the NBA regular season.

Dynamically determines the current NBA season from the calendar date and
discovers playoff teams from league standings.  Supports fetching historical
seasons (last 5 years) for matchup-model training data.

Wraps nba_api endpoints with joblib-backed caching, exponential backoff, and a
graceful fallback to the hardcoded dataset so the pipeline is always runnable.

Public API:
    current_nba_season()             -> str
    current_playoff_year()           -> int
    get_team_stats(season=None)      -> pd.DataFrame
    get_player_stats(season=None)    -> pd.DataFrame
    get_playoff_seed_map(season=None)-> dict
    get_series_results(season=None)  -> dict  (balldontlie + ESPN fallback)
    get_historical_matchup_data()    -> pd.DataFrame
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Memory

from . import fallback_data

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_memory = Memory(location=str(_CACHE_DIR), verbose=0)


# ---------------------------------------------------------------------------
# Season utilities
# ---------------------------------------------------------------------------

def current_nba_season() -> str:
    """Return the NBA season string for the current date, e.g. ``'2025-26'``.

    The NBA season starts in October; games from January through June belong
    to the season that began the previous October.
    """
    today = date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    end_short = (start_year + 1) % 100
    return f"{start_year}-{end_short:02d}"


def current_playoff_year() -> int:
    """Calendar year of the current / upcoming playoffs (Apr-Jun window)."""
    today = date.today()
    return today.year + 1 if today.month >= 10 else today.year


def historical_seasons(n_years: int = 5) -> List[str]:
    """Return the previous *n_years* season strings, most recent first."""
    current = current_nba_season()
    start_year = int(current.split("-")[0])
    out: List[str] = []
    for i in range(1, n_years + 1):
        yr = start_year - i
        end_short = (yr + 1) % 100
        out.append(f"{yr}-{end_short:02d}")
    return out


# ---------------------------------------------------------------------------
# Team / conference maps  (all 30 NBA teams)
# ---------------------------------------------------------------------------

FULL_NBA_TEAM_ID_MAP: Dict[int, str] = {
    1610612737: "ATL", 1610612738: "BOS", 1610612751: "BKN", 1610612766: "CHA",
    1610612741: "CHI", 1610612739: "CLE", 1610612742: "DAL", 1610612743: "DEN",
    1610612765: "DET", 1610612744: "GSW", 1610612745: "HOU", 1610612754: "IND",
    1610612746: "LAC", 1610612747: "LAL", 1610612763: "MEM", 1610612748: "MIA",
    1610612749: "MIL", 1610612750: "MIN", 1610612740: "NOP", 1610612752: "NYK",
    1610612760: "OKC", 1610612753: "ORL", 1610612755: "PHI", 1610612756: "PHX",
    1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS", 1610612761: "TOR",
    1610612762: "UTA", 1610612764: "WAS",
}

FULL_NBA_ABBREV_TO_ID: Dict[str, int] = {v: k for k, v in FULL_NBA_TEAM_ID_MAP.items()}

# Conference affiliations (stable since 2004-05)
CONFERENCE_MAP: Dict[str, str] = {
    "ATL": "East", "BOS": "East", "BKN": "East", "CHA": "East", "CHI": "East",
    "CLE": "East", "DET": "East", "IND": "East", "MIA": "East", "MIL": "East",
    "NYK": "East", "ORL": "East", "PHI": "East", "TOR": "East", "WAS": "East",
    "DAL": "West", "DEN": "West", "GSW": "West", "HOU": "West", "LAC": "West",
    "LAL": "West", "MEM": "West", "MIN": "West", "NOP": "West", "OKC": "West",
    "PHX": "West", "POR": "West", "SAC": "West", "SAS": "West", "UTA": "West",
}

# Backward-compatible alias used by older call-sites.
NBA_TEAM_ID_MAP = FULL_NBA_TEAM_ID_MAP


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
def _fetch_raw_team_stats(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import LeagueDashTeamStats

    logger.info("Fetching LeagueDashTeamStats (advanced) for %s...", season)
    adv = _retry_with_backoff(
        LeagueDashTeamStats,
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
    )
    return adv.get_data_frames()[0]


@_memory.cache
def _fetch_raw_team_base_stats(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import LeagueDashTeamStats

    logger.info("Fetching LeagueDashTeamStats (base) for %s...", season)
    base = _retry_with_backoff(
        LeagueDashTeamStats,
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Base",
    )
    return base.get_data_frames()[0]


@_memory.cache
def _fetch_raw_team_clutch_stats(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import LeagueDashTeamClutch

    logger.info("Fetching LeagueDashTeamClutch for %s...", season)
    clutch = _retry_with_backoff(
        LeagueDashTeamClutch,
        season=season,
        season_type_all_star="Regular Season",
    )
    return clutch.get_data_frames()[0]


@_memory.cache
def _fetch_raw_player_stats(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import LeagueDashPlayerStats

    logger.info("Fetching LeagueDashPlayerStats (per36) for %s...", season)
    per36 = _retry_with_backoff(
        LeagueDashPlayerStats,
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="Per36",
    )
    per36_df = per36.get_data_frames()[0]

    logger.info("Fetching LeagueDashPlayerStats (advanced) for %s...", season)
    adv = _retry_with_backoff(
        LeagueDashPlayerStats,
        season=season,
        season_type_all_star="Regular Season",
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
def _fetch_team_game_log(nba_team_id: int, season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import TeamGameLog

    log = _retry_with_backoff(
        TeamGameLog,
        team_id=nba_team_id,
        season=season,
        season_type_all_star="Regular Season",
    )
    return log.get_data_frames()[0]


@_memory.cache
def _fetch_playoff_game_log(season: str) -> pd.DataFrame:
    """Fetch all playoff games for *season* (each game appears twice, once per team)."""
    from nba_api.stats.endpoints import LeagueGameLog

    logger.info("Fetching playoff game log for %s...", season)
    raw = _retry_with_backoff(
        LeagueGameLog,
        season=season,
        season_type_all_star="Playoffs",
    )
    return raw.get_data_frames()[0]


def _season_id_regular(season: str) -> str:
    """Convert a season string like ``'2025-26'`` to an NBA stats ``SeasonID``.

    The stats API expects ``"2" + start_year`` for the regular season
    (e.g. ``"22025"`` for 2025-26).
    """
    return "2" + season.split("-")[0]


@_memory.cache
def _fetch_playoff_picture(season: str) -> Dict[str, pd.DataFrame]:
    """Fetch the authoritative playoff bracket via ``PlayoffPicture``.

    Returns a dict of four DataFrames — matchups and standings for both
    conferences — with the play-in results already applied to the 7/8 seeds.
    """
    from nba_api.stats.endpoints import PlayoffPicture

    logger.info("Fetching PlayoffPicture for %s...", season)
    pp = _retry_with_backoff(PlayoffPicture, season_id=_season_id_regular(season))
    frames = pp.get_data_frames()

    # The live API does not return frames in the order the nba_api library's
    # `expected_data` declares. Classify each frame by its columns + CONFERENCE
    # value so downstream code is order-independent.
    result: Dict[str, pd.DataFrame] = {}
    for df in frames:
        cols = set(df.columns)
        if "HIGH_SEED_RANK" in cols and "CONFERENCE" in cols:
            conf = str(df["CONFERENCE"].iloc[0]) if not df.empty else None
            key = "east_matchups" if conf == "East" else "west_matchups"
            result[key] = df
        elif "RANK" in cols and "TEAM_ID" in cols and "CONFERENCE" in cols:
            conf = str(df["CONFERENCE"].iloc[0]) if not df.empty else None
            key = "east_standings" if conf == "East" else "west_standings"
            result[key] = df
    return result


def _parse_playoff_picture(
    pp: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, Dict[int, str]], Dict[str, List[Tuple[int, int]]]]:
    """Return ``(seed_map, round1_pairs)`` extracted from a PlayoffPicture dict."""
    seed_map: Dict[str, Dict[int, str]] = {"East": {}, "West": {}}
    pairs: Dict[str, List[Tuple[int, int]]] = {"East": [], "West": []}
    for conf, key in (("East", "east_matchups"), ("West", "west_matchups")):
        df = pp.get(key)
        if df is None or df.empty:
            raise RuntimeError(f"Empty {conf} PlayoffPicture matchups")
        for _, row in df.iterrows():
            hi = int(row["HIGH_SEED_RANK"])
            lo = int(row["LOW_SEED_RANK"])
            hi_id = int(row["HIGH_SEED_TEAM_ID"])
            lo_id = int(row["LOW_SEED_TEAM_ID"])
            hi_abbr = FULL_NBA_TEAM_ID_MAP.get(hi_id)
            lo_abbr = FULL_NBA_TEAM_ID_MAP.get(lo_id)
            if hi_abbr is None or lo_abbr is None:
                raise RuntimeError(f"Unknown team id in {conf} matchup: {hi_id}, {lo_id}")
            seed_map[conf][hi] = hi_abbr
            seed_map[conf][lo] = lo_abbr
            pairs[conf].append((hi, lo))
        if sorted(seed_map[conf].keys()) != list(range(1, 9)):
            raise RuntimeError(f"{conf} PlayoffPicture missing seeds: {seed_map[conf]}")
    return seed_map, pairs


# ---------------------------------------------------------------------------
# Playoff team discovery
# ---------------------------------------------------------------------------

def _discover_playoff_teams(
    raw_adv_stats: pd.DataFrame,
) -> Tuple[Dict[int, str], Dict[str, Dict[int, str]]]:
    """Determine the top-8 teams per conference from advanced team stats.

    Returns
    -------
    playoff_ids : dict[int, str]
        ``{nba_int_id: abbreviation}`` for the 16 playoff teams.
    seed_map : dict[str, dict[int, str]]
        ``{"East": {1: "BOS", ...}, "West": {1: "OKC", ...}}``
    """
    df = raw_adv_stats.copy()
    df["abbrev"] = df["TEAM_ID"].map(FULL_NBA_TEAM_ID_MAP)
    df["conference"] = df["abbrev"].map(CONFERENCE_MAP)
    df = df.dropna(subset=["abbrev", "conference"])

    playoff_ids: Dict[int, str] = {}
    seed_map: Dict[str, Dict[int, str]] = {"East": {}, "West": {}}

    for conf in ("East", "West"):
        conf_teams = (
            df[df["conference"] == conf]
            .sort_values("W", ascending=False)
            .head(8)
            .reset_index(drop=True)
        )
        for idx, row in conf_teams.iterrows():
            seed = int(idx) + 1
            tid = row["abbrev"]
            nba_id = int(row["TEAM_ID"])
            playoff_ids[nba_id] = tid
            seed_map[conf][seed] = tid

    return playoff_ids, seed_map


# Per-season cache so multiple callers see the same bracket.
_cached_seed_map: Optional[Dict[str, Dict[int, str]]] = None
_cached_playoff_ids: Optional[Dict[int, str]] = None
_cached_round1_pairs: Optional[Dict[str, List[Tuple[int, int]]]] = None
_cached_season: Optional[str] = None


def reset_seed_cache() -> None:
    """Clear the in-memory seed/playoff-ids cache.

    joblib's on-disk cache is cleared separately via :func:`clear_cache`. This
    resets the process-local memo so a subsequent call re-fetches.
    """
    global _cached_seed_map, _cached_playoff_ids, _cached_round1_pairs, _cached_season
    _cached_seed_map = None
    _cached_playoff_ids = None
    _cached_round1_pairs = None
    _cached_season = None


def _ensure_playoff_discovery(
    season: str,
    raw_stats: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[int, str], Dict[str, Dict[int, str]]]:
    """Return ``(playoff_ids, seed_map)``, fetching and caching if needed.

    Prefers the authoritative :class:`PlayoffPicture` endpoint (which reflects
    play-in results and official tiebreakers). Falls back to W-rank from
    ``LeagueDashTeamStats`` when the bracket endpoint is unavailable (e.g. prior
    to the play-in concluding).
    """
    global _cached_seed_map, _cached_playoff_ids, _cached_round1_pairs, _cached_season
    if _cached_season == season and _cached_playoff_ids is not None:
        return _cached_playoff_ids, _cached_seed_map  # type: ignore[return-value]

    seed_map: Optional[Dict[str, Dict[int, str]]] = None
    pairs: Optional[Dict[str, List[Tuple[int, int]]]] = None
    try:
        pp = _fetch_playoff_picture(season)
        seed_map, pairs = _parse_playoff_picture(pp)
        logger.info("Playoff teams resolved via PlayoffPicture for %s", season)
    except Exception as exc:
        logger.warning("PlayoffPicture unavailable (%s); using W-rank fallback.", exc)

    if seed_map is None:
        if raw_stats is None:
            raw_stats = _fetch_raw_team_stats(season)
        _, seed_map = _discover_playoff_teams(raw_stats)

    inverse_abbrev = {v: k for k, v in FULL_NBA_TEAM_ID_MAP.items()}
    playoff_ids: Dict[int, str] = {}
    for conf_seeds in seed_map.values():
        for abbr in conf_seeds.values():
            nba_id = inverse_abbrev.get(abbr)
            if nba_id is not None:
                playoff_ids[nba_id] = abbr

    _cached_playoff_ids = playoff_ids
    _cached_seed_map = seed_map
    _cached_round1_pairs = pairs
    _cached_season = season
    return playoff_ids, seed_map


def _game_log_to_records(raw: pd.DataFrame, pace: float) -> List[dict]:
    """Convert a raw TeamGameLog frame into the 20-game summary structure."""
    if raw is None or raw.empty:
        return []
    recent = raw.head(20).copy()
    # TeamGameLog does not expose per-game ratings directly; approximate.
    records = []
    for i, row in enumerate(recent.itertuples(index=False)):
        pts_for = float(getattr(row, "PTS", np.nan))
        plus_minus = float(getattr(row, "PLUS_MINUS", 0.0))
        pts_against = pts_for - plus_minus
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
                "margin": round(plus_minus, 1),
            }
        )
    return records


def _normalize_team_frame(
    raw: pd.DataFrame,
    playoff_ids: Dict[int, str],
    seed_map: Dict[str, Dict[int, str]],
) -> pd.DataFrame:
    """Trim the raw LeagueDashTeamStats frame to playoff teams and canonical cols."""
    keep_ids = set(playoff_ids.keys())
    sub = raw[raw["TEAM_ID"].isin(keep_ids)].copy()
    sub["team_id"] = sub["TEAM_ID"].map(playoff_ids)

    reverse_seed: Dict[str, Tuple[str, int]] = {}
    for conf, seeds in seed_map.items():
        for seed, tid in seeds.items():
            reverse_seed[tid] = (conf, seed)

    sub["conference"] = sub["team_id"].map(lambda t: reverse_seed.get(t, ("East", 8))[0])
    sub["seed"] = sub["team_id"].map(lambda t: reverse_seed.get(t, ("East", 8))[1])

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


def _merge_base_stats(
    df: pd.DataFrame, season: str, playoff_ids: Dict[int, str],
) -> pd.DataFrame:
    """Merge base shooting / assist / rebound stats into the team frame."""
    try:
        raw = _fetch_raw_team_base_stats(season)
        keep_ids = set(playoff_ids.keys())
        sub = raw[raw["TEAM_ID"].isin(keep_ids)].copy()
        sub["team_id"] = sub["TEAM_ID"].map(playoff_ids)

        # 3P% — FG3_PCT is a 0-1 fraction; convert to percentage
        fg3_pct = sub.set_index("team_id").get("FG3_PCT", pd.Series(dtype=float))
        df["three_pt_pct"] = df["team_id"].map(fg3_pct).fillna(35.0) * 100.0
        # Some endpoints already return as percentage; clamp to sensible range
        df.loc[df["three_pt_pct"] > 100.0, "three_pt_pct"] = df["three_pt_pct"] / 100.0
        df.loc[df["three_pt_pct"] < 1.0, "three_pt_pct"] = df["three_pt_pct"] * 100.0

        # AST / TOV ratio
        ast = sub.set_index("team_id").get("AST", pd.Series(dtype=float))
        tov = sub.set_index("team_id").get("TOV", pd.Series(dtype=float))
        ast_tov = (ast / tov.replace(0, np.nan)).fillna(1.5)
        df["ast_to_tov"] = df["team_id"].map(ast_tov).fillna(1.5)

        # OREB%
        oreb = sub.set_index("team_id").get("OREB", pd.Series(dtype=float))
        # Approximate oreb% as oreb / (oreb + opp_dreb); fallback to raw count percentile
        df["oreb_pct"] = df["team_id"].map(oreb).fillna(10.0)
        # Normalize to a ~25-30 range if raw counts (>50 means per-game, <50 means rate)
        if df["oreb_pct"].mean() > 50:
            df["oreb_pct"] = df["oreb_pct"].rank(pct=True) * 5.0 + 24.0

        logger.info("Merged base team stats (3P%%, AST/TO, OREB)")
    except Exception as exc:
        logger.warning("Failed to fetch base team stats (%s); using defaults", exc)
        df["three_pt_pct"] = 35.5
        df["ast_to_tov"] = 1.7
        df["oreb_pct"] = 26.0
    return df


def _merge_clutch_stats(
    df: pd.DataFrame, season: str, playoff_ids: Dict[int, str],
) -> pd.DataFrame:
    """Merge clutch net rating into the team frame."""
    try:
        raw = _fetch_raw_team_clutch_stats(season)
        keep_ids = set(playoff_ids.keys())
        sub = raw[raw["TEAM_ID"].isin(keep_ids)].copy()
        sub["team_id"] = sub["TEAM_ID"].map(playoff_ids)

        # Clutch PLUS_MINUS per game as a proxy for clutch net rating
        if "PLUS_MINUS" in sub.columns:
            gp = sub.get("GP", pd.Series(dtype=float)).replace(0, 1)
            clutch_net = (sub["PLUS_MINUS"] / gp).astype(float)
            clutch_map = dict(zip(sub["team_id"], clutch_net))
        elif "NET_RATING" in sub.columns:
            clutch_map = dict(zip(sub["team_id"], sub["NET_RATING"].astype(float)))
        else:
            clutch_map = {}

        df["clutch_net_rtg"] = df["team_id"].map(clutch_map).fillna(0.0)
        # Opponent 3P% — not available from clutch endpoint; default from base or constant
        if "opp_three_pt_pct" not in df.columns:
            df["opp_three_pt_pct"] = 35.0
        logger.info("Merged clutch stats")
    except Exception as exc:
        logger.warning("Failed to fetch clutch stats (%s); using defaults", exc)
        df["clutch_net_rtg"] = 0.0
        if "opp_three_pt_pct" not in df.columns:
            df["opp_three_pt_pct"] = 35.0
    return df


def get_team_stats(season: Optional[str] = None) -> pd.DataFrame:
    """
    Return a DataFrame of regular-season team statistics for the 16 playoff teams.

    When *season* is ``None`` the current NBA season is used automatically.

    Schema:
        team_id, team_name, seed, conference,
        off_rtg, def_rtg, pace, net_rtg, w, l,
        three_pt_pct, ast_to_tov, oreb_pct, clutch_net_rtg, opp_three_pt_pct,
        last20_game_log (list of dicts)
    """
    if season is None:
        season = current_nba_season()

    if not _nba_api_available():
        return fallback_data.build_team_stats_df()

    try:
        raw = _fetch_raw_team_stats(season)
        playoff_ids, seed_map = _ensure_playoff_discovery(season, raw)
        df = _normalize_team_frame(raw, playoff_ids, seed_map)
        df = _merge_base_stats(df, season, playoff_ids)
        df = _merge_clutch_stats(df, season, playoff_ids)
        # Opponent 3P% default (would need opponent shooting splits endpoint for real data)
        if "opp_three_pt_pct" not in df.columns:
            df["opp_three_pt_pct"] = 35.0

        inverse = {v: k for k, v in playoff_ids.items()}
        logs: List[List[dict]] = []
        for _, row in df.iterrows():
            nba_id = inverse.get(row["team_id"])
            if nba_id is None:
                logs.append([])
                continue
            try:
                raw_log = _fetch_team_game_log(int(nba_id), season)
                logs.append(_game_log_to_records(raw_log, float(row["pace"])))
            except Exception as exc:
                logger.warning("Game log fetch failed for %s: %s", row["team_id"], exc)
                logs.append([])
        df["last20_game_log"] = logs
        logger.info("Fetched team stats for %d teams via nba_api (%s)", len(df), season)
        return df
    except Exception as exc:
        logger.warning("Falling back to hardcoded team stats: %s", exc)
        return fallback_data.build_team_stats_df()


def _normalize_player_frame(raw: pd.DataFrame, playoff_ids: Dict[int, str]) -> pd.DataFrame:
    keep_ids = set(playoff_ids.keys())
    sub = raw[raw["TEAM_ID"].isin(keep_ids)].copy()
    sub["team_id"] = sub["TEAM_ID"].map(playoff_ids)

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


def get_player_stats(season: Optional[str] = None) -> pd.DataFrame:
    """
    Return a DataFrame of per-36 + usage for players on the 16 playoff teams.

    When *season* is ``None`` the current NBA season is used automatically.
    """
    if season is None:
        season = current_nba_season()

    if not _nba_api_available():
        return fallback_data.build_player_stats_df()

    try:
        raw = _fetch_raw_player_stats(season)
        playoff_ids, _ = _ensure_playoff_discovery(season)
        df = _normalize_player_frame(raw, playoff_ids)
        df = _augment_injuries(df)
        logger.info("Fetched player stats for %d players via nba_api (%s)", len(df), season)
        return df
    except Exception as exc:
        logger.warning("Falling back to hardcoded player stats: %s", exc)
        return fallback_data.build_player_stats_df()


def clear_cache() -> None:
    """Remove all joblib cache entries."""
    _memory.clear(warn=False)
    logger.info("Cleared nba_api cache at %s", _CACHE_DIR)


def get_playoff_seed_map(
    season: Optional[str] = None,
    return_pairs: bool = False,
):
    """Return the seed map (and optional round-1 pairs) for *season*.

    Preference order:
      1. Authoritative :class:`PlayoffPicture` endpoint (includes play-in).
      2. Top-8 by regular-season wins from ``LeagueDashTeamStats``.
      3. Hardcoded ``fallback_data`` bracket.

    When *return_pairs* is ``True`` the return value is
    ``(seed_map, pairs_or_None)``; otherwise just the ``seed_map``.
    ``pairs`` is a ``{"East": [(high_seed, low_seed), ...], "West": [...]}``
    dict taken from the authoritative source. It is ``None`` whenever the
    bracket was derived rather than fetched.
    """
    if season is None:
        season = current_nba_season()

    seed_map: Optional[Dict[str, Dict[int, str]]] = None
    pairs: Optional[Dict[str, List[Tuple[int, int]]]] = None

    if _nba_api_available():
        try:
            _, seed_map = _ensure_playoff_discovery(season)
            pairs = _cached_round1_pairs
        except Exception as exc:
            logger.warning("Live seed discovery failed (%s); using hardcoded.", exc)
            seed_map = None

    if seed_map is None:
        seed_map = fallback_data.get_seed_to_team_id()
        pairs = None

    if return_pairs:
        return seed_map, pairs
    return seed_map


# ---------------------------------------------------------------------------
# Playoff series results  (balldontlie + ESPN)
# ---------------------------------------------------------------------------

_ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    "?seasontype=3&dates={date_range}&limit=900"
)
_BALLDONTLIE_GAMES = "https://api.balldontlie.io/v1/games"


def _series_key(a: str, b: str) -> str:
    return "_".join(sorted([a, b]))


def _build_series_from_games(games: List[Dict]) -> Dict[str, Dict]:
    """Aggregate a list of completed game dicts into series win records.

    Each game dict must have keys: ``home_abbr``, ``away_abbr``, ``winner_abbr``.
    """
    series: Dict[str, Dict] = {}
    for g in games:
        home, away, winner = g["home_abbr"], g["away_abbr"], g["winner_abbr"]
        key = _series_key(home, away)
        if key not in series:
            series[key] = {
                "teams": tuple(sorted([home, away])),
                "wins": {home: 0, away: 0},
                "complete": False,
                "winner": None,
            }
        series[key]["wins"][winner] = series[key]["wins"].get(winner, 0) + 1
        if max(series[key]["wins"].values()) >= 4:
            series[key]["complete"] = True
            series[key]["winner"] = max(series[key]["wins"], key=lambda t: series[key]["wins"][t])
    return series


def _fetch_series_espn() -> Dict[str, Dict]:
    """Return current playoff series records from the ESPN unofficial scoreboard API.

    Covers the full April-June window of the current playoff year in one request.
    No API key required.
    """
    import requests

    year = current_playoff_year()
    date_range = f"{year}0412-{year}0630"
    url = _ESPN_SCOREBOARD.format(date_range=date_range)
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    games: List[Dict] = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        if not comp.get("status", {}).get("type", {}).get("completed", False):
            continue
        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue
        abbrs = [c["team"]["abbreviation"] for c in competitors]
        scores = [int(c.get("score", 0) or 0) for c in competitors]
        winner = abbrs[0] if scores[0] > scores[1] else abbrs[1]
        games.append({"home_abbr": abbrs[0], "away_abbr": abbrs[1], "winner_abbr": winner})

    return _build_series_from_games(games)


def _fetch_series_balldontlie() -> Dict[str, Dict]:
    """Return current playoff series records from balldontlie.io v1 API.

    Requires ``BALLDONTLIE_API_KEY`` env var (free-tier key from balldontlie.io).
    Paginates via cursor until all playoff games for the season are collected.
    """
    import requests

    api_key = os.environ.get("BALLDONTLIE_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    year = int(current_nba_season().split("-")[0])

    games: List[Dict] = []
    cursor: Optional[str] = None
    while True:
        params: list = [("postseason", "true"), ("per_page", 100), ("seasons[]", year)]
        if cursor:
            params.append(("cursor", cursor))
        resp = requests.get(_BALLDONTLIE_GAMES, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        for game in data.get("data", []):
            if str(game.get("status", "")).lower() != "final":
                continue
            home = game.get("home_team", {}).get("abbreviation", "")
            away = game.get("visitor_team", {}).get("abbreviation", "")
            home_score = int(game.get("home_team_score") or 0)
            away_score = int(game.get("visitor_team_score") or 0)
            if not home or not away or home_score == away_score:
                continue
            winner = home if home_score > away_score else away
            games.append({"home_abbr": home, "away_abbr": away, "winner_abbr": winner})

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

    return _build_series_from_games(games)


def get_series_results(season: Optional[str] = None) -> Dict[str, Dict]:
    """Return current playoff series win records.

    Tries sources in priority order:
      1. balldontlie.io   — if ``BALLDONTLIE_API_KEY`` env var is set.
      2. ESPN unofficial  — keyless, no registration required.

    Returns a dict keyed by sorted team-abbreviation pair, e.g.::

        {
            "BOS_NYK": {
                "teams": ("BOS", "NYK"),
                "wins": {"BOS": 3, "NYK": 1},
                "complete": False,
                "winner": None,
            },
            ...
        }

    Returns an empty dict when all sources fail.
    """
    if season is not None:
        pass  # season param reserved for future use; sources auto-detect current year

    if os.environ.get("BALLDONTLIE_API_KEY"):
        try:
            series = _fetch_series_balldontlie()
            if series:
                logger.info("Series results from balldontlie.io (%d series)", len(series))
                return series
        except Exception as exc:
            logger.warning("balldontlie series fetch failed: %s", exc)

    try:
        series = _fetch_series_espn()
        if series:
            logger.info("Series results from ESPN (%d series)", len(series))
            return series
    except Exception as exc:
        logger.warning("ESPN series fetch failed: %s", exc)

    logger.warning("All series result sources failed; returning empty.")
    return {}


# ---------------------------------------------------------------------------
# Historical data for matchup-model training
# ---------------------------------------------------------------------------

def _build_simple_team_vector(row: Dict, all_rows: List[Dict]) -> List[float]:
    """Build a 21-dim feature vector from season-level team stats only.

    Player-impact / game-log features are filled with neutral defaults so that
    the vector is compatible with :class:`TeamStateAggregator` output.
    """
    off_rtgs = np.array([r["off_rtg"] for r in all_rows], dtype=float)
    def_rtgs = np.array([r["def_rtg"] for r in all_rows], dtype=float)
    paces = np.array([r["pace"] for r in all_rows], dtype=float)
    three_pts = np.array([r.get("three_pt_pct", 35.5) for r in all_rows], dtype=float)
    ast_tovs = np.array([r.get("ast_to_tov", 1.7) for r in all_rows], dtype=float)
    orebs = np.array([r.get("oreb_pct", 26.0) for r in all_rows], dtype=float)
    clutches = np.array([r.get("clutch_net_rtg", 0.0) for r in all_rows], dtype=float)

    def _zs(arr: np.ndarray, val: float) -> float:
        mu, sd = float(arr.mean()), float(arr.std())
        return (val - mu) / sd if sd > 1e-9 else 0.0

    net_rtg = row["off_rtg"] - row["def_rtg"]
    elo = 1500.0 + net_rtg * 15.0

    exp = 16.5
    off_p = max(row["off_rtg"], 1.0) ** exp
    def_p = max(row["def_rtg"], 1.0) ** exp
    pyth = off_p / (off_p + def_p)

    w = float(row.get("w", 41))
    l_val = float(row.get("l", 41))
    actual_wpct = w / max(w + l_val, 1.0)
    overperf = actual_wpct - pyth

    net_rtgs = off_rtgs - def_rtgs
    rank_pct = float(np.searchsorted(np.sort(net_rtgs), net_rtg)) / max(len(net_rtgs), 1)
    momentum = rank_pct * 2.0 - 1.0

    return [
        elo,                                                    # [0]  elo_rating
        elo,                                                    # [1]  injury_adj (no data)
        momentum,                                               # [2]  lstm_momentum
        _zs(off_rtgs, row["off_rtg"]),                          # [3]  off_rtg_norm
        -_zs(def_rtgs, row["def_rtg"]),                         # [4]  def_rtg_norm
        _zs(paces, row["pace"]),                                # [5]  pace_norm
        0.0,                                                    # [6]  top3 (unknown)
        0.0,                                                    # [7]  depth (unknown)
        pyth,                                                   # [8]  pythagorean
        overperf,                                               # [9]  overperformance
        0.0,                                                    # [10] volatility
        0.0,                                                    # [11] off_rtg_trend
        0.5,                                                    # [12] close_game_wr
        0.5,                                                    # [13] home_win_pct
        0.0,                                                    # [14] best_player
        0.33,                                                   # [15] star_conc
        1.0,                                                    # [16] healthy_pct
        _zs(three_pts, row.get("three_pt_pct", 35.5)),         # [17]
        _zs(ast_tovs, row.get("ast_to_tov", 1.7)),             # [18]
        _zs(orebs, row.get("oreb_pct", 26.0)),                 # [19]
        _zs(clutches, row.get("clutch_net_rtg", 0.0)),         # [20]
    ]


def _fetch_historical_team_rows(season: str) -> List[Dict]:
    """Fetch all 30 teams' stats for *season* and return as list of dicts."""
    raw = _fetch_raw_team_stats(season)
    df = raw.copy()
    df["abbrev"] = df["TEAM_ID"].map(FULL_NBA_TEAM_ID_MAP)
    df = df.dropna(subset=["abbrev"])

    try:
        base = _fetch_raw_team_base_stats(season)
        bc = base.copy()
        bc["abbrev"] = bc["TEAM_ID"].map(FULL_NBA_TEAM_ID_MAP)
        bc = bc.dropna(subset=["abbrev"]).set_index("abbrev")
    except Exception:
        bc = pd.DataFrame()

    rows: List[Dict] = []
    for _, r in df.iterrows():
        tid = r["abbrev"]
        d: Dict = {
            "team_id": tid,
            "off_rtg": float(r.get("OFF_RATING", 110.0)),
            "def_rtg": float(r.get("DEF_RATING", 110.0)),
            "pace": float(r.get("PACE", 99.0)),
            "w": int(r.get("W", 41)),
            "l": int(r.get("L", 41)),
        }
        if not bc.empty and tid in bc.index:
            br = bc.loc[tid]
            fg3 = float(br.get("FG3_PCT", 0.35))
            d["three_pt_pct"] = fg3 * 100.0 if fg3 < 1.0 else fg3
            ast_v = float(br.get("AST", 25.0))
            tov_v = float(br.get("TOV", 14.0))
            d["ast_to_tov"] = ast_v / max(tov_v, 1.0)
            d["oreb_pct"] = float(br.get("OREB", 10.0))
        else:
            d["three_pt_pct"] = 35.5
            d["ast_to_tov"] = 1.7
            d["oreb_pct"] = 26.0
        d["clutch_net_rtg"] = 0.0
        rows.append(d)
    return rows


def get_historical_matchup_data(n_years: int = 5) -> pd.DataFrame:
    """Fetch playoff game outcomes for the last *n_years* seasons and pair
    them with simplified 21-dim team vectors.

    Returns a DataFrame with columns ``team_a_vector``, ``team_b_vector``,
    ``team_a_won``.  Returns an empty DataFrame when ``nba_api`` is
    unavailable or data cannot be fetched.
    """
    if not _nba_api_available():
        logger.info("nba_api unavailable; no historical matchup data.")
        return pd.DataFrame(columns=["team_a_vector", "team_b_vector", "team_a_won"])

    seasons = historical_seasons(n_years)
    all_matchups: List[Dict] = []

    for season in seasons:
        try:
            team_rows = _fetch_historical_team_rows(season)
            if not team_rows:
                continue
            team_vectors = {
                r["team_id"]: _build_simple_team_vector(r, team_rows) for r in team_rows
            }

            playoff_df = _fetch_playoff_game_log(season)
            if playoff_df is None or playoff_df.empty:
                logger.info("No playoff games found for %s", season)
                continue

            playoff_df = playoff_df.copy()
            playoff_df["abbrev"] = playoff_df["TEAM_ID"].map(FULL_NBA_TEAM_ID_MAP)
            playoff_df = playoff_df.dropna(subset=["abbrev"])

            season_count = 0
            for _, grp in playoff_df.groupby("GAME_ID"):
                if len(grp) != 2:
                    continue
                rows_pair = grp.sort_values("TEAM_ID").to_dict("records")
                a, b = rows_pair[0], rows_pair[1]
                tid_a, tid_b = a["abbrev"], b["abbrev"]
                if tid_a not in team_vectors or tid_b not in team_vectors:
                    continue
                a_won = str(a.get("WL", "")).upper() == "W"
                all_matchups.append({
                    "team_a_vector": team_vectors[tid_a],
                    "team_b_vector": team_vectors[tid_b],
                    "team_a_won": a_won,
                })
                season_count += 1

            logger.info("Loaded %d playoff games from %s", season_count, season)
            time.sleep(1.0)  # polite delay between seasons
        except Exception as exc:
            logger.warning("Failed to fetch historical data for %s: %s", season, exc)
            continue

    if not all_matchups:
        logger.info("No historical matchup data collected.")
        return pd.DataFrame(columns=["team_a_vector", "team_b_vector", "team_a_won"])

    logger.info("Total historical matchup samples: %d", len(all_matchups))
    return pd.DataFrame(all_matchups)
