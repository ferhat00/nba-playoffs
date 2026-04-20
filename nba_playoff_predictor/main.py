"""
End-to-end orchestrator for the NBA Playoff Prediction System.

Run:
    python -m nba_playoff_predictor.main
    python -m nba_playoff_predictor.main --quick
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python main.py` from within the package directory.
_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT.parent))

from nba_playoff_predictor.data import fallback_data, fetcher  # noqa: E402
from nba_playoff_predictor.models.matchup_predictor import XGBoostMatchupEngine  # noqa: E402
from nba_playoff_predictor.models.player_evaluator import BayesianPlayerEvaluator  # noqa: E402
from nba_playoff_predictor.models.team_aggregator import TeamStateAggregator  # noqa: E402
from nba_playoff_predictor.simulation.bracket_simulator import (  # noqa: E402
    BracketSpec,
    PlayoffSimulator,
)


def _configure_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
    except Exception:
        pass


def _build_bracket_spec(team_df: pd.DataFrame) -> BracketSpec:
    seed_map = fetcher.get_playoff_seed_map()
    id_to_name = dict(zip(team_df["team_id"], team_df["team_name"]))
    seeds_by_team = dict(zip(team_df["team_id"], team_df["seed"]))
    conf_by_team = dict(zip(team_df["team_id"], team_df["conference"]))
    return BracketSpec(
        seed_map=seed_map,
        team_id_to_name=id_to_name,
        seeds_by_team=seeds_by_team,
        conference_by_team=conf_by_team,
    )


def _log_top_players_by_conference(
    impact_df: pd.DataFrame, player_df: pd.DataFrame, team_df: pd.DataFrame
) -> None:
    conf_map = dict(zip(team_df["team_id"], team_df["conference"]))
    name_map = dict(zip(player_df["player_id"].astype(str), player_df.get("player_name", player_df["player_id"])))
    merged = impact_df.copy()
    merged["player_id"] = merged["player_id"].astype(str)
    merged["conference"] = merged["team_id"].map(conf_map)
    merged["player_name"] = merged["player_id"].map(name_map).fillna(merged["player_id"])
    for conf in ("East", "West"):
        sub = merged[merged["conference"] == conf].sort_values("composite_impact", ascending=False).head(5)
        lines = "; ".join(f"{r.player_name} ({r.team_id}, {r.composite_impact:+.2f})" for r in sub.itertuples())
        logging.getLogger(__name__).info("Top-5 %s impact: %s", conf, lines)


def run(n_simulations: int, mcmc_samples: int) -> None:
    log = logging.getLogger(__name__)
    _set_seeds(42)

    playoff_year = fetcher.current_playoff_year()
    season = fetcher.current_nba_season()

    # 1. Fetch data (cache-aware).
    log.info("Step 1/7: fetching current season data (%s)", season)
    team_df = fetcher.get_team_stats()
    player_df = fetcher.get_player_stats()
    log.info("Loaded %d teams, %d players", len(team_df), len(player_df))

    # 1b. Fetch historical playoff data for matchup model training.
    log.info("Step 1b/7: fetching historical matchup data (up to 5 years)")
    historical_df = fetcher.get_historical_matchup_data(n_years=5)

    # 2. Bayesian player evaluator.
    log.info("Step 2/7: fitting BayesianPlayerEvaluator (MCMC samples=%d)", mcmc_samples)
    if mcmc_samples < 500:
        log.warning("Running with fewer than 500 MCMC samples (%d); posteriors will be noisier.", mcmc_samples)
    evaluator = BayesianPlayerEvaluator(n_samples=mcmc_samples, n_chains=2, random_seed=42)
    evaluator.fit(player_df)
    impact_df = evaluator.get_player_impact_scores()
    _log_top_players_by_conference(impact_df, player_df, team_df)

    # 3. Team state aggregator.
    log.info("Step 3/7: building TeamStateAggregator")
    aggregator = TeamStateAggregator(lstm_epochs=15, random_seed=42)
    aggregator.build(team_df, impact_df, raw_player_df=player_df)
    for tid, vec in aggregator.all_vectors().items():
        name = team_df.loc[team_df["team_id"] == tid, "team_name"].iloc[0]
        log.info("Vector[%s] %s: %s", tid, name, np.array2string(vec, precision=3, separator=", "))

    # 4. Matchup predictor.
    log.info("Step 4/7: training XGBoostMatchupEngine")
    engine = XGBoostMatchupEngine(n_estimators=300, max_depth=4, learning_rate=0.05, random_seed=42)
    engine.train(
        historical_matchup_df=historical_df if not historical_df.empty else None,
        team_vectors=aggregator.all_vectors(),
        n_synthetic=5000,
    )
    if engine.holdout_auc is not None:
        log.info("Matchup engine holdout AUC: %.4f", engine.holdout_auc)

    # 5. Simulate.
    log.info("Step 5/7: simulating %d brackets", n_simulations)
    bracket_spec = _build_bracket_spec(team_df)
    simulator = PlayoffSimulator(team_df, aggregator, engine, bracket_spec)
    simulator.run(n_simulations=n_simulations)

    # 6. Output.
    log.info("Step 6/7: writing outputs")
    table = simulator.results_table()
    print(f"\n=== {playoff_year} NBA Playoff — Championship Probabilities ===")
    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    print(table.to_string(index=False))
    print()

    out_path = simulator.visualize(playoff_year=playoff_year)
    log.info("Saved bracket visualization: %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="NBA Playoff Prediction System")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test mode: 1000 sims + 100 MCMC samples (finishes in <60s).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO logging.",
    )
    args = parser.parse_args()

    _configure_logging(verbose=not args.quiet)

    if args.quick:
        n_sims = 1_000
        mcmc = 100
    else:
        n_sims = 100_000
        mcmc = 500

    run(n_simulations=n_sims, mcmc_samples=mcmc)


if __name__ == "__main__":
    main()
