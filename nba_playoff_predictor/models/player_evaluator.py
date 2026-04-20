"""
Bayesian hierarchical player-impact model.

The model partially pools each player's offensive and defensive impact toward
their team's mean. This shrinks small-sample estimates and produces posterior
mean impact scores that feed the team-level aggregation layer.

Model:
    offensive_impact[i] ~ Normal(team_mu_off[team[i]], team_sigma_off[team[i]])
    defensive_impact[i] ~ Normal(team_mu_def[team[i]], team_sigma_def[team[i]])
    team_mu_*    ~ Normal(0, 1)
    team_sigma_* ~ HalfNormal(1)

Features observed per player:
    y_off ~ standardized( 0.55 * pts_per36 + 0.20 * ast_per36 + 0.25 * bpm )
    y_def ~ standardized( 0.60 * reb_per36 + 0.40 * bpm ) * proxy
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import pymc as pm  # type: ignore

    _PYMC_AVAILABLE = True
except Exception as _exc:  # pragma: no cover - environment-specific
    logger.warning("PyMC unavailable (%s); will use closed-form hierarchical estimator.", _exc)
    pm = None  # type: ignore
    _PYMC_AVAILABLE = False


def _standardize(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd <= 1e-9:
        return np.zeros_like(x)
    return (x - mu) / sd


class BayesianPlayerEvaluator:
    """Hierarchical Bayesian estimator of per-player offensive/defensive impact."""

    def __init__(self, n_samples: int = 500, n_chains: int = 2, random_seed: int = 42) -> None:
        self.n_samples = int(n_samples)
        self.n_chains = int(n_chains)
        self.random_seed = int(random_seed)
        self._impact_df: Optional[pd.DataFrame] = None
        self._fitted: bool = False

    def _build_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        pts = df["pts_per36"].to_numpy(dtype=float)
        reb = df["reb_per36"].to_numpy(dtype=float)
        ast = df["ast_per36"].to_numpy(dtype=float)
        bpm = df["bpm"].to_numpy(dtype=float)

        y_off_raw = 0.55 * pts + 0.20 * ast + 0.25 * bpm
        y_def_raw = 0.60 * reb + 0.40 * bpm
        return _standardize(y_off_raw), _standardize(y_def_raw)

    def _fit_pymc(self, df: pd.DataFrame, y_off: np.ndarray, y_def: np.ndarray) -> pd.DataFrame:
        assert pm is not None
        team_codes, team_levels = pd.factorize(df["team_id"].values)
        n_teams = len(team_levels)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pm.Model():
                team_mu_off = pm.Normal("team_mu_off", mu=0.0, sigma=1.0, shape=n_teams)
                team_sigma_off = pm.HalfNormal("team_sigma_off", sigma=1.0, shape=n_teams)
                team_mu_def = pm.Normal("team_mu_def", mu=0.0, sigma=1.0, shape=n_teams)
                team_sigma_def = pm.HalfNormal("team_sigma_def", sigma=1.0, shape=n_teams)

                off_impact = pm.Normal(
                    "off_impact",
                    mu=team_mu_off[team_codes],
                    sigma=team_sigma_off[team_codes],
                    shape=len(df),
                )
                def_impact = pm.Normal(
                    "def_impact",
                    mu=team_mu_def[team_codes],
                    sigma=team_sigma_def[team_codes],
                    shape=len(df),
                )

                pm.Normal("y_off", mu=off_impact, sigma=0.5, observed=y_off)
                pm.Normal("y_def", mu=def_impact, sigma=0.5, observed=y_def)

                trace = pm.sample(
                    draws=self.n_samples,
                    tune=max(200, self.n_samples // 2),
                    chains=self.n_chains,
                    cores=1,
                    random_seed=self.random_seed,
                    progressbar=False,
                    target_accept=0.9,
                    return_inferencedata=True,
                )

        posterior = trace.posterior
        off_mean = posterior["off_impact"].mean(dim=("chain", "draw")).values
        def_mean = posterior["def_impact"].mean(dim=("chain", "draw")).values

        out = pd.DataFrame(
            {
                "player_id": df["player_id"].values,
                "team_id": df["team_id"].values,
                "offensive_impact": off_mean.astype(float),
                "defensive_impact": def_mean.astype(float),
            }
        )
        return out

    def _fit_closed_form(self, df: pd.DataFrame, y_off: np.ndarray, y_def: np.ndarray) -> pd.DataFrame:
        """Analytical partial-pooling estimator used when PyMC is unavailable.

        Shrinks each observation toward the team mean with weight proportional
        to the team's within-sample variance (James-Stein style).
        """
        logger.warning("Using closed-form shrinkage estimator instead of PyMC.")
        out_rows = []
        for team_id, grp in df.groupby("team_id"):
            idx = grp.index.to_numpy()
            yo = y_off[idx]
            yd = y_def[idx]
            mu_o, mu_d = float(yo.mean()), float(yd.mean())
            var_o = float(yo.var()) + 1e-6
            var_d = float(yd.var()) + 1e-6
            # Shrinkage factor lambda = var_between / (var_between + var_within)
            lam_o = 1.0 / (1.0 + var_o)
            lam_d = 1.0 / (1.0 + var_d)
            off_shr = lam_o * mu_o + (1 - lam_o) * yo
            def_shr = lam_d * mu_d + (1 - lam_d) * yd
            for j, i in enumerate(idx):
                out_rows.append(
                    {
                        "player_id": df.loc[i, "player_id"],
                        "team_id": team_id,
                        "offensive_impact": float(off_shr[j]),
                        "defensive_impact": float(def_shr[j]),
                    }
                )
        return pd.DataFrame(out_rows)

    def fit(self, player_df: pd.DataFrame) -> None:
        """Fit the hierarchical model against a player-stats DataFrame."""
        if player_df.empty:
            raise ValueError("player_df is empty; cannot fit BayesianPlayerEvaluator.")
        required = {"player_id", "team_id", "pts_per36", "reb_per36", "ast_per36", "bpm"}
        missing = required - set(player_df.columns)
        if missing:
            raise ValueError(f"player_df missing required columns: {missing}")

        df = player_df.reset_index(drop=True)
        y_off, y_def = self._build_features(df)

        logger.info(
            "Fitting BayesianPlayerEvaluator on %d players across %d teams",
            len(df),
            df["team_id"].nunique(),
        )
        if _PYMC_AVAILABLE:
            try:
                impact_df = self._fit_pymc(df, y_off, y_def)
            except Exception as exc:  # pragma: no cover
                logger.warning("PyMC sampling failed (%s); using closed-form fallback", exc)
                impact_df = self._fit_closed_form(df, y_off, y_def)
        else:
            impact_df = self._fit_closed_form(df, y_off, y_def)

        impact_df["composite_impact"] = 0.55 * impact_df["offensive_impact"] + 0.45 * impact_df[
            "defensive_impact"
        ]
        self._impact_df = impact_df
        self._fitted = True
        logger.info("BayesianPlayerEvaluator fit complete")

    def get_player_impact_scores(self) -> pd.DataFrame:
        """Return per-player posterior means as a DataFrame."""
        if not self._fitted or self._impact_df is None:
            raise RuntimeError("BayesianPlayerEvaluator.fit must be called before scoring.")
        return self._impact_df.copy()
