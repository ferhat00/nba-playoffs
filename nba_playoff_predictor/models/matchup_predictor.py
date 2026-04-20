"""
XGBoost classifier over team-vector differentials to predict single-game win prob.

Feature vector per matchup = team_a_vector - team_b_vector (length 8).
When real historical matchup data is not supplied, `train()` generates a
synthetic set by sampling from Normal distributions parameterized by each
team's actual stats and labeling wins with a logistic of net_rtg differential.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score  # type: ignore

try:
    import xgboost as xgb  # type: ignore

    _XGB_AVAILABLE = True
except Exception as _exc:  # pragma: no cover
    logging.getLogger(__name__).warning(
        "xgboost unavailable (%s); falling back to sklearn GradientBoostingClassifier",
        _exc,
    )
    xgb = None  # type: ignore
    _XGB_AVAILABLE = False

logger = logging.getLogger(__name__)


class XGBoostMatchupEngine:
    """XGBoost-backed per-game win probability estimator."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_seed: int = 42,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.random_seed = int(random_seed)
        self.model: Optional["xgb.XGBClassifier"] = None
        self._trained: bool = False
        self._holdout_auc: Optional[float] = None

    # --- Synthetic data generation ---------------------------------------

    @staticmethod
    def _logistic(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _generate_synthetic_data(
        self,
        team_vectors: Dict[str, np.ndarray],
        n_samples: int = 2500,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_seed)
        team_ids = list(team_vectors.keys())
        if len(team_ids) < 2:
            raise ValueError("Need at least 2 teams to generate synthetic matchups.")

        rows = []
        # Elo index 0, off_n index 3, def_n index 4 — proxy net rating differential.
        for _ in range(n_samples):
            a, b = rng.choice(team_ids, size=2, replace=False)
            va = team_vectors[a].copy()
            vb = team_vectors[b].copy()
            # Perturb each vector slightly to simulate noisy game-level states.
            noise = rng.normal(0.0, 0.1, size=va.shape)
            va_obs = va + noise
            vb_obs = vb + rng.normal(0.0, 0.1, size=vb.shape)

            # True win probability driven by injury-adjusted Elo differential.
            elo_diff = va[1] - vb[1]
            top3_diff = va[6] - vb[6]
            momentum_diff = va[2] - vb[2]
            score = 0.012 * elo_diff + 0.6 * top3_diff + 0.3 * momentum_diff
            p_win = float(self._logistic(np.array([score]))[0])
            won = bool(rng.random() < p_win)
            rows.append(
                {
                    "team_a_vector": va_obs.tolist(),
                    "team_b_vector": vb_obs.tolist(),
                    "team_a_won": won,
                }
            )
        return pd.DataFrame(rows)

    # --- Training / inference --------------------------------------------

    @staticmethod
    def _diff_features(df: pd.DataFrame) -> np.ndarray:
        a = np.array(df["team_a_vector"].tolist(), dtype=float)
        b = np.array(df["team_b_vector"].tolist(), dtype=float)
        return a - b

    def train(
        self,
        historical_matchup_df: Optional[pd.DataFrame] = None,
        team_vectors: Optional[Dict[str, np.ndarray]] = None,
        n_synthetic: int = 2500,
    ) -> None:
        """Fit the XGBoost classifier. Accepts real data or synthesizes from team_vectors."""
        if historical_matchup_df is None or historical_matchup_df.empty:
            if team_vectors is None:
                raise ValueError("Supply either historical_matchup_df or team_vectors.")
            logger.info("No historical matchup data supplied; generating %d synthetic samples", n_synthetic)
            df = self._generate_synthetic_data(team_vectors, n_samples=n_synthetic)
        else:
            df = historical_matchup_df.copy()

        X = self._diff_features(df)
        y = df["team_a_won"].astype(int).to_numpy()

        # 80/20 split for holdout AUC reporting.
        rng = np.random.default_rng(self.random_seed)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        split = int(len(idx) * 0.8)
        train_idx, test_idx = idx[:split], idx[split:]

        if _XGB_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                eval_metric="logloss",
                random_state=self.random_seed,
                tree_method="hist",
                n_jobs=1,
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier  # type: ignore

            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_seed,
            )
        self.model.fit(X[train_idx], y[train_idx])
        self._trained = True

        if len(test_idx) > 0 and len(np.unique(y[test_idx])) > 1:
            probs = self.model.predict_proba(X[test_idx])[:, 1]
            self._holdout_auc = float(roc_auc_score(y[test_idx], probs))
            logger.info("XGBoostMatchupEngine holdout AUC: %.4f", self._holdout_auc)
        else:
            logger.warning("Holdout set too small or single-class; AUC not computed")

    def predict_win_probability(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Return P(team_a wins a single game) given the two team vectors."""
        if not self._trained or self.model is None:
            raise RuntimeError("XGBoostMatchupEngine.train must be called first.")
        diff = (np.asarray(vec_a, dtype=float) - np.asarray(vec_b, dtype=float)).reshape(1, -1)
        prob = float(self.model.predict_proba(diff)[0, 1])
        return max(0.01, min(0.99, prob))

    @property
    def holdout_auc(self) -> Optional[float]:
        return self._holdout_auc
