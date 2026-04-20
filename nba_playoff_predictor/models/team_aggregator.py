"""
TeamStateAggregator: collapses team stats + player impacts into an 8-dim vector.

Vector layout:
    [0] elo_rating              — initialized from net_rtg
    [1] injury_adjusted_elo     — elo minus 50 pts per injured top-5 impact player
    [2] lstm_momentum           — tanh-activated final hidden state mean
    [3] off_rtg_norm            — z-scored offensive rating
    [4] def_rtg_norm            — z-scored defensive rating (sign-flipped: higher = better)
    [5] pace_norm               — z-scored pace
    [6] top3_player_impact      — mean composite impact of top-3 players
    [7] depth_score             — mean composite impact of players 4..N
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch  # type: ignore
    from torch import nn  # type: ignore

    _TORCH_AVAILABLE = True
except Exception as _exc:  # pragma: no cover
    logger.warning("PyTorch unavailable (%s); momentum will fallback to net_rtg percentile.", _exc)
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False


class _LSTMMomentum(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore
    """Tiny LSTM that consumes [off_rtg, def_rtg, pace] sequences and outputs a scalar."""

    def __init__(self, input_size: int = 3, hidden_size: int = 32, num_layers: int = 2) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for _LSTMMomentum.")
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        _, (h_n, _) = self.lstm(x)
        last = h_n[-1]
        logit = self.head(last)
        return logit.squeeze(-1)


class TeamStateAggregator:
    """Builds an 8-dimensional state vector per playoff team."""

    VECTOR_DIM: int = 8

    def __init__(self, lstm_epochs: int = 15, lstm_lr: float = 1e-2, random_seed: int = 42) -> None:
        self.lstm_epochs = int(lstm_epochs)
        self.lstm_lr = float(lstm_lr)
        self.random_seed = int(random_seed)
        self._vectors: Dict[str, np.ndarray] = {}
        self._team_df: Optional[pd.DataFrame] = None
        self._impact_df: Optional[pd.DataFrame] = None
        self._built: bool = False

    # --- Elo ---------------------------------------------------------------

    @staticmethod
    def _base_elo(net_rtg: float) -> float:
        return 1500.0 + float(net_rtg) * 15.0

    def _injury_penalty(self, team_id: str, player_impact_df: pd.DataFrame, raw_player_df: pd.DataFrame) -> float:
        """Subtract 50 Elo per top-5 composite-impact player marked injured."""
        team_players = player_impact_df[player_impact_df["team_id"] == team_id]
        if team_players.empty:
            return 0.0
        top5 = team_players.sort_values("composite_impact", ascending=False).head(5)
        injured_ids = set(
            raw_player_df.loc[raw_player_df["injured"] == True, "player_id"].astype(str).tolist()
        )
        n_injured = int(top5["player_id"].astype(str).isin(injured_ids).sum())
        return 50.0 * n_injured

    # --- LSTM --------------------------------------------------------------

    def _fit_and_score_momentum(self, team_df: pd.DataFrame) -> Dict[str, float]:
        """Train the LSTM on all teams' last-20 logs; return per-team momentum scalars."""
        out: Dict[str, float] = {}
        if not _TORCH_AVAILABLE:
            return self._momentum_from_rank(team_df)

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Assemble training tensors.
        seq_list, label_list, team_ids_with_data = [], [], []
        for _, row in team_df.iterrows():
            games = row["last20_game_log"] or []
            if len(games) < 10:
                continue
            feats = np.array(
                [[g["off_rtg"], g["def_rtg"], g["pace"]] for g in games],
                dtype=np.float32,
            )
            # Normalize feature-wise.
            mu = feats.mean(axis=0, keepdims=True)
            sd = feats.std(axis=0, keepdims=True) + 1e-6
            feats_n = (feats - mu) / sd
            # Next-game win targets — shift by 1.
            wins = np.array([1 if g["won"] else 0 for g in games], dtype=np.float32)
            if len(wins) < 2:
                continue
            # Sliding windows: predict game t using games [0..t-1], min window 5.
            for t in range(5, len(feats_n)):
                seq_list.append(feats_n[:t])
                label_list.append(wins[t])
            team_ids_with_data.append(row["team_id"])

        if not seq_list:
            logger.info("No team had sufficient game logs; using rank-percentile momentum.")
            return self._momentum_from_rank(team_df)

        # Pad variable-length sequences to max length with zeros.
        max_len = max(len(s) for s in seq_list)
        feat_dim = seq_list[0].shape[1]
        X = np.zeros((len(seq_list), max_len, feat_dim), dtype=np.float32)
        for i, s in enumerate(seq_list):
            X[i, -len(s) :, :] = s
        y = np.array(label_list, dtype=np.float32)

        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)

        model = _LSTMMomentum(input_size=feat_dim, hidden_size=32, num_layers=2)
        opt = torch.optim.Adam(model.parameters(), lr=self.lstm_lr)
        loss_fn = nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(self.lstm_epochs):
            opt.zero_grad()
            logits = model(X_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            opt.step()
        logger.info("LSTM momentum trained (%d epochs, final loss=%.4f)", self.lstm_epochs, float(loss.item()))

        # Score each team: feed the full last-20 sequence, take tanh of hidden state mean.
        model.eval()
        with torch.no_grad():
            for _, row in team_df.iterrows():
                games = row["last20_game_log"] or []
                if len(games) < 10:
                    out[row["team_id"]] = 0.0
                    continue
                feats = np.array(
                    [[g["off_rtg"], g["def_rtg"], g["pace"]] for g in games],
                    dtype=np.float32,
                )
                mu = feats.mean(axis=0, keepdims=True)
                sd = feats.std(axis=0, keepdims=True) + 1e-6
                feats_n = (feats - mu) / sd
                x = torch.from_numpy(feats_n).unsqueeze(0)
                _, (h_n, _) = model.lstm(x)
                momentum = float(torch.tanh(h_n[-1].mean()).item())
                out[row["team_id"]] = momentum

        # Backfill any teams that lacked logs.
        fallback = self._momentum_from_rank(team_df)
        for tid, val in fallback.items():
            out.setdefault(tid, val)
        return out

    @staticmethod
    def _momentum_from_rank(team_df: pd.DataFrame) -> Dict[str, float]:
        ranks = team_df["net_rtg"].rank(pct=True).astype(float)
        # Map [0,1] -> [-1, 1]
        scaled = (ranks * 2.0 - 1.0).to_numpy()
        return {tid: float(v) for tid, v in zip(team_df["team_id"].values, scaled)}

    # --- Build / query -----------------------------------------------------

    def _depth_scores(self, team_id: str, player_impact_df: pd.DataFrame) -> tuple[float, float]:
        players = player_impact_df[player_impact_df["team_id"] == team_id]
        if players.empty:
            return 0.0, 0.0
        sorted_p = players.sort_values("composite_impact", ascending=False)
        top3 = float(sorted_p.head(3)["composite_impact"].mean())
        rest = sorted_p.iloc[3:]
        depth = float(rest["composite_impact"].mean()) if len(rest) else 0.0
        return top3, depth

    def build(self, team_df: pd.DataFrame, player_impact_df: pd.DataFrame, raw_player_df: Optional[pd.DataFrame] = None) -> None:
        """Compute and cache each team's 8-dim state vector."""
        required = {"team_id", "off_rtg", "def_rtg", "pace", "net_rtg", "last20_game_log"}
        missing = required - set(team_df.columns)
        if missing:
            raise ValueError(f"team_df missing columns: {missing}")

        if raw_player_df is None:
            raw_player_df = player_impact_df.assign(injured=False)

        self._team_df = team_df.reset_index(drop=True).copy()
        self._impact_df = player_impact_df.copy()

        off = self._team_df["off_rtg"].to_numpy(dtype=float)
        de = self._team_df["def_rtg"].to_numpy(dtype=float)
        pa = self._team_df["pace"].to_numpy(dtype=float)
        off_n = (off - off.mean()) / (off.std() + 1e-9)
        def_n = -1.0 * (de - de.mean()) / (de.std() + 1e-9)  # flip: higher = better defense
        pace_n = (pa - pa.mean()) / (pa.std() + 1e-9)

        momentum_map = self._fit_and_score_momentum(self._team_df)

        for i, row in self._team_df.iterrows():
            team_id = row["team_id"]
            elo = self._base_elo(row["net_rtg"])
            pen = self._injury_penalty(team_id, player_impact_df, raw_player_df)
            elo_adj = elo - pen
            momentum = float(momentum_map.get(team_id, 0.0))
            top3, depth = self._depth_scores(team_id, player_impact_df)

            vec = np.array(
                [
                    elo,
                    elo_adj,
                    momentum,
                    float(off_n[i]),
                    float(def_n[i]),
                    float(pace_n[i]),
                    top3,
                    depth,
                ],
                dtype=float,
            )
            self._vectors[team_id] = vec

        self._built = True
        logger.info("TeamStateAggregator built for %d teams", len(self._vectors))

    def get_team_vector(self, team_id: str) -> np.ndarray:
        if not self._built:
            raise RuntimeError("TeamStateAggregator.build must be called before querying.")
        if team_id not in self._vectors:
            raise KeyError(f"Unknown team_id: {team_id}")
        return self._vectors[team_id].copy()

    def all_vectors(self) -> Dict[str, np.ndarray]:
        if not self._built:
            raise RuntimeError("TeamStateAggregator.build must be called first.")
        return {k: v.copy() for k, v in self._vectors.items()}
