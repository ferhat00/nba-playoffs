"""
Monte Carlo simulation of the 2025 NBA Playoffs + plotly bracket visualization.

Design notes
------------
1. Inference cost matters. Calling XGBoost inside every simulated game would
   dominate runtime. Instead we precompute a 16x16 matrix of single-game win
   probabilities (no home-court bonus applied) once up-front. Each simulated
   game is then a single lookup + ±0.025 home-court adjustment + one RNG draw.

2. Parallelization uses `multiprocessing.Pool` over simulation batches. The
   worker payload is trivially picklable (dict + seed), which sidesteps the
   usual Windows pickling issues with XGBoost Booster objects.

3. The bracket visualization is a hand-rolled plotly figure: nodes arranged
   as a tournament tree, color-coded by championship probability.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

HOME_COURT_BONUS: float = 0.025
HOME_GAMES: Tuple[int, ...] = (1, 2, 5, 7)  # at higher seed


# ---------------------------------------------------------------------------
# Worker-safe helpers (top-level so multiprocessing can pickle them on Windows)
# ---------------------------------------------------------------------------


def _simulate_series(
    high_team: str,
    low_team: str,
    prob_matrix: Dict[Tuple[str, str], float],
    rng: np.random.Generator,
    high_has_home_court: bool = True,
) -> str:
    """Return the team_id that wins a best-of-7 series."""
    base_p = prob_matrix[(high_team, low_team)]
    wins_high = 0
    wins_low = 0
    for game_num in range(1, 8):
        if high_has_home_court:
            at_home = high_team if game_num in HOME_GAMES else low_team
        else:
            at_home = low_team if game_num in HOME_GAMES else high_team

        if at_home == high_team:
            p = base_p + HOME_COURT_BONUS
        else:
            p = base_p - HOME_COURT_BONUS
        p = min(0.99, max(0.01, p))
        if rng.random() < p:
            wins_high += 1
        else:
            wins_low += 1
        if wins_high == 4:
            return high_team
        if wins_low == 4:
            return low_team
    # Best-of-7 guaranteed to finish in <=7 games; unreachable.
    return high_team if wins_high > wins_low else low_team


def _simulate_batch(payload: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Simulate `n_sims` brackets; return counts keyed by team_id and round."""
    seed = payload["seed"]
    n_sims = payload["n_sims"]
    prob_matrix = payload["prob_matrix"]
    seed_map = payload["seed_map"]  # {"East": {1: "CLE", ...}, "West": {...}}
    seeds_by_team = payload["seeds_by_team"]  # {team_id: seed}

    rng = np.random.default_rng(seed)

    counts: Dict[str, Dict[str, int]] = {}
    for conf in ("East", "West"):
        for tid in seed_map[conf].values():
            counts[tid] = {"second_round": 0, "conf_finals": 0, "finals": 0, "championship": 0}

    for _ in range(n_sims):
        conf_winners: Dict[str, str] = {}

        for conf in ("East", "West"):
            s = seed_map[conf]
            # Round 1 matchups: (1,8) (4,5) (3,6) (2,7)
            r1_pairs = [(s[1], s[8]), (s[4], s[5]), (s[3], s[6]), (s[2], s[7])]
            r1_winners = [
                _simulate_series(h, l, prob_matrix, rng, high_has_home_court=True)
                for h, l in r1_pairs
            ]
            for w in r1_winners:
                counts[w]["second_round"] += 1

            # Round 2 pairings: winner(1v8) vs winner(4v5); winner(3v6) vs winner(2v7)
            r2_pairs = [
                (r1_winners[0], r1_winners[1]),
                (r1_winners[2], r1_winners[3]),
            ]
            r2_winners = []
            for a, b in r2_pairs:
                # Higher seed (smaller number) gets home-court.
                if seeds_by_team[a] <= seeds_by_team[b]:
                    high, low = a, b
                else:
                    high, low = b, a
                winner = _simulate_series(high, low, prob_matrix, rng, high_has_home_court=True)
                r2_winners.append(winner)

            for w in r2_winners:
                counts[w]["conf_finals"] += 1

            # Conference Finals
            a, b = r2_winners[0], r2_winners[1]
            if seeds_by_team[a] <= seeds_by_team[b]:
                high, low = a, b
            else:
                high, low = b, a
            conf_winner = _simulate_series(high, low, prob_matrix, rng, high_has_home_court=True)
            counts[conf_winner]["finals"] += 1
            conf_winners[conf] = conf_winner

        # NBA Finals: seeding across conferences decided by regular-season wins
        # implied by seed in this bracket — we use seed number as a proxy; ties
        # broken arbitrarily in favor of East.
        east_rep = conf_winners["East"]
        west_rep = conf_winners["West"]
        if seeds_by_team[east_rep] <= seeds_by_team[west_rep]:
            high, low = east_rep, west_rep
        else:
            high, low = west_rep, east_rep
        champ = _simulate_series(high, low, prob_matrix, rng, high_has_home_court=True)
        counts[champ]["championship"] += 1

    return counts


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


@dataclass
class BracketSpec:
    seed_map: Dict[str, Dict[int, str]]  # {"East": {1: team_id, ...}, "West": ...}
    team_id_to_name: Dict[str, str]
    seeds_by_team: Dict[str, int]  # team_id -> seed (1..8), independent of conference
    conference_by_team: Dict[str, str]


class PlayoffSimulator:
    """Monte Carlo bracket simulator with parallel execution and plotly output."""

    def __init__(
        self,
        team_df: pd.DataFrame,
        aggregator: "Any",
        matchup_engine: "Any",
        bracket_spec: BracketSpec,
        output_dir: str = "output",
        n_workers: Optional[int] = None,
    ) -> None:
        self.team_df = team_df.reset_index(drop=True)
        self.aggregator = aggregator
        self.matchup_engine = matchup_engine
        self.bracket_spec = bracket_spec
        self.output_dir = output_dir
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self._prob_matrix: Optional[Dict[Tuple[str, str], float]] = None
        self._results: Optional[Dict[str, Dict[str, float]]] = None

        os.makedirs(self.output_dir, exist_ok=True)

    # --- Precomputation ---------------------------------------------------

    def _build_prob_matrix(self) -> Dict[Tuple[str, str], float]:
        """Compute P(team_a beats team_b) for every pair — excluding home court bonus."""
        team_ids = list(self.bracket_spec.seeds_by_team.keys())
        vectors = {tid: self.aggregator.get_team_vector(tid) for tid in team_ids}
        matrix: Dict[Tuple[str, str], float] = {}
        for a in team_ids:
            for b in team_ids:
                if a == b:
                    matrix[(a, b)] = 0.5
                else:
                    matrix[(a, b)] = float(
                        self.matchup_engine.predict_win_probability(vectors[a], vectors[b])
                    )
        return matrix

    # --- Public API -------------------------------------------------------

    def run(self, n_simulations: int = 100_000) -> Dict[str, Dict[str, float]]:
        """Run the Monte Carlo simulation and return per-team round probabilities."""
        if self._prob_matrix is None:
            logger.info("Precomputing 16x16 single-game win probability matrix...")
            self._prob_matrix = self._build_prob_matrix()

        # Split n_simulations into batches per worker.
        n_workers = self.n_workers
        batch_size = max(1, n_simulations // n_workers)
        batch_counts: List[int] = [batch_size] * n_workers
        batch_counts[-1] += n_simulations - sum(batch_counts)

        payloads = [
            {
                "seed": 42 + i,
                "n_sims": batch_counts[i],
                "prob_matrix": self._prob_matrix,
                "seed_map": self.bracket_spec.seed_map,
                "seeds_by_team": self.bracket_spec.seeds_by_team,
            }
            for i in range(n_workers)
        ]

        logger.info("Running %d simulations across %d workers...", n_simulations, n_workers)
        aggregated: Dict[str, Dict[str, int]] = {}
        # Prefer multiprocessing, but fall back to serial if the user environment
        # can't spawn workers (e.g. notebooks on Windows).
        use_mp = n_workers > 1 and n_simulations >= 2_000
        if use_mp:
            try:
                with Pool(processes=n_workers) as pool:
                    it = pool.imap_unordered(_simulate_batch, payloads)
                    for partial in tqdm(it, total=n_workers, desc="Simulation batches"):
                        self._merge_counts(aggregated, partial)
            except Exception as exc:  # pragma: no cover
                logger.warning("Multiprocessing failed (%s); running serially.", exc)
                use_mp = False
        if not use_mp:
            for p in tqdm(payloads, desc="Simulation batches"):
                partial = _simulate_batch(p)
                self._merge_counts(aggregated, partial)

        # Normalize counts into probabilities keyed by team NAME for output.
        results: Dict[str, Dict[str, float]] = {}
        for tid, rounds in aggregated.items():
            name = self.bracket_spec.team_id_to_name[tid]
            results[name] = {k: v / float(n_simulations) for k, v in rounds.items()}

        self._results = results
        logger.info("Simulation complete")
        return results

    @staticmethod
    def _merge_counts(dst: Dict[str, Dict[str, int]], src: Dict[str, Dict[str, int]]) -> None:
        for tid, rounds in src.items():
            if tid not in dst:
                dst[tid] = {k: 0 for k in rounds}
            for k, v in rounds.items():
                dst[tid][k] += v

    def results_table(self) -> pd.DataFrame:
        """Return a DataFrame ranked by championship probability."""
        if self._results is None:
            raise RuntimeError("PlayoffSimulator.run must be called first.")
        rows = []
        for name, probs in self._results.items():
            rows.append(
                {
                    "team": name,
                    "championship%": 100.0 * probs["championship"],
                    "finals%": 100.0 * probs["finals"],
                    "conf_finals%": 100.0 * probs["conf_finals"],
                    "second_round%": 100.0 * probs["second_round"],
                }
            )
        df = pd.DataFrame(rows).sort_values("championship%", ascending=False).reset_index(drop=True)
        return df

    # --- Visualization ----------------------------------------------------

    def visualize(self, filename: str = "bracket_probabilities.html") -> str:
        """Render a plotly bracket tree to `output/bracket_probabilities.html`."""
        if self._results is None:
            raise RuntimeError("PlayoffSimulator.run must be called first.")

        import plotly.graph_objects as go

        spec = self.bracket_spec
        results = self._results
        seeds_by_team = spec.seeds_by_team

        # Layout: 5 columns on each side (R1..Finals) + 1 central championship node.
        # West on the left (x negative), East on the right (x positive).
        # Y slots encode the 8 seeds per conference with matchups adjacent.
        #
        #   West R1     West R2    West CF     West F     Champion   East F   East CF   East R2   East R1

        east_order = [1, 8, 4, 5, 3, 6, 2, 7]
        west_order = [1, 8, 4, 5, 3, 6, 2, 7]

        # X positions for bracket rounds.
        X_WEST = {"R1": -4, "R2": -3, "CF": -2, "F": -1}
        X_EAST = {"R1": 4, "R2": 3, "CF": 2, "F": 1}
        X_CHAMP = 0

        # Y positions for R1 slots (1 slot per seed, 8 total per conference).
        Y_R1 = [i for i in range(7, -1, -1)]  # 7..0
        Y_R2 = [6.5, 4.5, 2.5, 0.5]
        Y_CF = [5.5, 1.5]
        Y_F = [3.5]

        champ_probs = {name: results[name]["championship"] for name in results}
        max_champ = max(champ_probs.values()) if champ_probs else 1.0

        node_x, node_y, node_text, node_hover, node_color = [], [], [], [], []
        edge_x, edge_y = [], []

        def color_for(p: float) -> float:
            # Map probability to 0..1 intensity.
            return float(p / max_champ) if max_champ > 0 else 0.0

        def add_node(x, y, team_id: Optional[str], label: str, hover: str) -> None:
            node_x.append(x)
            node_y.append(y)
            node_text.append(label)
            node_hover.append(hover)
            if team_id is not None:
                name = spec.team_id_to_name[team_id]
                node_color.append(color_for(champ_probs.get(name, 0.0)))
            else:
                node_color.append(0.0)

        def add_edge(x0, y0, x1, y1) -> None:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        def render_conference(conf: str, x_map: Dict[str, int], order: List[int]) -> None:
            seed_map = spec.seed_map[conf]
            seed_to_y = {seed: Y_R1[i] for i, seed in enumerate(order)}

            # R1 nodes (one per team).
            for seed in order:
                tid = seed_map[seed]
                name = spec.team_id_to_name[tid]
                p_r2 = results[name]["second_round"]
                label = f"#{seed} {name.split()[-1]}<br>→R2: {p_r2*100:.1f}%"
                hover = (
                    f"{name}<br>Seed: {seed} ({conf})<br>"
                    f"Second round: {p_r2*100:.2f}%<br>"
                    f"Conf finals: {results[name]['conf_finals']*100:.2f}%<br>"
                    f"Finals: {results[name]['finals']*100:.2f}%<br>"
                    f"Championship: {results[name]['championship']*100:.2f}%"
                )
                add_node(x_map["R1"], seed_to_y[seed], tid, label, hover)

            # R1 matchup pairs & edges to R2 slot.
            r1_pairs = [(1, 8), (4, 5), (3, 6), (2, 7)]
            for i, (hi, lo) in enumerate(r1_pairs):
                y_r2 = Y_R2[i]
                add_edge(x_map["R1"], seed_to_y[hi], x_map["R2"], y_r2)
                add_edge(x_map["R1"], seed_to_y[lo], x_map["R2"], y_r2)

                # R2 summary node: top candidate + probability of reaching CF from this slot.
                candidates = [seed_map[hi], seed_map[lo]]
                # Each team's p(reach conf finals) summed across whichever team fills slot.
                slot_team = max(candidates, key=lambda t: results[spec.team_id_to_name[t]]["conf_finals"])
                slot_name = spec.team_id_to_name[slot_team]
                p_cf = results[slot_name]["conf_finals"]
                label = f"{slot_name.split()[-1]}*<br>→CF: {p_cf*100:.1f}%"
                hover_cands = "<br>".join(
                    f"{spec.team_id_to_name[t].split()[-1]}: "
                    f"reach R2 {results[spec.team_id_to_name[t]]['second_round']*100:.1f}%"
                    for t in candidates
                )
                add_node(x_map["R2"], y_r2, slot_team, label, f"R2 slot<br>{hover_cands}")

            # R2 -> CF edges.
            for i in range(2):
                y_cf = Y_CF[i]
                add_edge(x_map["R2"], Y_R2[i * 2], x_map["CF"], y_cf)
                add_edge(x_map["R2"], Y_R2[i * 2 + 1], x_map["CF"], y_cf)

                # CF slot node — top team from the four possibilities.
                r1_idx_a, r1_idx_b = i * 2, i * 2 + 1
                slot_teams = [
                    seed_map[r1_pairs[r1_idx_a][0]],
                    seed_map[r1_pairs[r1_idx_a][1]],
                    seed_map[r1_pairs[r1_idx_b][0]],
                    seed_map[r1_pairs[r1_idx_b][1]],
                ]
                slot_team = max(slot_teams, key=lambda t: results[spec.team_id_to_name[t]]["finals"])
                slot_name = spec.team_id_to_name[slot_team]
                p_f = results[slot_name]["finals"]
                label = f"{slot_name.split()[-1]}*<br>→Finals: {p_f*100:.1f}%"
                hover_cands = "<br>".join(
                    f"{spec.team_id_to_name[t].split()[-1]}: "
                    f"reach CF {results[spec.team_id_to_name[t]]['conf_finals']*100:.1f}%"
                    for t in slot_teams
                )
                add_node(x_map["CF"], y_cf, slot_team, label, f"CF slot<br>{hover_cands}")

            # CF -> F edges + Finals slot node.
            add_edge(x_map["CF"], Y_CF[0], x_map["F"], Y_F[0])
            add_edge(x_map["CF"], Y_CF[1], x_map["F"], Y_F[0])

            conf_teams = [tid for tid in seed_map.values()]
            slot_team = max(conf_teams, key=lambda t: results[spec.team_id_to_name[t]]["championship"])
            slot_name = spec.team_id_to_name[slot_team]
            p_c = results[slot_name]["championship"]
            label = f"{slot_name.split()[-1]}*<br>Champ: {p_c*100:.1f}%"
            hover_cands = "<br>".join(
                f"{spec.team_id_to_name[t].split()[-1]}: Finals "
                f"{results[spec.team_id_to_name[t]]['finals']*100:.1f}%"
                for t in conf_teams
            )
            add_node(x_map["F"], Y_F[0], slot_team, label, f"{conf} Finals<br>{hover_cands}")

        render_conference("West", X_WEST, west_order)
        render_conference("East", X_EAST, east_order)

        # Central championship node.
        add_edge(X_WEST["F"], Y_F[0], X_CHAMP, Y_F[0])
        add_edge(X_EAST["F"], Y_F[0], X_CHAMP, Y_F[0])
        best_name = max(champ_probs.items(), key=lambda kv: kv[1])[0]
        add_node(
            X_CHAMP,
            Y_F[0],
            None,
            f"🏆<br>{best_name.split()[-1]}<br>{champ_probs[best_name]*100:.1f}%",
            f"Championship favorite: {best_name} ({champ_probs[best_name]*100:.2f}%)",
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="rgba(120,120,120,0.45)", width=1.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=node_text,
                textposition="middle center",
                textfont=dict(size=10, color="white"),
                hovertext=node_hover,
                hoverinfo="text",
                marker=dict(
                    size=68,
                    color=node_color,
                    colorscale="Viridis",
                    cmin=0.0,
                    cmax=1.0,
                    line=dict(color="black", width=1),
                    colorbar=dict(title="Champ. prob<br>(normalized)", thickness=12, len=0.6),
                ),
                showlegend=False,
            )
        )

        fig.update_layout(
            title=f"2025 NBA Playoff Bracket — Championship Probabilities (favorite: {best_name} {champ_probs[best_name]*100:.1f}%)",
            plot_bgcolor="rgb(245,245,245)",
            paper_bgcolor="white",
            xaxis=dict(visible=False, range=[-5, 5]),
            yaxis=dict(visible=False, range=[-1, 8]),
            height=720,
            margin=dict(l=20, r=20, t=60, b=20),
        )

        # Round labels.
        for label, x in [
            ("West R1", -4), ("West R2", -3), ("West CF", -2), ("West F", -1),
            ("Champion", 0),
            ("East F", 1), ("East CF", 2), ("East R2", 3), ("East R1", 4),
        ]:
            fig.add_annotation(x=x, y=7.9, text=f"<b>{label}</b>", showarrow=False, font=dict(size=11))

        out_path = os.path.join(self.output_dir, filename)
        fig.write_html(out_path, include_plotlyjs="cdn")
        logger.info("Wrote bracket visualization to %s", out_path)
        return out_path
