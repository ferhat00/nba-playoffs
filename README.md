# NBA Playoff Predictor

Probabilistic prediction system for the NBA Playoffs. Given the current season's team and player statistics it runs a Bayesian player-impact model, builds a 21-dimensional team state vector, trains an XGBoost matchup classifier, and runs a Monte Carlo bracket simulation to produce championship win probabilities for every playoff team.

Results are printed to the console and saved as an interactive Plotly bracket visualization in `output/`.

---

## How it works — end to end

```
nba_api / ESPN / balldontlie
          │
          ▼
   1. Data fetching        fetcher.py       Pull team stats, player stats, game logs,
                                            clutch stats, and historical matchup data.
                                            Results are joblib-cached; fallback data
                                            is used if the API is unavailable.
          │
          ▼
   2. Bayesian player      player_          Hierarchical Bayesian model (PyMC + NUTS
      evaluator            evaluator.py     MCMC). Partially pools each player's
                                            offensive/defensive impact toward their
                                            team mean, producing posterior impact
                                            scores with uncertainty estimates.
          │
          ▼
   3. Team state           team_            Collapses team stats + player impacts into
      aggregation          aggregator.py    a 21-dimensional vector per team. Includes
                                            an LSTM trained on the last-20-game log to
                                            capture momentum.
          │
          ▼
   4. Matchup predictor    matchup_         XGBoost classifier trained on the difference
                           predictor.py     between two teams' 21-dim vectors. Uses real
                                            historical matchup data when available;
                                            otherwise generates 5 000 synthetic samples.
                                            Outputs a single-game win probability.
          │
          ▼
   5. Monte Carlo          bracket_         Simulates 100 000 full 16-team playoff
      bracket simulation   simulator.py     brackets. Each series is best-of-7 with a
                                            ±2.5 % home-court adjustment per game.
                                            Worker batches run in parallel via
                                            multiprocessing.Pool.
          │
          ▼
   6. Output               output/          Prints a championship-probability table and
                                            saves an interactive Plotly bracket HTML.
```

### Models in detail

| Layer | Method | Library |
|---|---|---|
| Player impact | Hierarchical Bayesian NUTS MCMC — partial pooling of per-player off/def impact toward team mean | PyMC ≥ 5.10, numpyro (GPU) |
| Team momentum | 2-layer LSTM (hidden=32) trained on rolling 20-game sequences of `[off_rtg, def_rtg, pace]` | PyTorch |
| Matchup probability | XGBoost gradient-boosted classifier on 21-dim vector differences; holdout AUC reported | XGBoost / scikit-learn fallback |
| Bracket simulation | Parallelized Monte Carlo over 100 000 independently seeded bracket draws | NumPy + multiprocessing |

### 21-dimensional team state vector

| Index | Feature | Description |
|---|---|---|
| 0 | `elo_rating` | Initialized from season net rating |
| 1 | `injury_adjusted_elo` | Elo minus 50 pts per injured top-5 impact player |
| 2 | `lstm_momentum` | LSTM final hidden-state mean (tanh-activated) |
| 3 | `off_rtg_norm` | Z-scored offensive rating |
| 4 | `def_rtg_norm` | Z-scored defensive rating (sign-flipped) |
| 5 | `pace_norm` | Z-scored pace |
| 6 | `top3_player_impact` | Mean composite impact of top-3 players |
| 7 | `depth_score` | Mean composite impact of players 4–N |
| 8 | `pythagorean_win_pct` | Pythagorean expectation (exponent 16.5) |
| 9 | `overperformance` | Actual win% minus Pythagorean (regression signal) |
| 10 | `net_rtg_volatility` | Std of game-level net rating (last 20 games) |
| 11 | `off_rtg_trend` | OLS slope of offensive rating (last 20 games) |
| 12 | `close_game_win_rate` | Win% in games decided by ≤ 5 points |
| 13 | `home_win_pct` | Home-game win% (last 20 games) |
| 14 | `best_player_impact` | Maximum composite impact on the roster |
| 15 | `star_concentration` | Herfindahl index of composite impacts |
| 16 | `roster_healthy_pct` | Impact-weighted fraction of healthy players |
| 17 | `three_pt_pct_norm` | Z-scored team 3-point percentage |
| 18 | `ast_to_tov_norm` | Z-scored assist-to-turnover ratio |
| 19 | `oreb_pct_norm` | Z-scored offensive rebound rate |
| 20 | `clutch_net_rtg_norm` | Z-scored clutch-time net rating |

---

## Repository structure

```
nba_playoff_predictor/
├── main.py                  # Orchestrator — runs the full 6-step pipeline
├── requirements.txt
├── data/
│   ├── fetcher.py           # nba_api calls, caching, fallbacks, credential loading
│   └── fallback_data.py     # Hardcoded 2024-25 stats for all 16 playoff teams/players
├── models/
│   ├── player_evaluator.py  # BayesianPlayerEvaluator (PyMC MCMC)
│   ├── team_aggregator.py   # TeamStateAggregator → 21-dim vector + LSTM
│   └── matchup_predictor.py # XGBoostMatchupEngine
└── simulation/
    └── bracket_simulator.py # PlayoffSimulator (Monte Carlo + Plotly)

config.yaml                  # Runtime parameters and optional API keys (see below)
nba_playoffs_kaggle.ipynb    # Kaggle notebook — GPU-accelerated full run
output/                      # Generated files (bracket HTML)
```

---

## Running locally

### 1. Prerequisites

- Python 3.10 or later
- A virtual environment is recommended

### 2. Install dependencies

```bash
cd nba_playoff_predictor
pip install -r requirements.txt
```

PyTorch is listed as a dependency for the LSTM momentum model. Install the CPU wheel if you don't have a GPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Run

**Quick smoke test** (~60 s, 1 000 simulations, 100 MCMC draws):

```bash
python -m nba_playoff_predictor.main --quick
```

**Full run** (100 000 simulations, 500 MCMC draws):

```bash
python -m nba_playoff_predictor.main
```

**Additional flags:**

| Flag | Effect |
|---|---|
| `--quick` | 1 000 sims / 100 MCMC draws — fast sanity check |
| `--quiet` | Suppress INFO logging |
| `--refresh-cache` | Clear joblib cache and re-pull live data |

Results are printed to stdout and an interactive HTML bracket is written to `output/bracket_probabilities.html`.

### 4. Optional: live series data

Set the `BALLDONTLIE_API_KEY` environment variable to a free key from [balldontlie.io](https://www.balldontlie.io/) to enable live playoff-series results. Without it the pipeline falls back to a keyless ESPN endpoint.

```bash
export BALLDONTLIE_API_KEY=your_key_here   # Linux / macOS
$env:BALLDONTLIE_API_KEY="your_key_here"   # PowerShell
```

---

## Running on Kaggle (GPU)

The notebook `nba_playoffs_kaggle.ipynb` is configured to run on a Kaggle **P100 GPU**.

### Why P100?
The MCMC bottleneck runs via `nuts_sampler="numpyro"` (JAX/CUDA). P100's FP64 throughput is ideal for NUTS leapfrog steps. T4's FP16 strengths don't apply, and numpyro doesn't auto-shard across two T4s.

### What is GPU-accelerated?
- PyMC MCMC → numpyro (biggest speedup)
- PyTorch LSTM → `.to("cuda")`
- XGBoost intentionally left on CPU (~5 000 rows; CUDA overhead exceeds benefit)
- Monte Carlo brackets → multiprocessing (CPU parallel)

### Setup steps

1. In Kaggle notebook settings: **Accelerator = GPU P100**, **Internet = On**.
2. *(Optional)* Set up a private runtime config dataset:
   - Fill in `config.yaml` (repo root) with your parameters and API key.
   - Upload it as a **private** Kaggle dataset named `nba-playoffs` (slug: `nba-playoffs`).
   - In the notebook click **Add Data**, find your dataset, and add it.
   - The file will be read automatically from `/kaggle/input/nba-playoffs/config.yaml`.
3. Run all cells. Cell 6 loads the config; Cell 7 runs the pipeline.

### `config.yaml` reference

```yaml
balldontlie_api_key: ""   # optional — enables live series results
quick_mode: false         # true → 1 000 sims / 100 draws (smoke test)
n_simulations: 100000     # ignored when quick_mode: true
mcmc_samples: 500         # ignored when quick_mode: true
random_seed: 42
```

---

## Data sources

| Source | Credentials | Data retrieved |
|---|---|---|
| [nba_api](https://github.com/swar/nba_api) | None required | Team stats, player stats, game logs, clutch stats, pace, 3P%, AST/TOV, OREB% |
| ESPN (unofficial) | None required | Playoff series results (keyless fallback) |
| [balldontlie.io](https://www.balldontlie.io/) | Free API key (optional) | Playoff series results (preferred when key is set) |
| Hardcoded fallback | — | 2024-25 season stats for all 16 playoff teams; used if nba_api is unavailable |

---

## License

See [LICENSE](LICENSE).
