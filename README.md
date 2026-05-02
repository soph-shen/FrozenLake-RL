# Enhanced Frozen Lake: RL with Partial Observability & Wind

This project extends the classic Frozen Lake environment by introducing complex dynamics that challenge standard Reinforcement Learning approaches. The project focuses on handling Partially Observable Markov Decision Processes (POMDPs) and non-stationary environment dynamics.

## Key Features
1. **Partial Observability**: The agent sees a local window rather than the full grid.
2. **Non-Stationary Wind**: Stochastic wind shifts that the agent must learn to account for.
3. **Sparse & Penalized Rewards**: High stakes for success and significant penalties for failures and inefficiency.

---

## Directory Structure

### `map8/` (8x8 Grid)
* **`enhanced_frozen_lake.py`**: The upgraded environment.
    * **View**: 5×5 local observation window.
    * **Dynamics**: Random map regeneration per episode; 12 holes (guaranteed solvable via BFS); 5% wind probability.
    * **Rewards**: Goal (+15.0), Hole (-2.0), Living Penalty (-0.1).
* **`drqn_agent.py`**: Deep Recurrent Q-Network. Wind state is fed through a dedicated embedding layer and concatenated with vision features before the LSTM.
* **`qr_drqn_agent.py`**: Quantile Regression DRQN. Predicts 51 quantiles of the return distribution to better manage stochastic wind uncertainty.
* **`rnd_drqn_agent.py`**: RND-DRQN. Adds a curiosity bonus via a "target" and "predictor" network to encourage exploration of unvisited states.
* **`evaluate_models.py`**: Head-to-head evaluation script. Automatically benchmarks all `.pth` models over 100 fixed-seed episodes and prints a comparison table (Success Rate, Reward, Steps).
* **`play_agent.py`**: Interactive visualization using Matplotlib. Shows a dual-view (8×8 global and 5×5 local) with wind and action indicators.

---

## Model Registry

| Model Type | Map Size | Features |
| :--- | :--- | :--- |
| **DRQN** | 8x8 | Wind Embedding |
| **QR-DRQN** | 8x8 | 51 Quantiles |
| **RND-DRQN** | 8x8 | Curiosity Bonus |
| **DRQN** | 16x16 | Curriculum Learning |

---

## Usage
### Setup
uv sync
### Run (8x8 Map)
cd map8_v5
uv run python play_agent.py       # Watch a trained agent play after picking model from a list
uv run python evaluate_models.py   # Benchmark all .pth models head-to-head over 100 fixed-seed episodes and print a comparison table
