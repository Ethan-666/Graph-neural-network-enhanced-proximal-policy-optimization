# Dynamic Bus Bridging Dispatching for Metro Disruptions Using Deep Reinforcement Learning

This repository is the open-source implementation for the paper **Dynamic Bus Bridging Dispatching for Metro Disruptions Using Deep Reinforcement Learning**.

The code implements an event-driven GNN-PPO dispatcher for metro disruption bus-bridging experiments. It supports vehicle-level dispatching decisions among regular service, depot standby, temporary OD bridging tasks, and return movements, with feasible-action masking and PPO actor-critic training.

## Environment

Python 3.10 or later is required. The main dependencies are listed in `requirements.txt` and `pyproject.toml`:

- `numpy`
- `pandas`
- `torch`
- `matplotlib`
- `tqdm`
- `openpyxl`

## Install

From the project root:

```powershell
cd D:\pycharmProject\Revision8.26\GNN-PPO
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Data

The prepared data directory should contain:

```text
data/static/station_features.xlsx
data/static/line_features.xlsx
data/static/bus_departure_info_16-20.xlsx
data/processed/od_by_time/<date>_<HHMMSS>.csv
```

For the paper-aligned GNN graph, also provide:

```text
data/static/multimodal_nodes.csv
data/static/multimodal_edges.csv
```

Optional regular-bus queue node features can be provided through:

```text
data/static/regular_bus_node_queues.csv
```

The required graph file format is documented in:

```text
data/static/PAPER_GRAPH_SCHEMA.md
```

If the source data are outside this repository, copy them into the local `data/` tree with:

```powershell
python -m gnn_ppo.prepare_data --source-root ..
```

## Train

Depot-supported setting:

```powershell
python -m gnn_ppo.train --run-name gnn_ppo_depot_seed42
```

Bus-line-only setting:

```powershell
python -m gnn_ppo.train --no-depot --run-name gnn_ppo_bus_line_seed42
```

For a short smoke run:

```powershell
python -m gnn_ppo.train --episodes 5 --run-name smoke
```

Training outputs are saved under:

```text
outputs/<run-name>/
```

Key files include:

- `checkpoint.pt`
- `rewards_history.csv`
- `loss_history.csv`
- `reward_loss_curves.png`
- `reward_loss_curves.pdf`
- `action_catalog.csv`
- `config_used.json`

## Evaluate

Evaluate a trained checkpoint:

```powershell
python -m gnn_ppo.evaluate --checkpoint outputs\gnn_ppo_depot_seed42\checkpoint.pt
```

Evaluate the smoke-run checkpoint:

```powershell
python -m gnn_ppo.evaluate --checkpoint outputs\smoke\checkpoint.pt
```

Evaluation outputs are written to:

```text
outputs/<run-name>/evaluation/
```

Key files include:

- `evaluation_summary.csv`
- `actions_HHMMSS.csv`
- `config_used_evaluation.json`

`evaluation_summary.csv` is the main inference summary table.

## Configuration

Default settings are in:

```text
configs/default.json
```

Important fields include:

- `data`: OD date, static data paths, output directory, episode duration, and evaluation windows.
- `env`: capacity, reward weights, depot settings, dispatch-time threshold, and demand-coverage filtering.
- `graph`: paper-aligned multimodal graph inputs.
- `model`: PPO and GNN hyperparameters.
- `train`: training episodes, update interval, and random seed.
