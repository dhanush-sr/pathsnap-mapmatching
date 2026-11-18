# PathSnap: Map-Matching & Route Reconstruction

Matching noisy 1 Hz GPS observations to a road network and reconstructing the driven route.

## Status

| Stage | Status |
|---|---|
| Data loading, cleaning and projection | Done |
| Exploratory data analysis | Done |
| Reference-label alignment, candidate generation, feature engineering | Done |
| Model training and comparison | In progress |
| Bayesian hyperparameter optimisation | Planned |
| HMM / Viterbi route reconstruction | Planned |
| FastAPI backend and evaluator UI | Planned |

## Layout

| Path | Contents |
|---|---|
| `pathsnap/data.py` | Raw-file loading, cleaning, local metric projection, road-graph construction |
| `pathsnap/features.py` | Route-to-observation alignment, candidate generation, feature engineering |
| `scripts/eda.py` | Dataset inventory, missing-value analysis, trajectory and network plots, Folium map |
| `scripts/build_dataset.py` | Candidate feature tables for the train and student_test partitions |
| `results/eda/` | EDA tables and plots |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the dataset so the partitions are at `Student folder/train` and `Student folder/student_test`,
or set `PATHSNAP_DATA` to the folder that contains them.

```bash
python scripts/eda.py
python scripts/build_dataset.py
```
