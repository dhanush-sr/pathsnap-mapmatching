# PathSnap: Distributed Map-Matching & Route Reconstruction

PathSnap matches noisy 1 Hz GPS observations to a road network and reconstructs the driven route.
A LightGBM candidate classifier scores the nearest road arcs for every GPS point, and an HMM decoded
with Viterbi turns those scores into one connected route. The system runs on two laptops:

| Laptop | Role | Runs |
|---|---|---|
| **Laptop 1** | Backend | FastAPI service: feature extraction, classifier, HMM/Viterbi, road network, server-side scoring |
| **Laptop 2** | Frontend | Evaluator UI: lists the ten evaluation trips, sends the chosen trip to Laptop 1, shows the route and scores |

## Results

| Stage | Student-test result |
|---|---|
| Best untuned classifier (LightGBM) | Top-1 0.905, macro F1 0.941, log loss 0.062 |
| Bayesian-tuned LightGBM (40 Optuna TPE trials) | Top-1 0.907, macro F1 0.943, log loss 0.061 |
| Classifier only → classifier + HMM (mean of 10 trips) | point accuracy 0.918 → 0.954, route continuity 0.770 → 0.915, route F1 0.810 → 0.979 |

The HMM improves every one of the ten student-test trips. Decoder parameters were tuned on
out-of-fold probabilities of the training trips only.

## Repository layout

| Path | Contents |
|---|---|
| `pathsnap/data.py` | Raw-file loading, cleaning, local metric projection, road-graph construction |
| `pathsnap/features.py` | Route-to-observation label alignment, candidate generation, 16 candidate features |
| `pathsnap/evaluate.py` | Macro F1, precision, recall, log loss, Top-1 accuracy, route continuity |
| `pathsnap/models/` | One file per algorithm: Logistic Regression, Gaussian NB, KNN, Random Forest, LightGBM, MLP |
| `pathsnap/training.py` | Shared GroupKFold cross-validation, student-test evaluation and MLflow logging |
| `pathsnap/hmm.py` | HMM emission/transition model, Viterbi decoder, route filling, route metrics |
| `pathsnap/tracking.py` | MLflow configuration (local SQLite store and model registry) |
| `scripts/eda.py` | Dataset inventory, missing values, trajectory and network plots, Folium EDA map |
| `scripts/build_dataset.py` | Candidate feature tables for train and student_test |
| `scripts/feature_selection.py` | Feature correlation, importance and group ablation |
| `scripts/train_models.py` | Six-model comparison and plots |
| `scripts/bayes_opt.py` | Optuna TPE optimisation of LightGBM, tuned vs untuned comparison |
| `scripts/evaluate_hmm.py` | Decoder tuning and classifier-only vs HMM comparison |
| `scripts/hmm_baseline.py` | Nearest road, distance-only HMM, classifier only and classifier + HMM on student_test |
| `scripts/make_maps.py` | Folium maps: GPS, network, reference, classifier-only and HMM routes |
| `scripts/log_to_mlflow.py` | Records all results in MLflow and registers the final model |
| `api/` | **Laptop 1**: FastAPI backend (`run_api.py`, `app.py`) |
| `ui/` | **Laptop 2**: evaluator interface (`index.html`, `run_ui.py`) |
| `artifacts/` | Final tuned model, model and HMM configuration, comparison models |
| `results/` | Result tables, logs, plots and Folium maps |
| `report/` | Technical report (`PathSnap_Technical_Report.pdf`), its HTML source and figures |

## Setup (both laptops)

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Laptop 2 only needs Python itself; it does not need the requirements.

### Data

The dataset is not in the repository. On Laptop 1, place it like this:

```
pathsnap/
├── Student folder/
│   ├── train/            80 training trips
│   └── student_test/     10 student-testing trips
└── evaluator_test/       10 evaluator trips (only needed on Laptop 1 for the demo)
```

Set `PATHSNAP_DATA` to use another location for `Student folder`.

## Running the demo

Both laptops must be on the same network (same Wi-Fi or hotspot).

### Laptop 1: backend API

```bash
python api/run_api.py
```

The terminal prints a banner with the address to give Laptop 2, for example
`enter http://192.168.1.20:8000 in the UI on Laptop 2`. Every request from Laptop 2 is logged
there with its source IP, trip, processing time and scores; this is the log to show the evaluator.

| Option | Effect |
|---|---|
| `PATHSNAP_EVAL_DIR=<folder>` | Serve trips from another folder (default `evaluator_test/`, else `student_test`) |
| `PATHSNAP_MODEL_URI=models:/pathsnap-lightgbm/1` | Load the model from the MLflow registry instead of `artifacts/` |
| `--port 9000` | Use another port |

If Laptop 2 cannot connect, allow incoming connections for Python in the firewall prompt.

### Laptop 2: evaluator UI

```bash
python ui/run_ui.py --api http://<laptop-1-ip>:8000
```

The browser opens with the Laptop 1 address filled in. Pick one of the ten trips and press
**Run map matching**. The address can also be changed in the UI's *Laptop 1 · API endpoint* field.

### What stays on Laptop 1

The prediction pipeline loads each trip without its `.route` file. The reference route is read
only by the scoring step on Laptop 1, and the response contains the predicted route, per-point
matches and summary scores, never the reference route.

## API

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/health` | | status, number of trips, model, HMM parameters |
| GET | `/trips` | | `[{trip_id, n_points, duration_s}]` |
| POST | `/predict` | `{"trip_id": "00000015"}` | predicted route polyline, per-point matched arc and probability, scores |

Interactive docs: `http://<laptop-1-ip>:8000/docs`.

## Reproducing the experiments

```bash
python scripts/eda.py
python scripts/build_dataset.py
python scripts/feature_selection.py
python scripts/train_models.py          # or one model: python -m pathsnap.models.lightgbm_model
python scripts/bayes_opt.py
python scripts/evaluate_hmm.py
python scripts/hmm_baseline.py
python scripts/make_maps.py
python scripts/log_to_mlflow.py
```

The report is rebuilt from `report/report.html` with `sh report/build_pdf.sh`.

All random seeds are 42. The Random Forest and KNN model files are too large for GitHub;
`train_models.py` recreates them.

## Experiment tracking (MLflow)

Training runs started from `pathsnap/training.py` are logged to MLflow automatically.
`scripts/log_to_mlflow.py` records every saved result (model comparison, the 40 Optuna trials,
the HMM grid and the classifier vs HMM comparison) and registers the tuned model as
`pathsnap-lightgbm`.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open `http://localhost:5000`. The store (`mlflow.db`, `mlruns/`) is local and not committed.
