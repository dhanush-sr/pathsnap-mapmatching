"""MLflow experiment tracking for PathSnap.

Runs are stored locally: metadata in ./mlflow.db (SQLite, needed for the model registry) and
artifacts in ./mlruns. Browse them with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

Set PATHSNAP_MLFLOW=0 to switch tracking off, or MLFLOW_TRACKING_URI to use another server.
"""
import os

from .data import ROOT

ENABLED = os.environ.get("PATHSNAP_MLFLOW", "1") != "0"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}")
ARTIFACT_ROOT = os.path.join(ROOT, "mlruns")
REGISTERED_MODEL = "pathsnap-lightgbm"


def mlflow_experiment(name):
    """Point MLflow at the project store and select (or create) an experiment. Returns the module or None."""
    if not ENABLED:
        return None
    try:
        import mlflow
    except ImportError:
        return None
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=f"file://{ARTIFACT_ROOT}/{name}")
    mlflow.set_experiment(name)
    return mlflow


def clean_params(params):
    """MLflow params must be short strings; drop nested estimators and None values."""
    out = {}
    for k, v in params.items():
        if v is None or hasattr(v, "get_params"):
            continue
        out[k] = str(v)[:250]
    return out
