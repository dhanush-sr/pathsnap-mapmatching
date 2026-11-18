"""Shared training / evaluation routine used by every algorithm file in pathsnap/models/.

- 5-fold GroupKFold over the 80 training trips (a trip never spans folds)
- refit on all training trips, evaluate on the 10 student-testing trips
"""
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .data import ROOT, get_trip
from .evaluate import argmax_decode, candidate_metrics, continuity
from .features import FEATURES
from .tracking import clean_params, mlflow_experiment

SEED = 42
RES = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "artifacts", "models")


def load():
    tr = pd.read_pickle(os.path.join(RES, "features_train.pkl"))
    te = pd.read_pickle(os.path.join(RES, "features_student_test.pkl"))
    return tr, te


def route_continuity(proba, te):
    """Mean share of connected consecutive arcs when each point takes its top candidate."""
    vals = []
    for t, d in te.groupby("trip"):
        dec = argmax_decode(d.reset_index(drop=True), proba[d.index.values])
        vals.append(continuity(get_trip("student_test", t), [dec[k] for k in sorted(dec)]))
    return float(np.mean(vals))


def cross_validate(name, est, tr):
    rows = []
    for fold, (a, b) in enumerate(GroupKFold(n_splits=5).split(tr, groups=tr["trip"])):
        m = clone(est).fit(tr.iloc[a][FEATURES], tr.iloc[a]["label"])
        r = candidate_metrics(tr.iloc[b], m.predict_proba(tr.iloc[b][FEATURES])[:, 1])
        r["train_macro_f1"] = candidate_metrics(tr.iloc[a], m.predict_proba(tr.iloc[a][FEATURES])[:, 1])["macro_f1"]
        rows.append({"model": name, "fold": fold, **r})
        print(f"  {name} fold {fold}: macroF1={r['macro_f1']:.4f} top1={r['top1_accuracy']:.4f}")
    return rows


def run(spec, tr=None, te=None):
    """Cross-validate, refit, evaluate on student_test and save one algorithm.

    spec: a module from pathsnap.models with NAME, FAMILY, CONFIG and build().
    Returns (summary row, per-fold rows, student-test probabilities).
    """
    if tr is None:
        tr, te = load()
    os.makedirs(MODELS, exist_ok=True)
    est = spec.build()
    print(spec.NAME)
    folds = cross_validate(spec.NAME, est, tr)
    t0 = time.time(); m = clone(est).fit(tr[FEATURES], tr["label"]); train_time = time.time() - t0
    t0 = time.time(); p = m.predict_proba(te[FEATURES])[:, 1]; infer = time.time() - t0
    joblib.dump(m, os.path.join(MODELS, f"{spec.NAME}.joblib"))
    rel = te["reliable"].values
    r = candidate_metrics(te[rel], p[rel])
    cv = pd.DataFrame(folds)
    row = {
        "model": spec.NAME, "family": spec.FAMILY, "config": spec.CONFIG,
        "cv_macro_f1": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
        "cv_top1": cv.top1_accuracy.mean(), "cv_log_loss": cv.log_loss.mean(),
        "cv_train_macro_f1": cv.train_macro_f1.mean(),
        "test_top1": r["top1_accuracy"], "test_macro_f1": r["macro_f1"],
        "test_precision": r["precision"], "test_recall": r["recall"], "test_log_loss": r["log_loss"],
        "route_continuity": route_continuity(p, te),
        "train_time_s": train_time, "infer_time_s": infer,
        "infer_ms_per_obs": 1000 * infer / te.groupby(["trip", "obs"]).ngroups,
    }
    print(pd.Series(row).to_string())
    log_run(spec, est, row, os.path.join(MODELS, f"{spec.NAME}.joblib"))
    return row, folds, p


def log_run(spec, est, row, model_path):
    """Record one trained model in the MLflow experiment 'pathsnap-model-comparison'."""
    mlflow = mlflow_experiment("pathsnap-model-comparison")
    if mlflow is None:
        return
    with mlflow.start_run(run_name=spec.NAME):
        mlflow.set_tags({"family": spec.FAMILY, "source": "live training run"})
        mlflow.log_params({"config": spec.CONFIG, "cv": "5-fold GroupKFold by trip", "seed": SEED,
                           **clean_params(est.get_params(deep=True))})
        mlflow.log_metrics({k: float(v) for k, v in row.items() if isinstance(v, (int, float))})
        if os.path.getsize(model_path) < 50e6:
            mlflow.log_artifact(model_path, "model")
