"""Train and compare candidate classifiers from six algorithm families.

- 5-fold GroupKFold over the 80 training trips (a trip never spans folds)
- refit on all training trips, evaluate on the 10 student-testing trips
- outputs: results/cv_folds.csv, results/model_comparison.csv, results/plots/*, artifacts/models/*
"""
import json
import os
import sys
import time

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT, get_trip
from pathsnap.evaluate import argmax_decode, candidate_metrics, continuity
from pathsnap.features import FEATURES

SEED = 42
RES, PLOTS, MODELS = (os.path.join(ROOT, p) for p in ("results", "results/plots", "artifacts/models"))

MODEL_SPECS = {
    "LogisticRegression": ("Linear", "C=1.0, L2, lbfgs, standardised",
                           make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))),
    "GaussianNB": ("Probabilistic", "var_smoothing=1e-9", GaussianNB()),
    "KNN": ("Instance-based", "k=25, distance-weighted, standardised",
            make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1))),
    "RandomForest": ("Ensemble (bagging)", "200 trees, min_samples_leaf=5, sqrt features",
                     RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
    "LightGBM": ("Boosting", "400 trees, lr=0.05, 31 leaves",
                 lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                    random_state=SEED, verbose=-1)),
    "MLP": ("Neural network", "64-32 ReLU, adam, alpha=1e-4, early stopping",
            make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                          early_stopping=True, max_iter=60, random_state=SEED))),
}


def load():
    tr = pd.read_pickle(os.path.join(RES, "features_train.pkl"))
    te = pd.read_pickle(os.path.join(RES, "features_student_test.pkl"))
    return tr, te


def route_continuity(model_proba, te):
    vals = []
    for t, d in te.groupby("trip"):
        dec = argmax_decode(d.reset_index(drop=True), model_proba[d.index.values])
        vals.append(continuity(get_trip("student_test", t), [dec[k] for k in sorted(dec)]))
    return float(np.mean(vals))


def cross_validate(name, est, tr):
    rows = []
    for fold, (a, b) in enumerate(GroupKFold(n_splits=5).split(tr, groups=tr["trip"])):
        m = clone(est).fit(tr.iloc[a][FEATURES], tr.iloc[a]["label"])
        p = m.predict_proba(tr.iloc[b][FEATURES])[:, 1]
        r = candidate_metrics(tr.iloc[b], p)
        p_tr = m.predict_proba(tr.iloc[a][FEATURES])[:, 1]
        r["train_macro_f1"] = candidate_metrics(tr.iloc[a], p_tr)["macro_f1"]
        rows.append({"model": name, "fold": fold, **r})
        print(f"  {name} fold {fold}: macroF1={r['macro_f1']:.4f} top1={r['top1_accuracy']:.4f}")
    return rows


def main():
    for d in (PLOTS, MODELS):
        os.makedirs(d, exist_ok=True)
    tr, te = load()
    te_rel = te[te["reliable"]]
    folds, summary, probas = [], [], {}
    for name, (family, cfg, est) in MODEL_SPECS.items():
        print(name)
        folds += cross_validate(name, est, tr)
        t0 = time.time()
        m = clone(est).fit(tr[FEATURES], tr["label"])
        train_time = time.time() - t0
        t0 = time.time()
        p = m.predict_proba(te[FEATURES])[:, 1]
        infer = time.time() - t0
        probas[name] = p
        joblib.dump(m, os.path.join(MODELS, f"{name}.joblib"))
        r = candidate_metrics(te_rel, p[te["reliable"].values])
        cv = pd.DataFrame([f for f in folds if f["model"] == name])
        summary.append({
            "model": name, "family": family, "config": cfg,
            "cv_macro_f1": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
            "cv_top1": cv.top1_accuracy.mean(), "cv_log_loss": cv.log_loss.mean(),
            "cv_train_macro_f1": cv.train_macro_f1.mean(),
            "test_top1": r["top1_accuracy"], "test_macro_f1": r["macro_f1"],
            "test_precision": r["precision"], "test_recall": r["recall"], "test_log_loss": r["log_loss"],
            "route_continuity": route_continuity(p, te),
            "train_time_s": train_time, "infer_time_s": infer,
            "infer_ms_per_obs": 1000 * infer / te.groupby(["trip", "obs"]).ngroups,
        })
        print(pd.Series(summary[-1]).to_string())
    pd.DataFrame(folds).to_csv(os.path.join(RES, "cv_folds.csv"), index=False)
    S = pd.DataFrame(summary)
    S.to_csv(os.path.join(RES, "model_comparison.csv"), index=False)
    plots(S, pd.DataFrame(folds), te, probas)
    histories(tr)


def plots(S, F, te, probas):
    names = S.model.tolist()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, S.cv_macro_f1, yerr=S.cv_macro_f1_std, capsize=4, color="#4C72B0")
    ax.set_ylabel("CV macro F1 (5-fold GroupKFold)"); ax.set_ylim(0.5, 1)
    for i, v in enumerate(S.cv_macro_f1):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(f"{PLOTS}/cv_macro_f1.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, S.test_top1, w, label="Top-1 accuracy")
    ax.bar(x + w / 2, S.test_macro_f1, w, label="Macro F1")
    ax.set_xticks(x, names, rotation=20); ax.set_ylim(0.5, 1); ax.legend(); ax.set_title("Student-test partition")
    plt.tight_layout(); plt.savefig(f"{PLOTS}/test_top1_f1.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, S.test_log_loss, color="#C44E52"); ax.set_ylabel("Student-test log loss"); ax.set_yscale("log")
    plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(f"{PLOTS}/log_loss.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    F.boxplot(column="macro_f1", by="model", ax=ax); plt.suptitle(""); ax.set_title("CV macro F1 per fold")
    plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(f"{PLOTS}/cv_fold_spread.png", dpi=150); plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    rel = te["reliable"].values
    for ax, n in zip(axes.ravel(), names):
        ConfusionMatrixDisplay.from_predictions(te.label[rel], (probas[n][rel] >= 0.5).astype(int), ax=ax,
                                                colorbar=False, display_labels=["other", "true arc"])
        ax.set_title(n)
    plt.tight_layout(); plt.savefig(f"{PLOTS}/confusion_matrices.png", dpi=150); plt.close()


def histories(tr):
    """Epoch-wise train/validation log loss for the two models that train iteratively."""
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(tr, groups=tr["trip"]))
    Xa, ya, Xb, yb = tr.iloc[a][FEATURES], tr.iloc[a]["label"], tr.iloc[b][FEATURES], tr.iloc[b]["label"]
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1)
    m.fit(Xa, ya, eval_set=[(Xa, ya), (Xb, yb)], eval_names=["train", "valid"], eval_metric="binary_logloss")
    ev = m.evals_result_
    hist = {"lightgbm": {k: ev[k]["binary_logloss"] for k in ev}}
    sc = StandardScaler().fit(Xa)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, random_state=SEED, batch_size=200)
    tl, vl = [], []
    for _ in range(40):
        mlp.partial_fit(sc.transform(Xa), ya, classes=[0, 1])
        tl.append(log_loss(ya, mlp.predict_proba(sc.transform(Xa))[:, 1]))
        vl.append(log_loss(yb, mlp.predict_proba(sc.transform(Xb))[:, 1]))
    hist["mlp"] = {"train": tl, "valid": vl}
    json.dump(hist, open(os.path.join(RES, "training_histories.json"), "w"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (k, xl) in zip(axes, [("lightgbm", "boosting iteration"), ("mlp", "epoch")]):
        ax.plot(hist[k]["train"], label="train"); ax.plot(hist[k]["valid"], label="validation (held-out trips)")
        ax.set_title(f"{k.upper()} log loss"); ax.set_xlabel(xl); ax.legend()
    plt.tight_layout(); plt.savefig(f"{PLOTS}/training_curves.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
