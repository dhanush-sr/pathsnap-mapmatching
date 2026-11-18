"""Train and compare the six candidate classifiers defined in pathsnap/models/.

- 5-fold GroupKFold over the 80 training trips (a trip never spans folds)
- refit on all training trips, evaluate on the 10 student-testing trips
- outputs: results/cv_folds.csv, results/model_comparison.csv, results/plots/*, artifacts/models/*
"""
import json
import os
import sys

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT
from pathsnap.features import FEATURES
from pathsnap.models import gaussian_nb, knn, lightgbm_model, logistic_regression, mlp, random_forest
from pathsnap.training import SEED, load, run

RES, PLOTS = (os.path.join(ROOT, p) for p in ("results", "results/plots"))
ALL = [logistic_regression, gaussian_nb, knn, random_forest, lightgbm_model, mlp]


def main():
    os.makedirs(PLOTS, exist_ok=True)
    tr, te = load()
    summary, folds, probas = [], [], {}
    for spec in ALL:
        row, f, p = run(spec, tr, te)
        summary.append(row); folds += f; probas[spec.NAME] = p
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
