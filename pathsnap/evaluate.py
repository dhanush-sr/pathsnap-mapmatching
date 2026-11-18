"""Shared metrics for candidate classifiers and decoded arc sequences."""
import numpy as np
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, accuracy_score


def top1_accuracy(df, proba):
    """Observation-level Top-1: the highest-scoring candidate must be the reference arc.
    Observations whose reference arc is not among the candidates count as errors."""
    d = df[["trip", "obs", "label"]].assign(p=proba)
    best = d.sort_values("p", ascending=False).drop_duplicates(["trip", "obs"])
    return float(best["label"].mean())


def candidate_metrics(df, proba, thr=0.5):
    y = df["label"].values
    yhat = (proba >= thr).astype(int)
    return {
        "macro_f1": f1_score(y, yhat, average="macro"),
        "precision": precision_score(y, yhat, average="macro", zero_division=0),
        "recall": recall_score(y, yhat, average="macro", zero_division=0),
        "accuracy": accuracy_score(y, yhat),
        "log_loss": log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]),
        "top1_accuracy": top1_accuracy(df, proba),
    }


def candidate_lists(df, proba):
    """Group the long candidate table into per-observation arrays (arcs, probs, frac)."""
    df = df.assign(p=proba)
    order = np.argsort(df["obs"].values, kind="stable")
    obs = df["obs"].values[order]
    cuts = np.flatnonzero(np.diff(obs)) + 1
    groups = np.split(order, cuts)
    ids = obs[np.r_[0, cuts]] if len(obs) else np.array([], int)
    arcs = df["arc"].values
    p = df["p"].values
    f = df["proj_frac"].values
    return ids, [arcs[g] for g in groups], [p[g] for g in groups], [f[g] for g in groups]


def argmax_decode(df, proba):
    ids, A, Pr, _ = candidate_lists(df, proba)
    return {i: int(a[p.argmax()]) for i, a, p in zip(ids.tolist(), A, Pr)}


def collapse(seq):
    seq = list(seq)
    return [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]


def continuity(trip, arc_seq):
    """Share of consecutive distinct predicted arcs that are directly connected."""
    s = collapse(arc_seq)
    if len(s) < 2:
        return 1.0
    s = np.array(s)
    return float(np.mean(trip.arcs[s[:-1], 1] == trip.arcs[s[1:], 0]))
