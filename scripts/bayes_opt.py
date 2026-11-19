"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected LightGBM classifier.

Objective: mean macro F1 over 5-fold GroupKFold on the 80 training trips (trip = group).
Outputs: results/bayes_trials.csv, results/bayes_best.json, results/tuned_vs_untuned.csv,
         results/plots/bayes_*.png, artifacts/final_model.joblib
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
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT
from pathsnap.evaluate import candidate_metrics
from pathsnap.features import FEATURES

SEED = 42
N_TRIALS = 40
RES = os.path.join(ROOT, "results")
UNTUNED = dict(n_estimators=400, learning_rate=0.05, num_leaves=31)

SPACE = {
    "n_estimators": ("int", 100, 1000),
    "learning_rate": ("log", 0.01, 0.2),
    "num_leaves": ("int", 15, 255),
    "min_child_samples": ("int", 10, 300),
    "subsample": ("float", 0.5, 1.0),
    "colsample_bytree": ("float", 0.4, 1.0),
    "reg_lambda": ("log", 1e-3, 10.0),
    "scale_pos_weight": ("float", 1.0, 4.0),
}


def make(params):
    return lgb.LGBMClassifier(**params, subsample_freq=1 if params.get("subsample", 1) < 1 else 0,
                              random_state=SEED, verbose=-1)


def cv(params, tr):
    rows = []
    for a, b in GroupKFold(n_splits=5).split(tr, groups=tr["trip"]):
        m = make(params).fit(tr.iloc[a][FEATURES], tr.iloc[a]["label"])
        rows.append(candidate_metrics(tr.iloc[b], m.predict_proba(tr.iloc[b][FEATURES])[:, 1]))
    return pd.DataFrame(rows).mean()


def evaluate(params, tr, te, name):
    c = cv(params, tr)
    t0 = time.time(); m = make(params).fit(tr[FEATURES], tr["label"]); fit = time.time() - t0
    rel = te[te["reliable"]]
    t0 = time.time(); p = m.predict_proba(rel[FEATURES])[:, 1]; inf = time.time() - t0
    r = candidate_metrics(rel, p)
    return m, {"model": name, "cv_macro_f1": c.macro_f1, "cv_top1": c.top1_accuracy, "cv_log_loss": c.log_loss,
               "test_top1": r["top1_accuracy"], "test_macro_f1": r["macro_f1"], "test_log_loss": r["log_loss"],
               "train_time_s": fit, "infer_time_s": inf}


def main():
    tr = pd.read_pickle(os.path.join(RES, "features_train.pkl"))
    te = pd.read_pickle(os.path.join(RES, "features_student_test.pkl"))

    def objective(trial):
        params = {}
        for k, (kind, lo, hi) in SPACE.items():
            if kind == "int":
                params[k] = trial.suggest_int(k, lo, hi)
            elif kind == "log":
                params[k] = trial.suggest_float(k, lo, hi, log=True)
            else:
                params[k] = trial.suggest_float(k, lo, hi)
        c = cv(params, tr)
        trial.set_user_attr("cv_top1", float(c.top1_accuracy))
        trial.set_user_attr("cv_log_loss", float(c.log_loss))
        return c.macro_f1

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    search_s = time.time() - t0
    df = study.trials_dataframe()
    df.to_csv(os.path.join(RES, "bayes_trials.csv"), index=False)
    best = study.best_params
    json.dump({"best_params": best, "best_cv_macro_f1": study.best_value, "n_trials": N_TRIALS,
               "search_space": {k: list(v) for k, v in SPACE.items()}, "objective": "mean 5-fold GroupKFold macro F1",
               "sampler": "TPE (seed 42)", "search_time_s": search_s},
              open(os.path.join(RES, "bayes_best.json"), "w"), indent=2)
    print("best", study.best_value, best, f"search {search_s:.0f}s")

    _, u = evaluate(UNTUNED, tr, te, "LightGBM (untuned)")
    m, t = evaluate(best, tr, te, "LightGBM (Bayesian-tuned)")
    pd.DataFrame([u, t]).to_csv(os.path.join(RES, "tuned_vs_untuned.csv"), index=False)
    print(pd.DataFrame([u, t]).T)
    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    joblib.dump(m, os.path.join(ROOT, "artifacts", "final_model.joblib"))
    json.dump({"features": FEATURES, "params": best}, open(os.path.join(ROOT, "artifacts", "model_config.json"), "w"), indent=2)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["number"], df["value"], "o", alpha=.6, label="trial")
    ax.plot(df["number"], df["value"].cummax(), "r-", label="best so far")
    ax.set_xlabel("trial"); ax.set_ylabel("CV macro F1"); ax.set_title("Optuna TPE optimisation history"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RES, "plots", "bayes_history.png"), dpi=150); plt.close()
    imp = optuna.importance.get_param_importances(study)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(list(imp)[::-1], list(imp.values())[::-1]); ax.set_title("Hyperparameter importance (fANOVA)")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "plots", "bayes_importance.png"), dpi=150); plt.close()


if __name__ == "__main__":
    main()
