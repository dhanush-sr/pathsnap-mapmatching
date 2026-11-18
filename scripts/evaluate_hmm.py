"""Tune the HMM decoder and compare classifier-only vs classifier + HMM/Viterbi.

Decoder parameters are tuned on OUT-OF-FOLD classifier probabilities of the 80 training
trips (5-fold GroupKFold), so the student-test comparison stays an honest hold-out.
Outputs: artifacts/hmm_config.json, results/hmm_tuning.csv, results/hmm_vs_classifier.csv,
         results/plots/hmm_vs_classifier.png
"""
import itertools
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
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT, get_trip, trip_ids
from pathsnap.features import FEATURES, labelled_features
from pathsnap.evaluate import argmax_decode, continuity
from pathsnap.hmm import DEFAULTS, fill_route, route_scores, viterbi

RES = os.path.join(ROOT, "results")
RELIABLE_M = 50.0
GRID = {"beta": [2.0, 4.0, 8.0, 16.0, 32.0], "w_emit": [0.5, 1.0, 2.0]}


def full_tables(part):
    out = {}
    for t in trip_ids(part):
        df, ref, rd = labelled_features(get_trip(part, t))
        out[t] = (df, ref, rd <= RELIABLE_M)
    return out


def point_acc(dec, ref, rel):
    idx = np.where(rel)[0]
    return float(np.mean([dec.get(int(i), -1) == ref[i] for i in idx])) if len(idx) else np.nan


def main():
    cfg = json.load(open(os.path.join(ROOT, "artifacts", "model_config.json")))
    params = cfg["params"]
    train_fit = pd.read_pickle(os.path.join(RES, "features_train.pkl"))
    full = full_tables("train")
    tids = np.array(sorted(full))

    # out-of-fold probabilities for every observation of every training trip
    oof = {}
    for a, b in GroupKFold(n_splits=5).split(tids, groups=tids):
        fit = train_fit[train_fit["trip"].isin(tids[a])]
        m = lgb.LGBMClassifier(**params, subsample_freq=1 if params.get("subsample", 1) < 1 else 0,
                               random_state=42, verbose=-1).fit(fit[FEATURES], fit["label"])
        for t in tids[b]:
            oof[t] = m.predict_proba(full[t][0][FEATURES])[:, 1]

    rows = []
    for beta, w in itertools.product(GRID["beta"], GRID["w_emit"]):
        p = {"beta": beta, "w_emit": w}
        accs = [point_acc(viterbi(get_trip("train", t), full[t][0], oof[t], p), full[t][1], full[t][2]) for t in tids]
        rows.append({**p, "train_oof_point_acc": np.nanmean(accs)})
        print(rows[-1])
    tuning = pd.DataFrame(rows)
    base = np.nanmean([point_acc(argmax_decode(full[t][0], oof[t]), full[t][1], full[t][2]) for t in tids])
    tuning["train_oof_classifier_only"] = base
    tuning.to_csv(os.path.join(RES, "hmm_tuning.csv"), index=False)
    best = tuning.loc[tuning.train_oof_point_acc.idxmax()]
    hmm_cfg = {**DEFAULTS, "beta": float(best.beta), "w_emit": float(best.w_emit)}
    json.dump(hmm_cfg, open(os.path.join(ROOT, "artifacts", "hmm_config.json"), "w"), indent=2)
    print("best decoder", hmm_cfg)

    # student-test comparison with the final model
    model = joblib.load(os.path.join(ROOT, "artifacts", "final_model.joblib"))
    comp = []
    for t, (df, ref, rel) in full_tables("student_test").items():
        trip = get_trip("student_test", t)
        p = model.predict_proba(df[FEATURES])[:, 1]
        cls = argmax_decode(df, p)
        t0 = time.time(); hmm = viterbi(trip, df, p, hmm_cfg); dec_t = time.time() - t0
        cs, hs = [cls[k] for k in sorted(cls)], [hmm[k] for k in sorted(hmm)]
        rc, rh = route_scores(trip, fill_route(trip, cs), trip.route), route_scores(trip, fill_route(trip, hs), trip.route)
        comp.append({"trip": t, "n_obs": len(ref),
                     "acc_classifier": point_acc(cls, ref, rel), "acc_hmm": point_acc(hmm, ref, rel),
                     "continuity_classifier": continuity(trip, cs), "continuity_hmm": continuity(trip, hs),
                     "route_f1_classifier": rc["route_f1"], "route_f1_hmm": rh["route_f1"],
                     "route_len_f1_classifier": rc["route_len_f1"], "route_len_f1_hmm": rh["route_len_f1"],
                     "decode_time_s": dec_t})
    C = pd.DataFrame(comp)
    mean = C.drop(columns="trip").mean(numeric_only=True)
    C = pd.concat([C, pd.DataFrame([{"trip": "MEAN", **mean}])], ignore_index=True)
    C.to_csv(os.path.join(RES, "hmm_vs_classifier.csv"), index=False)
    print(C.round(4).to_string())

    c = C[C.trip != "MEAN"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(c)); w = .38
    for ax, (k, title) in zip(axes, [("acc", "Point accuracy"), ("continuity", "Route continuity"), ("route_f1", "Route F1 (arc set)")]):
        ax.bar(x - w / 2, c[f"{k}_classifier"], w, label="classifier only")
        ax.bar(x + w / 2, c[f"{k}_hmm"], w, label="classifier + HMM")
        ax.set_xticks(x, c.trip, rotation=60, fontsize=8); ax.set_title(title); ax.set_ylim(0.5, 1.01)
    axes[0].legend(loc="lower left")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "plots", "hmm_vs_classifier.png"), dpi=150); plt.close()


if __name__ == "__main__":
    main()
