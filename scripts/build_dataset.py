"""Build candidate-level feature tables for the train and student_test partitions.

Output: results/features_{partition}.pkl (one row per observation x candidate arc)
        results/obs_{partition}.csv     (per-observation reference arc + reliability)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT, get_trip, trip_ids
from pathsnap.features import labelled_features

RELIABLE_M = 50.0   # obs farther than this from their aligned reference arc are not labelled
TRAIN_STRIDE = 3    # 1 Hz data is highly redundant; every 3rd obs is used for fitting
OUT = os.path.join(ROOT, "results")


def main():
    os.makedirs(OUT, exist_ok=True)
    for part in ["train", "student_test"]:
        feats, obs = [], []
        t0 = time.time()
        for t in trip_ids(part):
            trip = get_trip(part, t)
            df, ref, rd = labelled_features(trip)
            reliable = rd <= RELIABLE_M
            obs.append(pd.DataFrame({"trip": t, "obs": np.arange(len(ref)), "ref_arc": ref,
                                     "ref_dist": rd, "reliable": reliable}))
            df["reliable"] = reliable[df["obs"].values]
            if part == "train":
                df = df[df["reliable"] & (df["obs"] % TRAIN_STRIDE == 0)]
            feats.append(df)
        feats = pd.concat(feats, ignore_index=True)
        obs = pd.concat(obs, ignore_index=True)
        feats.to_pickle(os.path.join(OUT, f"features_{part}.pkl"))
        obs.to_csv(os.path.join(OUT, f"obs_{part}.csv"), index=False)
        cov = feats[feats.reliable].groupby(["trip", "obs"]).label.max()
        print(f"{part}: {len(feats)} candidate rows, {obs.reliable.sum()}/{len(obs)} reliable obs, "
              f"candidate miss rate {1 - cov.mean():.4f}, positive rate {feats.label.mean():.3f}, "
              f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
