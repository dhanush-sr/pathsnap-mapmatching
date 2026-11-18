"""Compare four decoders on the student-test trips (point accuracy on reliable observations).

  nearest_road       closest candidate arc, no model, no sequence
  hmm_distance_only  HMM with Gaussian distance emissions (sigma 8 m), same transitions as ours;
                     emission weight chosen per trip from {0.5, 1, 2}, which favours this baseline
  classifier_only    top LightGBM candidate per observation
  classifier_hmm     LightGBM emissions + HMM (the final system, artifacts/hmm_config.json)

Output: results/hmm_baseline.csv
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT, get_trip, trip_ids
from pathsnap.evaluate import argmax_decode
from pathsnap.features import FEATURES, labelled_features
from pathsnap.hmm import viterbi

RELIABLE_M = 50.0


def main():
    cfg = json.load(open(os.path.join(ROOT, "artifacts", "hmm_config.json")))
    model = joblib.load(os.path.join(ROOT, "artifacts", "final_model.joblib"))
    rows = []
    for t in trip_ids("student_test"):
        trip = get_trip("student_test", t)
        df, ref, rd = labelled_features(trip)
        rel = np.where(rd <= RELIABLE_M)[0]
        acc = lambda dec: float(np.mean([dec.get(int(i), -1) == ref[i] for i in rel]))
        g = df["gauss_share"].values
        p = model.predict_proba(df[FEATURES])[:, 1]
        dist_hmm = max((acc(viterbi(trip, df, g, {**cfg, "w_emit": w})), w) for w in (0.5, 1.0, 2.0))
        rows.append({"trip": t, "nearest_road": acc(argmax_decode(df, g)),
                     "hmm_distance_only": dist_hmm[0], "hmm_distance_only_w_emit": dist_hmm[1],
                     "classifier_only": acc(argmax_decode(df, p)),
                     "classifier_hmm": acc(viterbi(trip, df, p, cfg))})
        print(rows[-1])
    R = pd.DataFrame(rows)
    mean = R.drop(columns=["trip", "hmm_distance_only_w_emit"]).mean()
    R = pd.concat([R, pd.DataFrame([{"trip": "MEAN", **mean}])], ignore_index=True)
    R.to_csv(os.path.join(ROOT, "results", "hmm_baseline.csv"), index=False)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
