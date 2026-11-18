"""Feature selection analysis on the training trips only (no student-test data is used).

1. Pearson correlation between features (redundancy check)
2. LightGBM split-gain importance
3. Permutation importance (log loss) on held-out training trips (GroupShuffleSplit)
4. Group ablation: 3-fold GroupKFold macro F1 / Top-1 with each feature group removed

Outputs: results/feature_selection/*.csv, *.png
"""
import os
import sys

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT
from pathsnap.evaluate import candidate_metrics
from pathsnap.features import FEATURES

SEED = 42
OUT = os.path.join(ROOT, "results", "feature_selection")
GROUPS = {
    "distance": ["dist", "dist_rank", "dist_gap", "dist_ratio", "gauss_share", "n_within_30"],
    "heading": ["heading_diff", "heading_x_speed", "speed"],
    "context": ["ctx_dist_mean", "ctx_heading_diff"],
    "geometry/topology": ["arc_len", "proj_frac", "end_clipped", "from_out_deg", "to_in_deg"],
}


def model():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1)


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = pd.read_pickle(os.path.join(ROOT, "results", "features_train.pkl"))
    assert set(sum(GROUPS.values(), [])) == set(FEATURES)

    corr = tr[FEATURES].corr()
    corr.to_csv(f"{OUT}/correlation.csv")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=70, fontsize=8)
    ax.set_yticks(range(len(FEATURES)), FEATURES, fontsize=8)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im); ax.set_title("Feature correlation (train)")
    plt.tight_layout(); plt.savefig(f"{OUT}/correlation.png", dpi=150); plt.close()

    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(tr, groups=tr["trip"]))
    m = model().fit(tr.iloc[a][FEATURES], tr.iloc[a]["label"])
    gain = pd.Series(m.booster_.feature_importance("gain"), index=FEATURES)
    val = tr.iloc[b].sample(min(60000, len(b)), random_state=SEED)
    perm = permutation_importance(m, val[FEATURES], val["label"], scoring="neg_log_loss",
                                  n_repeats=3, random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({"gain_share": gain / gain.sum(), "perm_logloss_increase": perm.importances_mean,
                        "perm_std": perm.importances_std}).sort_values("perm_logloss_increase", ascending=False)
    imp["group"] = [next(g for g, f in GROUPS.items() if i in f) for i in imp.index]
    imp.to_csv(f"{OUT}/importance.csv")
    print(imp.round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    s = imp.sort_values("gain_share")
    axes[0].barh(s.index, s.gain_share); axes[0].set_title("LightGBM split-gain share")
    s = imp.sort_values("perm_logloss_increase")
    axes[1].barh(s.index, s.perm_logloss_increase, xerr=s.perm_std); axes[1].set_title("Permutation importance (log-loss increase, held-out trips)")
    plt.tight_layout(); plt.savefig(f"{OUT}/importance.png", dpi=150); plt.close()

    rows = []
    for name, drop in [("all features", [])] + [(f"without {g}", f) for g, f in GROUPS.items()]:
        feats = [f for f in FEATURES if f not in drop]
        res = []
        for fa, fb in GroupKFold(n_splits=3).split(tr, groups=tr["trip"]):
            mm = model().fit(tr.iloc[fa][feats], tr.iloc[fa]["label"])
            res.append(candidate_metrics(tr.iloc[fb], mm.predict_proba(tr.iloc[fb][feats])[:, 1]))
        r = pd.DataFrame(res).mean()
        rows.append({"config": name, "n_features": len(feats), "cv_macro_f1": r.macro_f1,
                     "cv_top1": r.top1_accuracy, "cv_log_loss": r.log_loss})
        print(rows[-1])
    abl = pd.DataFrame(rows)
    abl.to_csv(f"{OUT}/ablation.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(abl.config[::-1], abl.cv_top1[::-1], color=["#C44E52"] * (len(abl) - 1) + ["#4C72B0"])
    ax.set_xlim(abl.cv_top1.min() - 0.03, abl.cv_top1.max() + 0.01)
    for i, v in enumerate(abl.cv_top1[::-1]):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("3-fold GroupKFold Top-1 accuracy"); ax.set_title("Feature-group ablation (train)")
    plt.tight_layout(); plt.savefig(f"{OUT}/ablation.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
