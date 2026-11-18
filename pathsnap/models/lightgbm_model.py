"""LightGBM: gradient-boosted decision trees (the selected model).

Trees are added one at a time, each fitted to the gradient of the binary log loss of the
current ensemble, with leaf-wise growth. Untuned configuration; Bayesian tuning is in
scripts/bayes_opt.py.
Run alone:  python -m pathsnap.models.lightgbm_model
"""
import lightgbm as lgb

NAME = "LightGBM"
FAMILY = "Boosting"
CONFIG = "400 trees, lr=0.05, 31 leaves"
SEED = 42


def build():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1)


if __name__ == "__main__":
    from pathsnap.training import run
    run(__import__(__name__))
