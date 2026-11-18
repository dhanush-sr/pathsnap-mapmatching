"""Gaussian Naive Bayes: probabilistic baseline.

Models each feature as an independent Gaussian per class and combines them with Bayes' rule.
Objective: maximum-likelihood estimates of per-class means and variances (no iterative training).
Run alone:  python -m pathsnap.models.gaussian_nb
"""
from sklearn.naive_bayes import GaussianNB

NAME = "GaussianNB"
FAMILY = "Probabilistic"
CONFIG = "var_smoothing=1e-9"


def build():
    return GaussianNB(var_smoothing=1e-9)


if __name__ == "__main__":
    from pathsnap.training import run
    run(__import__(__name__))
