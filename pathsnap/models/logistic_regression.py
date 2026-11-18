"""Logistic Regression: linear baseline.

Learns one weight per standardised feature and passes the weighted sum through a sigmoid,
giving P(candidate is the true arc). Loss: L2-regularised binary log loss, solved with L-BFGS.
Run alone:  python -m pathsnap.models.logistic_regression
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NAME = "LogisticRegression"
FAMILY = "Linear"
CONFIG = "C=1.0, L2, lbfgs, standardised"


def build():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


if __name__ == "__main__":
    from pathsnap.training import run
    run(__import__(__name__))
