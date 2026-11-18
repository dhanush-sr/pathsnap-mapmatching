"""Random Forest: bagging ensemble of decision trees.

200 trees, each grown on a bootstrap sample with a random subset of features at every split
(Gini impurity). The predicted probability is the average vote of the trees.
Run alone:  python -m pathsnap.models.random_forest
"""
from sklearn.ensemble import RandomForestClassifier

NAME = "RandomForest"
FAMILY = "Ensemble (bagging)"
CONFIG = "200 trees, min_samples_leaf=5, sqrt features"
SEED = 42


def build():
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=SEED)


if __name__ == "__main__":
    from pathsnap.training import run
    run(__import__(__name__))
