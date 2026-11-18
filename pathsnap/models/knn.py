"""K-Nearest Neighbours: instance-based.

Scores a candidate by the distance-weighted vote of the 25 most similar training candidates in
standardised feature space. No loss is optimised; the training set is stored and searched.
Run alone:  python -m pathsnap.models.knn
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NAME = "KNN"
FAMILY = "Instance-based"
CONFIG = "k=25, distance-weighted, standardised"


def build():
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1))


if __name__ == "__main__":
    from pathsnap.training import run
    run(__import__(__name__))
