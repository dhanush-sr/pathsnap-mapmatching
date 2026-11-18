"""Multi-layer perceptron: shallow neural network.

Two hidden ReLU layers (64 and 32 units) on standardised features, trained with Adam on the
binary cross-entropy (log loss) plus L2 penalty alpha. 10 % of the data is held out for early stopping.
Run alone:  python -m pathsnap.models.mlp
"""
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NAME = "MLP"
FAMILY = "Neural network"
CONFIG = "64-32 ReLU, adam, alpha=1e-4, early stopping"
SEED = 42


def build():
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                         early_stopping=True, max_iter=60, random_state=SEED))


if __name__ == "__main__":
    from pathsnap.training import run
    run(__import__(__name__))
