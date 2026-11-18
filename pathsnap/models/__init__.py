"""One file per algorithm compared in the model study. Each defines NAME, FAMILY, CONFIG, build()."""
from . import gaussian_nb, knn, lightgbm_model, logistic_regression, mlp, random_forest

ALL = [logistic_regression, gaussian_nb, knn, random_forest, lightgbm_model, mlp]
