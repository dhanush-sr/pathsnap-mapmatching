"""Record the project's measured results in MLflow and register the final model.

Reads the result files written by the experiment scripts (nothing is re-trained) and creates:
  pathsnap-model-comparison  one run per algorithm (params, CV + student-test metrics, plots, model file)
  pathsnap-bayes-opt         one parent run with the 40 Optuna trials as child runs, plus tuned vs untuned
  pathsnap-hmm               the decoder grid and the classifier-only vs HMM comparison
  pathsnap-final-model       the tuned LightGBM + HMM config, registered as 'pathsnap-lightgbm'

Each run is tagged source=recorded results so it is clear it was logged from saved outputs.
    python scripts/log_to_mlflow.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
"""
import json
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT
from pathsnap.features import FEATURES
from pathsnap.tracking import REGISTERED_MODEL, clean_params, mlflow_experiment

RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(RES, "plots")
TAG = {"source": "recorded results"}
num = lambda d: {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and pd.notna(v)}


def model_comparison():
    mlflow = mlflow_experiment("pathsnap-model-comparison")
    S = pd.read_csv(os.path.join(RES, "model_comparison.csv"))
    F = pd.read_csv(os.path.join(RES, "cv_folds.csv"))
    for _, r in S.iterrows():
        with mlflow.start_run(run_name=r.model):
            mlflow.set_tags({**TAG, "family": r.family})
            mlflow.log_params({"config": r.config, "cv": "5-fold GroupKFold by trip", "seed": 42})
            path = os.path.join(ROOT, "artifacts", "models", f"{r.model}.joblib")
            if os.path.exists(path):
                est = joblib.load(path)
                mlflow.log_params(clean_params(est.get_params(deep=True)))
                if os.path.getsize(path) < 50e6:
                    mlflow.log_artifact(path, "model")
            mlflow.log_metrics(num(r.to_dict()))
            for _, f in F[F.model == r.model].iterrows():
                mlflow.log_metrics({"fold_macro_f1": f.macro_f1, "fold_top1": f.top1_accuracy,
                                    "fold_log_loss": f.log_loss}, step=int(f.fold))
            for p in ("cv_macro_f1.png", "test_top1_f1.png", "log_loss.png", "confusion_matrices.png", "training_curves.png"):
                mlflow.log_artifact(os.path.join(PLOTS, p), "comparison_plots")
    print("model comparison:", len(S), "runs")


def bayes_opt():
    mlflow = mlflow_experiment("pathsnap-bayes-opt")
    best = json.load(open(os.path.join(RES, "bayes_best.json")))
    T = pd.read_csv(os.path.join(RES, "bayes_trials.csv"))
    with mlflow.start_run(run_name="optuna-tpe-lightgbm"):
        mlflow.set_tags(TAG)
        mlflow.log_params({"sampler": best["sampler"], "objective": best["objective"], "n_trials": best["n_trials"],
                           **{f"space_{k}": str(v) for k, v in best["search_space"].items()}})
        mlflow.log_metrics({"best_cv_macro_f1": best["best_cv_macro_f1"], "search_time_s": best["search_time_s"]})
        mlflow.log_params({f"best_{k}": v for k, v in best["best_params"].items()})
        for _, t in T.iterrows():
            with mlflow.start_run(run_name=f"trial-{int(t.number)}", nested=True):
                mlflow.set_tags(TAG)
                mlflow.log_params({c[len("params_"):]: t[c] for c in T.columns if c.startswith("params_")})
                mlflow.log_metrics(num({"cv_macro_f1": t.value, "cv_top1": t.get("user_attrs_cv_top1"),
                                        "cv_log_loss": t.get("user_attrs_cv_log_loss")}))
        for _, r in pd.read_csv(os.path.join(RES, "tuned_vs_untuned.csv")).iterrows():
            tag = "tuned" if "tuned" in r.model.lower() and "untuned" not in r.model.lower() else "untuned"
            mlflow.log_metrics({f"{tag}_{k}": v for k, v in num(r.to_dict()).items()})
        for p in ("bayes_history.png", "bayes_importance.png"):
            mlflow.log_artifact(os.path.join(PLOTS, p))
        mlflow.log_artifact(os.path.join(RES, "bayes_trials.csv"))
    print("bayes opt:", len(T), "trials")


def hmm():
    mlflow = mlflow_experiment("pathsnap-hmm")
    G = pd.read_csv(os.path.join(RES, "hmm_tuning.csv"))
    for _, g in G.iterrows():
        with mlflow.start_run(run_name=f"beta={g.beta} w_emit={g.w_emit}"):
            mlflow.set_tags({**TAG, "data": "out-of-fold probabilities, 80 training trips"})
            mlflow.log_params({"beta": g.beta, "w_emit": g.w_emit})
            mlflow.log_metrics({"train_oof_point_acc": g.train_oof_point_acc,
                                "train_oof_classifier_only": g.train_oof_classifier_only})
    C = pd.read_csv(os.path.join(RES, "hmm_vs_classifier.csv"), dtype={"trip": str})
    with mlflow.start_run(run_name="classifier-vs-hmm-student-test"):
        mlflow.set_tags(TAG)
        mlflow.log_params(json.load(open(os.path.join(ROOT, "artifacts", "hmm_config.json"))))
        mlflow.log_metrics({f"mean_{k}": v for k, v in num(C[C.trip == "MEAN"].iloc[0].to_dict()).items()})
        mlflow.log_artifact(os.path.join(RES, "hmm_vs_classifier.csv"))
        mlflow.log_artifact(os.path.join(PLOTS, "hmm_vs_classifier.png"))
    print("hmm:", len(G), "grid runs")


def final_model():
    mlflow = mlflow_experiment("pathsnap-final-model")
    model = joblib.load(os.path.join(ROOT, "artifacts", "final_model.joblib"))
    cfg = json.load(open(os.path.join(ROOT, "artifacts", "model_config.json")))
    tv = pd.read_csv(os.path.join(RES, "tuned_vs_untuned.csv")).iloc[1]
    with mlflow.start_run(run_name="lightgbm-tuned+hmm"):
        mlflow.set_tags(TAG)
        mlflow.log_params({**cfg["params"], "n_features": len(FEATURES)})
        mlflow.log_metrics(num(tv.to_dict()))
        mlflow.log_artifact(os.path.join(ROOT, "artifacts", "hmm_config.json"), "inference_config")
        mlflow.log_artifact(os.path.join(ROOT, "artifacts", "model_config.json"), "inference_config")
        info = mlflow.lightgbm.log_model(model, name="model", registered_model_name=REGISTERED_MODEL,
                                        input_example=pd.DataFrame([[0.0] * len(FEATURES)], columns=FEATURES))
    print("registered", REGISTERED_MODEL, "from", info.model_uri)


if __name__ == "__main__":
    model_comparison()
    bayes_opt()
    hmm()
    final_model()
