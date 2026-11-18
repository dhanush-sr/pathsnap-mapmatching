"""PathSnap inference API - runs on LAPTOP 1 (backend).

Start:  python api/run_api.py            (or: uvicorn api.app:app --host 0.0.0.0 --port 8000)
Env:    PATHSNAP_EVAL_DIR   folder holding the ten evaluation trips
                            (default: ./evaluator_test if present, otherwise the student_test partition)
        PATHSNAP_MODEL_URI  load the model from the MLflow registry instead of artifacts/,
                            e.g. models:/pathsnap-lightgbm/1

The prediction pipeline loads each trip WITHOUT its .route file. The reference route is read only
inside score(), on this machine, to compute accuracy; it is never sent to the UI.
"""
import json
import logging
import os
import socket
import sys
import time

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT, Trip, trip_ids
from pathsnap.evaluate import argmax_decode, continuity
from pathsnap.features import FEATURES, align_route, build_features
from pathsnap.hmm import fill_route, route_scores, viterbi

_default_eval = os.path.join(ROOT, "evaluator_test")
EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", _default_eval if os.path.isdir(_default_eval) else "student_test")
RELIABLE_M = 50.0
MODEL_URI = os.environ.get("PATHSNAP_MODEL_URI")   # e.g. models:/pathsnap-lightgbm/1 (MLflow registry)


def load_model():
    if MODEL_URI:
        import mlflow
        from pathsnap.tracking import TRACKING_URI
        mlflow.set_tracking_uri(TRACKING_URI)
        return mlflow.lightgbm.load_model(MODEL_URI), MODEL_URI
    return joblib.load(os.path.join(ROOT, "artifacts", "final_model.joblib")), "artifacts/final_model.joblib"


MODEL, MODEL_SOURCE = load_model()
HMM_CFG = json.load(open(os.path.join(ROOT, "artifacts", "hmm_config.json")))

log = logging.getLogger("pathsnap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

app = FastAPI(title="PathSnap API (Laptop 1)", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


TRIPS = []


@app.on_event("startup")
def startup():
    for t in trip_ids(EVAL_DIR):
        tr = Trip(EVAL_DIR, t, with_route=False)
        TRIPS.append({"trip_id": t, "n_points": len(tr.t), "duration_s": float(tr.t[-1] - tr.t[0])})
    bar = "=" * 64
    log.info(bar)
    log.info("PATHSNAP API  |  LAPTOP 1 (backend: classifier + HMM/Viterbi)")
    log.info("model      : %s (%s)", type(MODEL).__name__, MODEL_SOURCE)
    log.info("hmm        : beta=%s  w_emit=%s", HMM_CFG["beta"], HMM_CFG["w_emit"])
    log.info("trips      : %d from %s", len(TRIPS), os.path.abspath(EVAL_DIR) if os.path.isdir(EVAL_DIR) else EVAL_DIR)
    log.info("UI address : enter  http://%s:%s  in the UI on Laptop 2", lan_ip(), os.environ.get("PATHSNAP_PORT", "8000"))
    log.info(bar)


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/")
def root():
    return {"service": "PathSnap API", "laptop": 1, "endpoints": ["/health", "/trips", "/predict", "/docs"]}


@app.get("/health")
def health():
    return {"status": "ok", "trips": len(TRIPS), "model": type(MODEL).__name__, "hmm": HMM_CFG}


@app.get("/trips")
def trips(request: Request):
    log.info("GET  /trips    from %s  -> %d trips", request.client.host, len(TRIPS))
    return TRIPS


def score(trip_id, hmm_dec, cls_dec, route):
    """Server-side evaluation against the reference route; only summary numbers leave this function."""
    ref_trip = Trip(EVAL_DIR, trip_id, with_route=True)
    if ref_trip.route is None:
        return {}
    ref, rd = align_route(ref_trip)
    rel = np.where(rd <= RELIABLE_M)[0]
    acc = lambda dec: float(np.mean([dec.get(int(i), -1) == ref[i] for i in rel])) if len(rel) else None
    return {"point_accuracy_hmm": acc(hmm_dec), "point_accuracy_classifier": acc(cls_dec),
            "evaluated_points": int(len(rel)), **route_scores(ref_trip, route, ref_trip.route)}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    if req.trip_id not in {t["trip_id"] for t in TRIPS}:
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    log.info("POST /predict  from %s  trip=%s", request.client.host, req.trip_id)
    t0 = time.time()
    trip = Trip(EVAL_DIR, req.trip_id, with_route=False)   # prediction never sees the route file
    df = build_features(trip)
    proba = MODEL.predict_proba(df[FEATURES])[:, 1]
    hmm_dec = viterbi(trip, df, proba, HMM_CFG)
    cls_dec = argmax_decode(df, proba)
    seq = [hmm_dec[k] for k in sorted(hmm_dec)]
    route = fill_route(trip, seq)
    elapsed = time.time() - t0
    metrics = {**score(req.trip_id, hmm_dec, cls_dec, route),
               "route_continuity_hmm": continuity(trip, seq),
               "route_continuity_classifier": continuity(trip, [cls_dec[k] for k in sorted(cls_dec)])}
    pmap = dict(zip(zip(df["obs"].values.tolist(), df["arc"].values.tolist()), proba.tolist()))
    ll = trip.nodes_ll
    path = [[float(ll[trip.arcs[route[0], 0], 1]), float(ll[trip.arcs[route[0], 0], 0])]] if route else []
    path += [[float(ll[trip.arcs[a, 1], 1]), float(ll[trip.arcs[a, 1], 0])] for a in route]
    points = [{"i": i, "t": float(trip.t[i]), "lat": float(trip.track[i, 1]), "lon": float(trip.track[i, 0]),
               "arc": hmm_dec.get(i), "p": round(pmap.get((i, hmm_dec.get(i)), 0.0), 4)}
              for i in range(len(trip.t))]
    log.info("     200 OK    trip=%s  points=%d  segments=%d  time=%.2fs  acc_hmm=%s  acc_classifier=%s  route_f1=%s",
             req.trip_id, len(points), len(route), elapsed, _pct(metrics.get("point_accuracy_hmm")),
             _pct(metrics.get("point_accuracy_classifier")), _f3(metrics.get("route_f1")))
    return {"trip_id": req.trip_id, "processing_s": round(elapsed, 3), "n_points": len(points),
            "matched_segments": len(route), "route_path": path, "points": points, "metrics": metrics}


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _f3(x):
    return "n/a" if x is None else f"{x:.3f}"
