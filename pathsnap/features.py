"""Per-observation reference labels, candidate generation and feature engineering."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .data import point_seg_dist

K_CAND = 10          # candidate arcs per GPS observation (reverse twins included)
DENSIFY_M = 10.0     # spacing of sample points used to index arcs
SEARCH_PTS = 80      # nearest densified points looked up per observation
CONTEXT = 3          # +/- observations used for context features

FEATURES = [
    "dist", "dist_rank", "dist_gap", "dist_ratio", "heading_diff", "speed",
    "heading_x_speed", "arc_len", "proj_frac", "end_clipped", "ctx_dist_mean",
    "ctx_heading_diff", "n_within_30", "from_out_deg", "to_in_deg", "gauss_share",
]


def align_route(trip):
    """Monotone DP alignment of each GPS observation to an index of the reference route.

    The .route file is an ordered arc list without timestamps, so the per-observation
    reference arc is the assignment minimising total point-to-arc distance subject to
    the route being traversed in order (non-decreasing route index).
    """
    r = trip.route
    cost, _ = point_seg_dist(trip.xy, trip.arc_a[r], trip.arc_b[r])
    D = cost[0].copy()
    back = np.zeros(cost.shape, dtype=np.int32)
    for i in range(1, len(cost)):
        cm = np.minimum.accumulate(D)
        arg = _argmin_accumulate(D)
        back[i] = arg
        D = cost[i] + cm
    j = int(np.argmin(D))
    idx = np.empty(len(cost), dtype=np.int32)
    for i in range(len(cost) - 1, -1, -1):
        idx[i] = j
        j = back[i, j]
    return r[idx], cost[np.arange(len(cost)), idx]


def _argmin_accumulate(D):
    cm = np.minimum.accumulate(D)
    return np.maximum.accumulate(np.where(D == cm, np.arange(len(D)), 0)).astype(np.int32)


def track_kinematics(trip):
    xy, t = trip.xy, trip.t
    n = len(xy)
    lo, hi = np.clip(np.arange(n) - 2, 0, n - 1), np.clip(np.arange(n) + 2, 0, n - 1)
    d = xy[hi] - xy[lo]
    dt = np.maximum(t[hi] - t[lo], 1e-3)
    speed = np.hypot(*d.T) / dt
    heading = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360
    return speed, heading


def angdiff(a, b):
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)


def candidates(trip):
    """Return (cand_arcs (N,K), dist (N,K), frac (N,K)) using a KD-tree over densified arcs."""
    la = trip.local_arcs
    a, b, L = trip.arc_a[la], trip.arc_b[la], trip.arc_len[la]
    reps = np.maximum(np.ceil(L / DENSIFY_M).astype(int), 1) + 1
    owner = np.repeat(np.arange(len(la)), reps)
    s = np.concatenate([np.linspace(0, 1, r) for r in reps])
    pts = a[owner] + s[:, None] * (b - a)[owner]
    tree = cKDTree(pts)
    _, nn = tree.query(trip.xy, k=min(SEARCH_PTS, len(pts)))
    N = len(trip.xy)
    C = np.full((N, K_CAND), -1, dtype=np.int64)
    Dm = np.full((N, K_CAND), np.inf)
    F = np.zeros((N, K_CAND))
    for i in range(N):
        u = np.unique(owner[nn[i]])
        d, f = point_seg_dist(trip.xy[i:i + 1], a[u], b[u])
        d, f = d[0], f[0]
        o = np.argsort(d, kind="stable")[:K_CAND]
        k = len(o)
        C[i, :k], Dm[i, :k], F[i, :k] = la[u[o]], d[o], f[o]
    return C, Dm, F


def build_features(trip, sigma=8.0):
    """Long-format candidate table: one row per (observation, candidate arc)."""
    C, Dm, F = candidates(trip)
    speed, heading = track_kinematics(trip)
    N, K = C.shape
    valid = C >= 0
    Cs = np.where(valid, C, 0)
    hd = angdiff(trip.arc_heading[Cs], heading[:, None])
    dmin = Dm[:, :1]
    # context: distance / heading agreement of the same arc with neighbouring observations
    ctx_d = np.zeros((N, K))
    ctx_h = np.zeros((N, K))
    cnt = 0
    for off in range(-CONTEXT, CONTEXT + 1):
        if off == 0:
            continue
        j = np.clip(np.arange(N) + off, 0, N - 1)
        dd, _ = _pair_dist(trip, trip.xy[j], Cs)
        ctx_d += dd
        ctx_h += angdiff(trip.arc_heading[Cs], heading[j][:, None])
        cnt += 1
    g = np.exp(-0.5 * (Dm / sigma) ** 2)
    g = g / np.maximum(g.sum(1, keepdims=True), 1e-12)
    frac_end = (F <= 1e-6) | (F >= 1 - 1e-6)
    df = pd.DataFrame({
        "obs": np.repeat(np.arange(N), K),
        "arc": C.ravel(),
        "dist": Dm.ravel(),
        "dist_rank": np.tile(np.arange(K), N),
        "dist_gap": (Dm - dmin).ravel(),
        "dist_ratio": (Dm / np.maximum(dmin, 0.5)).ravel(),
        "heading_diff": hd.ravel(),
        "speed": np.repeat(speed, K),
        "heading_x_speed": (hd * np.minimum(speed, 15)[:, None] / 15).ravel(),
        "arc_len": trip.arc_len[Cs].ravel(),
        "proj_frac": F.ravel(),
        "end_clipped": frac_end.ravel().astype(np.int8),
        "ctx_dist_mean": (ctx_d / cnt).ravel(),
        "ctx_heading_diff": (ctx_h / cnt).ravel(),
        "n_within_30": np.repeat((Dm <= 30).sum(1), K),
        "from_out_deg": trip.out_deg[trip.arcs[Cs, 0]].ravel(),
        "to_in_deg": trip.in_deg[trip.arcs[Cs, 1]].ravel(),
        "gauss_share": g.ravel(),
    })
    df = df[valid.ravel()].reset_index(drop=True)
    df["trip"] = trip.id
    return df


def _pair_dist(trip, p, Cs):
    a, b = trip.arc_a[Cs], trip.arc_b[Cs]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(-1), 1e-9)
    f = np.clip(((p[:, None, :] - a) * ab).sum(-1) / L2, 0, 1)
    proj = a + f[..., None] * ab
    return np.hypot(*(p[:, None, :] - proj).transpose(2, 0, 1)), f


def labelled_features(trip):
    df = build_features(trip)
    ref, ref_dist = align_route(trip)
    df["label"] = (df["arc"].values == ref[df["obs"].values]).astype(np.int8)
    return df, ref, ref_dist
