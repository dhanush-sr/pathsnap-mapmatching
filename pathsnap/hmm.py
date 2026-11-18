"""HMM map matching with Viterbi decoding over classifier-scored candidate arcs.

Hidden state at observation t: one candidate arc of t (plus the projection point on it).
Emission:   w_emit * log P_classifier(arc is the true arc | features).
Transition: -|network_distance - gps_distance| / (beta * gap_scale), the route/straight-line
            agreement model of Newson & Krumm, computed with Dijkstra on the trip's road graph.
Unreachable pairs get a finite penalty (UNREACHABLE) instead of -inf, so the decoder never
dead-ends on disconnected candidates; it just strongly prefers connected ones.
"""
import numpy as np
from scipy.sparse.csgraph import dijkstra

from .evaluate import candidate_lists, collapse

DEFAULTS = {"beta": 6.0, "w_emit": 1.0, "unreachable": -40.0, "min_limit": 300.0,
            "limit_factor": 4.0, "gap_s": 5.0, "eps": 1e-4}


def viterbi(trip, df, proba, params=None):
    """Return dict obs_index -> decoded arc id (one per observation that has candidates)."""
    P = {**DEFAULTS, **(params or {})}
    ids, A, Pr, Fr = candidate_lists(df, proba)
    if len(ids) == 0:
        return {}
    xy, t = trip.xy, trip.t
    L, arcs = trip.arc_len, trip.arcs
    G = trip.graph
    score = P["w_emit"] * np.log(np.clip(Pr[0], P["eps"], 1))
    back = []
    for k in range(1, len(ids)):
        i0, i1 = ids[k - 1], ids[k]
        a0, a1, f0, f1 = A[k - 1], A[k], Fr[k - 1], Fr[k]
        gc = float(np.hypot(*(xy[i1] - xy[i0])))
        dt = max(float(t[i1] - t[i0]), 1.0)
        limit = max(P["min_limit"], P["limit_factor"] * gc)
        src = np.unique(arcs[a0, 1])
        sp = dijkstra(G, indices=src, limit=limit)            # (len(src), n_nodes)
        row = np.searchsorted(src, arcs[a0, 1])
        d = (1 - f0)[:, None] * L[a0][:, None] + sp[row][:, arcs[a1, 0]] + (f1 * L[a1])[None]
        same = a0[:, None] == a1[None]
        d = np.where(same, np.abs(f1[None] - f0[:, None]) * L[a0][:, None], d)
        scale = P["beta"] * (dt / P["gap_s"] if dt > P["gap_s"] else 1.0)
        logT = np.where(np.isfinite(d), -np.abs(d - gc) / scale, P["unreachable"])
        tot = score[:, None] + logT
        b = tot.argmax(0)
        back.append(b)
        score = tot[b, np.arange(len(a1))] + P["w_emit"] * np.log(np.clip(Pr[k], P["eps"], 1))
    j = int(score.argmax())
    out = np.empty(len(ids), dtype=np.int64)
    for k in range(len(ids) - 1, -1, -1):
        out[k] = A[k][j]
        if k:
            j = back[k - 1][j]
    return dict(zip(ids.tolist(), out.tolist()))


def fill_route(trip, arc_seq, max_gap_m=2000.0):
    """Connect consecutive decoded arcs with shortest paths -> continuous arc route."""
    s = collapse(arc_seq)
    if not s:
        return []
    lookup = {}
    la = trip.local_arcs
    for aid, (u, v) in zip(la, trip.arcs[la]):
        if (u, v) not in lookup or trip.arc_len[aid] < trip.arc_len[lookup[(u, v)]]:
            lookup[(u, v)] = aid
    route = [s[0]]
    for a, b in zip(s[:-1], s[1:]):
        u, v = trip.arcs[a, 1], trip.arcs[b, 0]
        if u != v:
            dist, pred = dijkstra(trip.graph, indices=u, limit=max_gap_m, return_predecessors=True)
            if np.isfinite(dist[v]):
                path, n = [], v
                while n != u:
                    path.append(lookup[(pred[n], n)])
                    n = pred[n]
                route.extend(path[::-1])
        route.append(b)
    return collapse(route)


def route_scores(trip, pred_route, ref_route):
    """Arc-set and length-weighted precision / recall / F1 of a route against the reference."""
    P, R = set(pred_route), set(ref_route.tolist() if hasattr(ref_route, "tolist") else ref_route)
    inter = P & R
    L = trip.arc_len
    lp, lr, li = (float(L[list(x)].sum()) if x else 0.0 for x in (P, R, inter))
    prec, rec = (len(inter) / len(P) if P else 0.0), (len(inter) / len(R) if R else 0.0)
    lprec, lrec = (li / lp if lp else 0.0), (li / lr if lr else 0.0)
    f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1(prec, rec),
            "route_len_precision": lprec, "route_len_recall": lrec, "route_len_f1": f1(lprec, lrec)}
