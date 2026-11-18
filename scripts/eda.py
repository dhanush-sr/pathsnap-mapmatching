"""Exploratory data analysis of the train and student_test partitions.

Outputs (results/eda/): inventory.csv, summary.csv, plots and a Folium map.
"""
import os
import sys

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import DATA_DIR, ROOT, Trip, load_raw, point_seg_dist, trip_ids
from pathsnap.features import align_route

OUT = os.path.join(ROOT, "results", "eda")


def inspect(part, t):
    raw = load_raw(part, t)
    tr = raw["track"]
    nodes, arcs, route = raw["nodes"], raw["arcs"], raw.get("route")
    sizes = {ext: os.path.getsize(os.path.join(DATA_DIR, part, t, f"{t}.{ext}")) for ext in ("track", "nodes", "arcs", "route")}
    dt = np.diff(tr[:, 2])
    trip = Trip(part, t)
    n = len(nodes)
    G = csr_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
    ncomp, lab = connected_components(G, directed=True, connection="weak")
    deg = np.bincount(arcs.ravel(), minlength=n)
    ref, rd = align_route(trip)
    ext = trip.xy.max(0) - trip.xy.min(0)
    step = np.hypot(*np.diff(trip.xy, axis=0).T)
    return {
        "partition": part, "trip": t, "n_obs": len(tr), "n_nodes": n, "n_arcs": len(arcs),
        "route_len": len(route), **{f"bytes_{k}": v for k, v in sizes.items()},
        "nan_values": int(np.isnan(tr).sum()), "invalid_coords": int(((np.abs(tr[:, 0]) > 180) | (np.abs(tr[:, 1]) > 90)).sum()),
        "non_increasing_ts": int((dt <= 0).sum()), "duration_s": float(tr[-1, 2] - tr[0, 2]),
        "dt_eq_1_share": float(np.mean(np.isclose(dt, 1.0))), "dt_max": float(dt.max()), "gaps_gt_5s": int((dt > 5).sum()),
        "max_step_m": float(step.max()), "extent_x_km": ext[0] / 1000, "extent_y_km": ext[1] / 1000,
        "lon_mean": float(tr[:, 0].mean()), "lat_mean": float(tr[:, 1].mean()),
        "isolated_nodes": int((deg == 0).sum()), "weak_components": int(ncomp),
        "largest_component_share": float(np.bincount(lab).max() / n),
        "route_contiguous_share": float(np.mean(arcs[route[:-1], 1] == arcs[route[1:], 0])),
        "route_repeated_arcs": int(len(route) - len(set(route))),
        "ref_dist_median_m": float(np.median(rd)), "ref_dist_p95_m": float(np.percentile(rd, 95)),
        "obs_unreliable_share": float(np.mean(rd > 50)), "_dt": dt, "_rd": rd,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = [inspect(p, t) for p in ("train", "student_test") for t in trip_ids(p)]
    inv = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    inv.to_csv(os.path.join(OUT, "inventory.csv"), index=False)
    summ = inv.groupby("partition").agg(
        trips=("trip", "count"), total_obs=("n_obs", "sum"), median_obs=("n_obs", "median"),
        max_obs=("n_obs", "max"), median_nodes=("n_nodes", "median"), median_arcs=("n_arcs", "median"),
        nan_values=("nan_values", "sum"), invalid_coords=("invalid_coords", "sum"),
        non_increasing_ts=("non_increasing_ts", "sum"), trips_with_gaps_gt_5s=("gaps_gt_5s", lambda s: int((s > 0).sum())),
        mean_dt1_share=("dt_eq_1_share", "mean"), isolated_nodes=("isolated_nodes", "sum"),
        median_components=("weak_components", "median"), mean_route_contiguity=("route_contiguous_share", "mean"),
        trips_unreliable_gt_10pct=("obs_unreliable_share", lambda s: int((s > 0.1).sum())))
    summ.to_csv(os.path.join(OUT, "summary.csv"))
    print(summ.T.to_string())
    print(inv.sort_values("obs_unreliable_share", ascending=False).head(6)[["partition", "trip", "n_obs", "obs_unreliable_share", "dt_max"]])

    # missing / invalid values per column
    fig, ax = plt.subplots(figsize=(7, 3.5))
    cols = ["nan_values", "invalid_coords", "non_increasing_ts", "gaps_gt_5s", "isolated_nodes"]
    vals = inv.groupby("partition")[cols].sum().T
    vals.plot.bar(ax=ax); ax.set_title("Missing / invalid value counts (all trips)"); ax.set_ylabel("count")
    plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(f"{OUT}/missing_values.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for p, g in inv.groupby("partition"):
        axes[0].hist(g.n_obs, bins=30, alpha=.6, label=p)
    axes[0].set_xscale("log"); axes[0].set_title("GPS observations per trip"); axes[0].legend()
    dts = np.concatenate([r["_dt"] for r in rows])
    axes[1].hist(np.clip(dts, 0, 10), bins=40); axes[1].set_yscale("log"); axes[1].set_title("Sampling interval (s, clipped at 10)")
    rds = np.concatenate([r["_rd"] for r in rows if r["partition"] == "train"])
    axes[2].hist(np.clip(rds, 0, 200), bins=50); axes[2].set_yscale("log"); axes[2].axvline(50, color="r", ls="--")
    axes[2].set_title("GPS-to-aligned-reference distance (m, train)")
    plt.tight_layout(); plt.savefig(f"{OUT}/distributions.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for p, g in inv.groupby("partition"):
        ax.scatter(g.lon_mean, g.lat_mean, s=18, label=p)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title("Trip locations (mean GPS coordinate)"); ax.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/trip_locations.png", dpi=150); plt.close()

    # trajectory plots in local metres, with road network and reference route
    sample = [("train", t) for t in trip_ids("train")[:4]] + [("student_test", t) for t in trip_ids("student_test")[:2]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (p, t) in zip(axes.ravel(), sample):
        tr = Trip(p, t)
        la = tr.local_arcs
        ax.add_collection(LineCollection(np.stack([tr.arc_a[la], tr.arc_b[la]], 1), colors="#d0d0d0", linewidths=.5))
        r = tr.route
        ax.add_collection(LineCollection(np.stack([tr.arc_a[r], tr.arc_b[r]], 1), colors="green", linewidths=2))
        ax.plot(*tr.xy.T, ".", ms=1.5, color="blue")
        lo, hi = tr.xy.min(0) - 100, tr.xy.max(0) + 100
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_title(f"{p} {t} ({len(tr.xy)} obs)")
    fig.suptitle("GPS observations (blue), reference route (green), road network (grey) - local metres")
    plt.tight_layout(); plt.savefig(f"{OUT}/trajectories.png", dpi=130); plt.close()

    # road network size / connectivity
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(inv.n_nodes, inv.n_arcs, c=(inv.partition == "train"), cmap="coolwarm", s=15)
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("nodes"); axes[0].set_ylabel("arcs")
    axes[0].set_title("Road network size per trip")
    axes[1].hist(inv.largest_component_share, bins=20); axes[1].set_title("Share of nodes in largest weak component")
    plt.tight_layout(); plt.savefig(f"{OUT}/road_network.png", dpi=150); plt.close()

    folium_map("train", trip_ids("train")[0], os.path.join(OUT, "eda_map_train.html"))


def folium_map(part, t, path):
    tr = Trip(part, t)
    ll = tr.nodes_ll
    m = folium.Map(location=[tr.lat0, tr.lon0], zoom_start=14)
    net = folium.FeatureGroup("Road network (near track)")
    near = tr.local_arcs
    d, _ = point_seg_dist(tr.xy[::10], tr.arc_a[near], tr.arc_b[near])
    near = near[d.min(0) < 150]
    for a in near:
        u, v = tr.arcs[a]
        folium.PolyLine([[ll[u, 1], ll[u, 0]], [ll[v, 1], ll[v, 0]]], color="#999", weight=1).add_to(net)
    net.add_to(m)
    ref = folium.FeatureGroup("Reference route")
    pts = [[ll[tr.arcs[tr.route[0], 0], 1], ll[tr.arcs[tr.route[0], 0], 0]]] + [[ll[tr.arcs[a, 1], 1], ll[tr.arcs[a, 1], 0]] for a in tr.route]
    folium.PolyLine(pts, color="green", weight=4).add_to(ref)
    ref.add_to(m)
    gps = folium.FeatureGroup("Raw GPS")
    for lon, lat, _ in tr.track[::2]:
        folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True).add_to(gps)
    gps.add_to(m)
    folium.LayerControl().add_to(m)
    m.fit_bounds([[tr.track[:, 1].min(), tr.track[:, 0].min()], [tr.track[:, 1].max(), tr.track[:, 0].max()]])
    m.save(path)


if __name__ == "__main__":
    main()
