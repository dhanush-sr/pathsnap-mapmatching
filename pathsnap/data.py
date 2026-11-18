"""Raw-data loading, projection to metres and road-network preparation."""
import os
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("PATHSNAP_DATA", os.path.join(ROOT, "Student folder"))
EARTH_R = 6371008.8
CROP_MARGIN_M = 400.0


def partition_dir(partition):
    """A partition name inside DATA_DIR (train, student_test) or a path to any folder of trip folders."""
    return partition if os.path.isdir(partition) else os.path.join(DATA_DIR, partition)


def trip_ids(partition):
    d = partition_dir(partition)
    return sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))


def _read(path, dtype=float):
    a = np.loadtxt(path, dtype=dtype, ndmin=2)
    return a


def load_raw(partition, trip, with_route=True):
    base = os.path.join(partition_dir(partition), trip, trip)
    raw = {
        "track": _read(base + ".track"),
        "nodes": _read(base + ".nodes"),
        "arcs": _read(base + ".arcs", int),
    }
    if with_route and os.path.exists(base + ".route"):
        raw["route"] = np.loadtxt(base + ".route", dtype=int, ndmin=1)
    return raw


def clean_track(track):
    """Drop non-finite / implausible rows and duplicate or non-increasing timestamps."""
    ok = np.isfinite(track).all(1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    track = track[ok]
    track = track[np.argsort(track[:, 2], kind="stable")]
    keep = np.r_[True, np.diff(track[:, 2]) > 0]
    return track[keep]


class Trip:
    """One trip: cleaned GPS track and its road network in a local metric frame.

    Local equirectangular projection around the track centroid. Trips span a few km,
    so the distortion is far below GPS noise (<0.1 %).
    """

    def __init__(self, partition, trip, with_route=True):
        raw = load_raw(partition, trip, with_route)
        self.partition, self.id = partition, trip
        self.track = clean_track(raw["track"])
        self.lon0, self.lat0 = self.track[:, 0].mean(), self.track[:, 1].mean()
        self.coslat = np.cos(np.radians(self.lat0))
        self.xy = self.project(self.track[:, 0], self.track[:, 1])
        self.t = self.track[:, 2]
        self.nodes_ll = raw["nodes"]
        self.nodes_xy = self.project(self.nodes_ll[:, 0], self.nodes_ll[:, 1])
        self.arcs = raw["arcs"]
        self.route = raw.get("route")
        a, b = self.nodes_xy[self.arcs[:, 0]], self.nodes_xy[self.arcs[:, 1]]
        self.arc_a, self.arc_b = a, b
        self.arc_len = np.maximum(np.hypot(*(b - a).T), 1e-3)
        self.arc_heading = np.degrees(np.arctan2(*(b - a).T[::-1])) % 360  # atan2(dy, dx)
        n = len(self.nodes_xy)
        self.out_deg = np.bincount(self.arcs[:, 0], minlength=n)
        self.in_deg = np.bincount(self.arcs[:, 1], minlength=n)
        # crop network to the track's neighbourhood for search and routing
        lo, hi = self.xy.min(0) - CROP_MARGIN_M, self.xy.max(0) + CROP_MARGIN_M
        inside = lambda p: ((p >= lo) & (p <= hi)).all(1)
        self.local_arcs = np.where(inside(a) | inside(b))[0]

    def project(self, lon, lat):
        x = np.radians(lon - self.lon0) * EARTH_R * self.coslat
        y = np.radians(lat - self.lat0) * EARTH_R
        return np.c_[x, y]

    @property
    def graph(self):
        if not hasattr(self, "_graph"):
            la = self.local_arcs
            n = len(self.nodes_xy)
            self._graph = csr_matrix(
                (self.arc_len[la], (self.arcs[la, 0], self.arcs[la, 1])), shape=(n, n)
            )
        return self._graph


@lru_cache(maxsize=128)
def get_trip(partition, trip, with_route=True):
    return Trip(partition, trip, with_route)


def point_seg_dist(p, a, b):
    """Distances from points p (N,2) to segments a->b (M,2) -> (N,M) dist and fraction."""
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(-1), 1e-9)
    ap = p[:, None, :] - a[None]
    f = np.clip((ap * ab[None]).sum(-1) / L2[None], 0, 1)
    proj = a[None] + f[..., None] * ab[None]
    return np.hypot(*(p[:, None, :] - proj).transpose(2, 0, 1)), f
