"""Folium maps for the student-test trips: GPS, road network, reference, classifier-only and HMM routes.

Output: results/maps/trip_<id>.html
"""
import json
import os
import sys

import folium
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT, get_trip, point_seg_dist, trip_ids
from pathsnap.features import FEATURES, build_features
from pathsnap.evaluate import argmax_decode
from pathsnap.hmm import fill_route, viterbi

OUT = os.path.join(ROOT, "results", "maps")
LEGEND = """<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:#fff;padding:10px 14px;
border:1px solid #ccc;border-radius:6px;font:13px sans-serif;line-height:1.7">
<b>Trip {tid}</b><br>
<span style="color:#1f5fd6">&#9679;</span> Raw GPS observations<br>
<span style="color:#999">&#9472;&#9472;</span> Road network (within 150 m)<br>
<span style="color:#2ca02c">&#9473;&#9473;</span> Reference route<br>
<span style="color:#ff7f0e">&#9473;&#9473;</span> Classifier-only route<br>
<span style="color:#b0179a">&#9473;&#9473;</span> HMM / Viterbi route</div>"""


def arc_path(trip, arcs):
    ll = trip.nodes_ll
    if not len(arcs):
        return []
    return [[ll[trip.arcs[arcs[0], 0], 1], ll[trip.arcs[arcs[0], 0], 0]]] + \
           [[ll[trip.arcs[a, 1], 1], ll[trip.arcs[a, 1], 0]] for a in arcs]


def make_map(trip, model, hmm_cfg):
    df = build_features(trip)
    p = model.predict_proba(df[FEATURES])[:, 1]
    cls, hmm = argmax_decode(df, p), viterbi(trip, df, p, hmm_cfg)
    ll = trip.nodes_ll
    m = folium.Map(location=[trip.lat0, trip.lon0], zoom_start=14, control_scale=True)

    net = folium.FeatureGroup("Road network (within 150 m)", show=True)
    la = trip.local_arcs
    d, _ = point_seg_dist(trip.xy[::5], trip.arc_a[la], trip.arc_b[la])
    for a in la[d.min(0) < 150]:
        u, v = trip.arcs[a]
        folium.PolyLine([[ll[u, 1], ll[u, 0]], [ll[v, 1], ll[v, 0]]], color="#999", weight=1, opacity=.7).add_to(net)
    net.add_to(m)
    for name, arcs, color, w in [
        ("Reference route", list(trip.route), "#2ca02c", 7),
        ("Classifier-only route", fill_route(trip, [cls[k] for k in sorted(cls)]), "#ff7f0e", 4),
        ("HMM / Viterbi route", fill_route(trip, [hmm[k] for k in sorted(hmm)]), "#b0179a", 3),
    ]:
        g = folium.FeatureGroup(name)
        folium.PolyLine(arc_path(trip, arcs), color=color, weight=w, opacity=.85, tooltip=name).add_to(g)
        g.add_to(m)
    gps = folium.FeatureGroup("Raw GPS observations")
    for i, (lon, lat, t) in enumerate(trip.track):
        if i % 2 == 0:
            folium.CircleMarker([lat, lon], radius=2, color="#1f5fd6", fill=True, weight=0, fill_opacity=.8,
                                tooltip=f"obs {i}, t={t:.0f}s").add_to(gps)
    gps.add_to(m)
    folium.Marker([trip.track[0, 1], trip.track[0, 0]], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([trip.track[-1, 1], trip.track[-1, 0]], tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND.format(tid=trip.id)))
    m.fit_bounds([[trip.track[:, 1].min(), trip.track[:, 0].min()], [trip.track[:, 1].max(), trip.track[:, 0].max()]])
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    model = joblib.load(os.path.join(ROOT, "artifacts", "final_model.joblib"))
    hmm_cfg = json.load(open(os.path.join(ROOT, "artifacts", "hmm_config.json")))
    for t in trip_ids("student_test"):
        make_map(get_trip("student_test", t), model, hmm_cfg).save(os.path.join(OUT, f"trip_{t}.html"))
        print("saved", t)


if __name__ == "__main__":
    main()
