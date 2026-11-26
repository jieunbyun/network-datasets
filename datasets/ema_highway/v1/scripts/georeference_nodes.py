"""
Georeference the EMA nodes: fit dataset x/y to latitude/longitude.

The 74 nodes and 129 links are the Eastern Massachusetts highway subnetwork of
the TransportationNetworks benchmark (bstabler/TransportationNetworks,
"Eastern-Massachusetts"; 258 directed links there = the 129 undirected links
here, under the same node numbering). Its EMA_intro.pdf names the town or zone
of every node in the caption of Fig. 3 -- reproduced in ``ZONES`` below.

Nodes in a single-town zone are control points at that town's coordinates
(``data/geo_control_points.json``). A similarity transform (uniform scale,
rotation, translation) from dataset x/y to a local planar frame in km is fitted
to them, and then places every node. Several nodes share one town (Waltham has
six), so every node takes its *fitted* position: town centres would stack them
on one point and erase the network's own geometry.

Accuracy is scored by leaving out one whole town at a time -- leaving out one
node would let its town-mates, at the same coordinates, give the answer away.
A similarity transform is used rather than an affine one because it predicts
held-out towns better.

Writes ``data/nodes_geo.json``.

Usage:
    python georeference_nodes.py                 # writes ../data/nodes_geo.json
    python georeference_nodes.py --check         # print the fit, write nothing
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"

# Fig. 3 caption, EMA_intro.pdf ("nodes:zone" pairs), verbatim apart from layout.
ZONES = {
    "Seabrook (NH)": [1], "NH": [2, 4, 5], "Haverhill": [3], "Lawrence": [6, 8], "Georgetown": [7, 9],
    "Lowell": [10, 11], "Salem": [12, 15], "Peabody": [13, 14], "Burlington": [16, 17, 18, 19],
    "Littleton": [20], "Lexington": [21], "Boston": [22], "Waltham": [23, 24, 25, 26, 27, 28],
    "Quincy": [29], "Marlborough/Framingham": [30, 31, 32, 33, 34], "Milford": [35, 71],
    "Franklin": [36], "Westwood/Quincy": [37, 38, 39, 40, 41], "Dedham": [42],
    "Foxborough": [43, 44, 45, 46, 47], "Taunton": [48, 74], "Plymouth": [49, 73], "Cape Cod": [50, 51],
    "Dartmouth": [52], "Fall River": [53], "RI": [54, 68, 70], "VT": [55, 56], "Westminster": [57],
    "Leominster": [58], "Worcester": [59, 60, 72], "Amherst": [61], "CT": [62, 63, 64, 65, 66],
    "Webster": [67], "Uxbridge": [69],
}

# Local equirectangular frame centred on the network; its scale error over
# ~150 km is well under 1%, far below the fit residuals.
LAT0, LON0 = 42.3, -71.4
KM_PER_DEG_LAT = 111.195
KM_PER_DEG_LON = KM_PER_DEG_LAT * np.cos(np.radians(LAT0))


def to_km(lat, lon):
    return np.c_[(np.asarray(lon) - LON0) * KM_PER_DEG_LON, (np.asarray(lat) - LAT0) * KM_PER_DEG_LAT]


def to_latlon(east, north):
    return LAT0 + north / KM_PER_DEG_LAT, LON0 + east / KM_PER_DEG_LON


def fit_similarity(xy, km):
    """Least-squares E = a x - b y + c, N = b x + a y + d."""
    x, y = xy[:, 0], xy[:, 1]
    one, zero = np.ones_like(x), np.zeros_like(x)
    design = np.r_[np.c_[x, -y, one, zero], np.c_[y, x, zero, one]]
    params, *_ = np.linalg.lstsq(design, np.r_[km[:, 0], km[:, 1]], rcond=None)
    return params


def apply_similarity(params, xy):
    a, b, c, d = params
    return np.c_[a * xy[:, 0] - b * xy[:, 1] + c, b * xy[:, 0] + a * xy[:, 1] + d]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="print the fit only")
    args = ap.parse_args()

    nodes = json.loads((DATA_DIR / "nodes.json").read_text("utf-8"))
    control = json.loads((DATA_DIR / "geo_control_points.json").read_text("utf-8"))
    towns = {k: v for k, v in control.items() if not k.startswith("_")}
    zone_of = {f"n{n}": zone for zone, ns in ZONES.items() for n in ns}
    assert set(zone_of) == set(nodes), "ZONES must name every node exactly once"

    ids, town_of, lat, lon = [], [], [], []
    for town, rec in towns.items():
        for node_id in rec["nodes"]:
            ids.append(node_id); town_of.append(town); lat.append(rec["lat"]); lon.append(rec["lon"])
    xy = np.array([[nodes[i]["x"], nodes[i]["y"]] for i in ids])
    km = to_km(lat, lon)

    params = fit_similarity(xy, km)
    residual = np.linalg.norm(apply_similarity(params, xy) - km, axis=1)
    held_out = np.empty(len(ids))
    for town in towns:
        out = np.array([t == town for t in town_of])
        held_out[out] = np.linalg.norm(apply_similarity(fit_similarity(xy[~out], km[~out]), xy[out]) - km[out], axis=1)
    loo_rms = float(np.sqrt(np.mean(held_out ** 2)))

    a, b = params[:2]
    print(f"scale {np.hypot(a, b):.3f} real km per dataset unit, rotation {np.degrees(np.arctan2(b, a)):.1f} deg")
    print(f"{len(ids)} control nodes in {len(towns)} towns: fit RMS {np.sqrt(np.mean(residual ** 2)):.1f} km, "
          f"leave-one-town-out RMS {loo_rms:.1f} km, max {held_out.max():.1f} km")
    worst = sorted(zip(held_out, ids, town_of), reverse=True)[:5]
    print("largest held-out errors: " + ", ".join(f"{i} ({t}) {e:.0f} km" for e, i, t in worst))
    if args.check:
        return

    res_by_id, loo_by_id = dict(zip(ids, residual)), dict(zip(ids, held_out))
    out = {
        "_about": (
            "Approximate WGS84 positions for nodes.json, from scripts/georeference_nodes.py: every node's "
            "x/y mapped through a similarity transform fitted to town centres "
            f"(leave-one-town-out RMS {loo_rms:.0f} km). 'zone' is the node's town or zone in the source "
            "benchmark. x/y are in units of about "
            f"{np.hypot(a, b):.2f} real km, not kilometres."
        ),
    }
    for node_id, node in nodes.items():
        east, north = apply_similarity(params, np.array([[node["x"], node["y"]]]))[0]
        f_lat, f_lon = to_latlon(east, north)
        rec = {"zone": zone_of[node_id], "lat": round(float(f_lat), 4), "lon": round(float(f_lon), 4),
               "method": "fitted_from_figure"}
        if node_id in res_by_id:
            rec.update(control_town=town_of[ids.index(node_id)],
                       fit_residual_km=round(float(res_by_id[node_id]), 1),
                       held_out_error_km=round(float(loo_by_id[node_id]), 1))
        else:
            rec["uncertainty_km"] = round(loo_rms)
        out[node_id] = rec

    path = DATA_DIR / "nodes_geo.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
