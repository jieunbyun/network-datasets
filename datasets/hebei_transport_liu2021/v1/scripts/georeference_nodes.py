"""
Georeference the Hebei nodes: fit figure x/y to latitude/longitude.

The figure coordinates in ``data/nodes.json`` are schematic positions digitised
from Fig. 7 of Liu et al. (2021), which draws the network on a map of Hebei (as
does Fig. 12 of Bai et al. 2021). The 14 labelled cities, with geodetic
coordinates from ``data/geo_control_points.json``, are used as control points to
fit a similarity transform (uniform scale, rotation, translation) from figure
units to a local planar frame in km. The fit then places the 8 unlabelled
junction nodes.

A similarity transform is used rather than an affine one because it predicts
held-out cities better (leave-one-out RMS ~22 km vs ~24 km): the residuals are
node-placement scatter in the drawing, not map distortion an extra shear could
absorb.

Writes ``data/nodes_geo.json``. For labelled nodes, ``lat``/``lon`` are the
city's own coordinates; for junctions they are the fitted position. Both carry
the fitted position and the fit's leave-one-out error so a user can judge it.

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

# Local equirectangular frame. Over Hebei (~600 km) the scale error of this
# approximation is well under 1%, far below the fit residuals.
LAT0, LON0 = 38.8, 116.5
KM_PER_DEG_LAT = 111.195
KM_PER_DEG_LON = KM_PER_DEG_LAT * np.cos(np.radians(LAT0))


def to_km(lat, lon):
    return np.c_[(np.asarray(lon) - LON0) * KM_PER_DEG_LON,
                 (np.asarray(lat) - LAT0) * KM_PER_DEG_LAT]


def to_latlon(east, north):
    return LAT0 + north / KM_PER_DEG_LAT, LON0 + east / KM_PER_DEG_LON


def fit_similarity(fig_xy, km_xy):
    """Least-squares E = a x - b y + c, N = b x + a y + d."""
    x, y = fig_xy[:, 0], fig_xy[:, 1]
    one, zero = np.ones_like(x), np.zeros_like(x)
    design = np.r_[np.c_[x, -y, one, zero], np.c_[y, x, zero, one]]
    params, *_ = np.linalg.lstsq(design, np.r_[km_xy[:, 0], km_xy[:, 1]], rcond=None)
    return params


def apply_similarity(params, fig_xy):
    a, b, c, d = params
    x, y = fig_xy[:, 0], fig_xy[:, 1]
    return np.c_[a * x - b * y + c, b * x + a * y + d]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="print the fit only")
    args = ap.parse_args()

    nodes = json.loads((DATA_DIR / "nodes.json").read_text("utf-8"))
    control = json.loads((DATA_DIR / "geo_control_points.json").read_text("utf-8"))
    ids = [k for k in control if not k.startswith("_")]

    fig_xy = np.array([[nodes[i]["x"], nodes[i]["y"]] for i in ids])
    km_xy = to_km([control[i]["lat"] for i in ids], [control[i]["lon"] for i in ids])

    params = fit_similarity(fig_xy, km_xy)
    residual = np.linalg.norm(apply_similarity(params, fig_xy) - km_xy, axis=1)
    loo = np.empty(len(ids))
    for k in range(len(ids)):
        keep = np.arange(len(ids)) != k
        p_k = fit_similarity(fig_xy[keep], km_xy[keep])
        loo[k] = np.linalg.norm(apply_similarity(p_k, fig_xy[k:k + 1])[0] - km_xy[k])
    loo_rms = float(np.sqrt(np.mean(loo ** 2)))

    a, b = params[:2]
    print(f"scale {np.hypot(a, b):.2f} km per figure unit, rotation {np.degrees(np.arctan2(b, a)):.1f} deg")
    print(f"fit RMS {np.sqrt(np.mean(residual ** 2)):.1f} km, leave-one-out RMS {loo_rms:.1f} km, max {loo.max():.1f} km")
    for i, r, l in zip(ids, residual, loo):
        print(f"  {i:4} {control[i]['label']:4} residual {r:5.1f} km, leave-one-out {l:5.1f} km")
    if args.check:
        return

    out = {
        "_about": (
            "Approximate WGS84 positions for nodes.json, from scripts/georeference_nodes.py. "
            "Labelled nodes: lat/lon are the city's own coordinates (geo_control_points.json). "
            "Junction nodes: lat/lon are fitted from the figure position and are uncertain by "
            f"roughly the leave-one-out RMS of {loo_rms:.0f} km. fitted_lat/fitted_lon is the "
            "figure position mapped through the fit, for every node."
        ),
    }
    loo_by_id = dict(zip(ids, loo))
    res_by_id = dict(zip(ids, residual))
    for node_id, node in nodes.items():
        east, north = apply_similarity(params, np.array([[node["x"], node["y"]]]))[0]
        f_lat, f_lon = to_latlon(east, north)
        rec = {"label": node["label"], "city": node["city"]}
        if node_id in control:
            rec.update(lat=control[node_id]["lat"], lon=control[node_id]["lon"],
                       method="city_coordinate", wikidata=control[node_id]["wikidata"],
                       fit_residual_km=round(float(res_by_id[node_id]), 1),
                       loo_error_km=round(float(loo_by_id[node_id]), 1))
        else:
            rec.update(lat=round(float(f_lat), 4), lon=round(float(f_lon), 4),
                       method="fitted_from_figure", uncertainty_km=round(loo_rms))
        rec.update(fitted_lat=round(float(f_lat), 4), fitted_lon=round(float(f_lon), 4))
        out[node_id] = rec

    path = DATA_DIR / "nodes_geo.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
