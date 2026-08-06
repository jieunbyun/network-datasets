"""Draw the IEEE 118-bus network with node-ID and edge-ID labels.

Standalone: uses only the Python standard library and matplotlib. No Graphviz
installation is required -- node positions are taken directly from the x/y
coordinates in nodes.json.

Adapted from the toynet_11edges drawer for a network an order of magnitude
denser (118 buses, 304 failable components), which forces three changes:

  * Each bus is stored as a *pair* of nodes, `busN` and `busN_int`, joined by
    the virtual edge `vbusN` -- that edge is the failable bus component. The
    two sit 0.15 apart, so drawing them separately would be noise at this
    scale. They are collapsed into a single circle carrying the `vbusN` ID,
    and only the 186 branches `brN` are drawn as edges.
  * Edge labels are rotated along their branch instead of staying horizontal;
    with 186 of them, horizontal labels collide far more than they clear.
  * Label placement measures each label once and translates the box, rather
    than re-measuring per candidate position -- otherwise the collision search
    dominates the runtime at this component count.

The stored coordinates are geographic and heavily clumped -- `bus114` and
`bus115` share a point exactly, and a tenth of all buses sit closer to their
nearest neighbour than one circle's width -- so the circles collide before any
label does. `--relax` (on by default) pushes only the colliding buses apart
while anchoring every bus to its stored position, which clears the pile-ups
without discarding the recognisable geography the way a spring layout would.
`--relax 0` restores strictly faithful positions.

Usage:
    python draw_graph_ids.py [--data-dir DIR] [--out FILE]
                             [--highlight BUS [BUS ...]] [--full-ids]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
# Keep SVG text editable (e.g., in Inkscape) instead of outlining it to paths.
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

EDGE_COLOR = "#b0b0b0"
NODE_EDGE = "#606060"
HIGHLIGHT_FACE = "#a8cee2"
LABEL_COLOR = "#1a1a1a"
# Bus roles, keyed by the `type` field of the busN node in nodes.json.
TYPE_FACE = {
    "source": "#f2d9a0",       # generator bus  -- vbusN is 4-state
    "output": "white",         # load bus       -- vbusN is 2-state
    "transmission": "#e8e8e8",  # no generation, no load
}
TYPE_LABEL = {
    "source": "generator bus (4-state)",
    "output": "load bus (2-state)",
    "transmission": "transmission only (2-state)",
}


def canonical(node_id):
    """`bus7_int` -> `bus7`; the two are one bus, drawn as one circle."""
    return node_id[: -len("_int")] if node_id.endswith("_int") else node_id


def relax_positions(pos, min_sep, iters=300, anchor=0.06):
    """
    Separate buses that sit closer together than `min_sep`, holding each near
    its stored coordinate.

    Repulsion acts only on pairs that are actually too close, so buses with
    room to spare never move; the anchor term pulls everything back towards
    nodes.json, so the drawing keeps its geographic shape. Returns a new
    position dict.
    """
    keys = list(pos)
    p = np.array([pos[k] for k in keys], dtype=float)
    p0 = p.copy()

    # Exactly coincident buses have no direction to separate along; offset them
    # deterministically (by index) so the repulsion has something to work with.
    for i in range(len(p)):
        same = np.flatnonzero((np.abs(p[:i] - p[i]) < 1e-12).all(axis=1))
        if same.size:
            angle = 2 * math.pi * i / len(p)
            p[i] += 1e-3 * min_sep * np.array([math.cos(angle), math.sin(angle)])

    for _ in range(iters):
        delta = p[:, None, :] - p[None, :, :]
        dist = np.hypot(delta[..., 0], delta[..., 1])
        np.fill_diagonal(dist, np.inf)
        close = dist < min_sep
        if not close.any():
            break
        # Half the shortfall each, so a colliding pair meets in the middle.
        push = np.where(close, (min_sep - dist) / 2, 0.0) / np.maximum(dist, 1e-12)
        p = p + (push[..., None] * delta).sum(axis=1) - anchor * (p - p0)

    return {k: (float(x), float(y)) for k, (x, y) in zip(keys, p)}


def strip_prefix(ids, prefix):
    """Map IDs like br04/bus2 to their bare numbers; non-numeric IDs are skipped."""
    return {
        i: int(i[len(prefix):])
        for i in ids
        if i.startswith(prefix) and i[len(prefix):].isdigit()
    }


def draw_graph_ids(
    data_dir: Path,
    out_path: Path,
    highlight_nodes: tuple = (),
    figsize: float = 16.0,
    label_fontsize: float = 5.5,
    node_size: float = 150.0,
    dpi: int = 300,
    full_ids: bool = False,
    relax: float = 1.5,
) -> Path:
    with open(data_dir / "nodes.json", encoding="utf-8") as f:
        nodes = json.load(f)
    with open(data_dir / "edges.json", encoding="utf-8") as f:
        edges = json.load(f)

    # One position per bus, taken from the outer `busN` node.
    pos = {
        nid: (float(a["x"]), float(a["y"]))
        for nid, a in nodes.items()
        if not nid.endswith("_int")
    }
    node_type = {
        nid: a.get("type", "transmission")
        for nid, a in nodes.items()
        if not nid.endswith("_int")
    }

    if relax > 0:
        # Circle diameter converted to data units: sqrt(area) points -> inches
        # -> data, using the coordinate span the figure has to cover.
        span = max(
            max(p[0] for p in pos.values()) - min(p[0] for p in pos.values()),
            max(p[1] for p in pos.values()) - min(p[1] for p in pos.values()),
        )
        diameter = math.sqrt(node_size) / 72 * (span / figsize)
        pos = relax_positions(pos, min_sep=relax * diameter)

    # `vbusN` edges are the buses themselves and are drawn as circles, not
    # lines; only the `brN` branches become segments.
    branches = {
        eid: e for eid, e in edges.items()
        if canonical(e["from"]) != canonical(e["to"])
    }
    vbus_of = {
        canonical(e["from"]): eid for eid, e in edges.items()
        if canonical(e["from"]) == canonical(e["to"])
    }

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_aspect("equal")
    ax.axis("off")

    segments = [
        (pos[canonical(e["from"])], pos[canonical(e["to"])])
        for e in branches.values()
    ]
    ax.add_collection(
        LineCollection(segments, colors=EDGE_COLOR, linewidths=0.9, zorder=1)
    )

    # Bare numbers by default (7, not br07/vbus7) to keep labels compact; with
    # --full-ids, pad branch numbers only to the width of the largest ID.
    branch_numbers = strip_prefix(branches, "br")
    bus_numbers = strip_prefix(pos, "bus")
    width = max((len(str(n)) for n in branch_numbers.values()), default=0)

    def edge_label(eid: str) -> str:
        if eid not in branch_numbers:
            return eid
        if full_ids:
            return f"br{branch_numbers[eid]:0{width}d}"
        return str(branch_numbers[eid])

    def node_label(nid: str) -> str:
        if nid not in bus_numbers:
            return nid
        if full_ids:
            return vbus_of.get(nid, nid)
        return str(bus_numbers[nid])

    highlight = {canonical(n) for n in highlight_nodes}

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    colors = [
        HIGHLIGHT_FACE if nid in highlight
        else TYPE_FACE.get(node_type[nid], "white")
        for nid in pos
    ]
    # clip_on=False: markers may poke past the axes box (margins are computed
    # from point positions, ignoring marker radius); let savefig include them.
    ax.scatter(
        xs,
        ys,
        s=node_size,
        c=colors,
        edgecolors=NODE_EDGE,
        linewidths=0.7,
        zorder=2,
        clip_on=False,
    )
    node_texts = []
    for nid, (x, y) in pos.items():
        node_texts.append(
            ax.text(
                x,
                y,
                node_label(nid),
                fontsize=label_fontsize,
                fontweight="bold",
                color=LABEL_COLOR,
                ha="center",
                va="center",
                zorder=3,
            )
        )

    # Rotate each branch label along its own branch, kept upright.
    texts = []
    for eid, e in branches.items():
        (x1, y1) = pos[canonical(e["from"])]
        (x2, y2) = pos[canonical(e["to"])]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        txt = ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            edge_label(eid),
            fontsize=label_fontsize,
            fontweight="bold",
            color=LABEL_COLOR,
            ha="center",
            va="center",
            rotation=angle,
            rotation_mode="anchor",
            zorder=3,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.3, alpha=0.85),
        )
        texts.append((txt, (x1, y1), (x2, y2)))

    ax.margins(0.03)
    ax.autoscale_view()

    # Edge labels sharing a region would sit on top of each other (or on a
    # node). For each label, try positions along its edge (with a small
    # perpendicular offset as a fallback) and keep the first that clears
    # everything already placed; if none does, keep the least-overlapping one.
    # The axes limits must be final before this: extents are in display space.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def overlap_area(box, others):
        total = 0.0
        for other in others:
            w = min(box[2], other[2]) - max(box[0], other[0])
            h = min(box[3], other[3]) - max(box[1], other[1])
            if w > 0 and h > 0:
                total += w * h
        return total

    def extents(artist, pad=0.0):
        b = artist.get_window_extent(renderer)
        return (b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad)

    # Uniform because the aspect ratio is equal.
    data_per_px = abs(
        ax.transData.inverted().transform((1.0, 0.0))[0]
        - ax.transData.inverted().transform((0.0, 0.0))[0]
    )

    # Edge labels must also keep clear of the labelled node circles.
    node_radius_px = math.sqrt(node_size) / 2 * fig.dpi / 72
    placed = [extents(t, node_radius_px) for t in node_texts]

    for txt, (x1, y1), (x2, y2) in texts:
        dx, dy = x2 - x1, y2 - y1
        norm = math.hypot(dx, dy) or 1.0
        nx_, ny_ = -dy / norm, dx / norm  # unit normal to the edge
        # Measure the label once at its midpoint; every candidate is the same
        # box translated, since the text is centre-anchored and its rotation
        # is fixed. Re-measuring per candidate is what makes the naive search
        # unusably slow at 186 branches.
        x0, y0, x1b, y1b = extents(txt)
        w, h = x1b - x0, y1b - y0
        mid_px = ax.transData.transform(((x1 + x2) / 2, (y1 + y2) / 2))
        # Perpendicular step of one label height, so colliding labels can sit
        # beside their edge rather than on top of each other.
        step = h * data_per_px
        best_xy, best_score, best_box = None, math.inf, None
        # Prefer sliding along the edge; move off it only as a last resort.
        for off in (0.0, step, -step, 2 * step, -2 * step):
            for t in (0.5, 0.4, 0.6, 0.3, 0.7, 0.22, 0.78):
                xy = (x1 + t * dx + off * nx_, y1 + t * dy + off * ny_)
                px, py = ax.transData.transform(xy)
                sx, sy = px - mid_px[0], py - mid_px[1]
                box = (x0 + sx, y0 + sy, x0 + sx + w, y0 + sy + h)
                score = overlap_area(box, placed)
                if score < best_score:
                    best_xy, best_score, best_box = xy, score, box
                if score == 0.0:
                    break
            if best_score == 0.0:
                break
        txt.set_position(best_xy)
        placed.append(best_box)

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor=face, markeredgecolor=NODE_EDGE,
               label=TYPE_LABEL[key])
        for key, face in TYPE_FACE.items()
    ]
    if highlight:
        handles.append(
            Line2D([], [], marker="o", linestyle="none", markersize=8,
                   markerfacecolor=HIGHLIGHT_FACE, markeredgecolor=NODE_EDGE,
                   label="highlighted")
        )
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.9,
              fontsize=label_fontsize * 1.8, borderpad=0.8, labelspacing=0.8)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    default_data = Path(__file__).resolve().parents[1] / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument(
        "--out", type=Path, default=default_data / "graph_ids.png"
    )
    parser.add_argument(
        "--highlight", nargs="*", default=[],
        help="buses to fill in blue, as 21 or bus21",
    )
    parser.add_argument(
        "--full-ids",
        action="store_true",
        help="label br007 / vbus7 instead of bare numbers (7); the longer text "
             "overflows the circles unless --node-size is raised too",
    )
    parser.add_argument(
        "--fontsize",
        type=float,
        default=5.5,  # keep in sync with label_fontsize in draw_graph_ids
        help="label font size (pt)",
    )
    parser.add_argument(
        "--figsize", type=float, default=16.0, help="figure width/height (inches)"
    )
    parser.add_argument(
        "--node-size",
        type=float,
        default=150.0,  # keep in sync with node_size in draw_graph_ids
        help="node marker area in points^2 (matplotlib scatter s)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--relax",
        type=float,
        default=1.5,
        help="minimum bus separation as a multiple of the circle diameter; "
             "0 keeps the stored coordinates exactly (default: 1.5)",
    )
    args = parser.parse_args()
    highlight = tuple(
        h if str(h).startswith("bus") else f"bus{h}" for h in args.highlight
    )
    out = draw_graph_ids(
        args.data_dir,
        args.out,
        highlight_nodes=highlight,
        full_ids=args.full_ids,
        label_fontsize=args.fontsize,
        figsize=args.figsize,
        node_size=args.node_size,
        dpi=args.dpi,
        relax=args.relax,
    )
    print(f"Saved {out}")
