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

Labels are the exact keys of the data files -- `bus7` from nodes.json on the
circles, `br3` from edges.json on the branches -- so anything in the drawing can
be looked up directly. `--bare-ids` strips them to 7 and 3, which is what the
118-bus drawing needs to keep labels inside the circles. `--vbus-labels` puts
the failable component `vbus7` on the circle instead of the node name `bus7`.

Node positions come from the x/y fields in nodes.json and nowhere else; no
layout algorithm is involved. `--relax` only nudges apart buses whose circles
would overlap (see relax_positions), and `--relax 0` disables even that.

Usage:
    python draw_graph_ids.py [--data-dir DIR] [--out FILE]
                             [--highlight BUS [BUS ...]]
                             [--bare-ids] [--vbus-labels]
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

# Branches are blue and leaders stay grey: a leader is an annotation, not part
# of the network, and at this density the two read as the same kind of line if
# they share a colour.
EDGE_COLOR = "#4c78a8"
NODE_EDGE = "#606060"
LEADER_COLOR = "#8a8a8a"
HIGHLIGHT_FACE = "#a8cee2"
LABEL_COLOR = "#1a1a1a"
# Bus roles, keyed by the `type` field of the busN node in nodes.json.
TYPE_FACE = {
    "source": "#f2d9a0",        # generation, and often load too -- 4-state
    "output only": "white",     # load, no generation            -- 2-state
    "transmission": "#e8e8e8",  # neither                        -- 2-state
}
TYPE_LABEL = {
    "source": "generator bus (4-state)",
    "output only": "load only, no generation (2-state)",
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


def nearest_on_segment(px, py, x1, y1, x2, y2):
    """Point of the segment (x1,y1)-(x2,y2) closest to (px,py)."""
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return x1, y1
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return x1 + t * dx, y1 + t * dy


def box_exit(cx, cy, w, h, tx, ty):
    """
    Where the ray from a box's centre towards (tx,ty) leaves the box.

    Used to start a leader at the edge of the label rather than under it, so
    the line does not run across its own text.
    """
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    scale = min(
        abs(w / 2 / dx) if dx else math.inf,
        abs(h / 2 / dy) if dy else math.inf,
    )
    return cx + dx * scale, cy + dy * scale


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
    full_ids: bool = True,
    vbus_labels: bool = False,
    relax: float = 1.5,
    legend_loc: str = "lower left",
    split_buses: bool = False,
    leaders: bool = True,
    leader_thres: float = 1.2,
) -> Path:
    with open(data_dir / "nodes.json", encoding="utf-8") as f:
        nodes = json.load(f)
    with open(data_dir / "edges.json", encoding="utf-8") as f:
        edges = json.load(f)

    # Collapsing `busN` and `busN_int` onto one circle is what hides the `vbusN`
    # component: its two endpoints become the same point, so it is dropped from
    # `branches` below. Leaving `canon` as the identity keeps the pair apart, and
    # every node and every edge in the data files then gets its own mark.
    canon = (lambda nid: nid) if split_buses else canonical

    # One position per drawn node, taken from its own x/y in nodes.json.
    pos = {
        nid: (float(a["x"]), float(a["y"]))
        for nid, a in nodes.items()
        if split_buses or not nid.endswith("_int")
    }
    node_type = {
        nid: a.get("type", "transmission")
        for nid, a in nodes.items()
        if split_buses or not nid.endswith("_int")
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

    # Without --split-buses the `vbusN` edges are the buses themselves and are
    # drawn as circles, not lines, so only the `brN` branches become segments.
    # With it, both endpoints survive and all 34 edges are drawn.
    branches = {
        eid: e for eid, e in edges.items()
        if canon(e["from"]) != canon(e["to"])
    }
    vbus_of = {
        canon(e["from"]): eid for eid, e in edges.items()
        if canon(e["from"]) == canon(e["to"])
    }

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_aspect("equal")
    ax.axis("off")

    segments = [
        (pos[canon(e["from"])], pos[canon(e["to"])])
        for e in branches.values()
    ]
    ax.add_collection(
        LineCollection(segments, colors=EDGE_COLOR, linewidths=0.9, zorder=1)
    )

    # With --full-ids (the default) a label is the exact key of nodes.json /
    # edges.json -- `bus7`, `br3` -- so it can be matched against the data files
    # character for character. --bare-ids drops the prefixes to bare numbers,
    # which is the only way the labels fit inside the circles at 118-bus scale.
    branch_numbers = strip_prefix(branches, "br")
    bus_numbers = strip_prefix(pos, "bus")

    def edge_label(eid: str) -> str:
        if full_ids or eid not in branch_numbers:
            return eid
        return str(branch_numbers[eid])

    def node_label(nid: str) -> str:
        # `bus7` names the node; `vbus7` is the failable component drawn as the
        # same circle. --vbus-labels asks for the latter.
        if vbus_labels:
            return vbus_of.get(nid, nid)
        if full_ids or nid not in bus_numbers:
            return nid
        return str(bus_numbers[nid])

    highlight = {canon(n) for n in highlight_nodes}

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
        (x1, y1) = pos[canon(e["from"])]
        (x2, y2) = pos[canon(e["to"])]
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

    settled = []  # (final label centre, box width, box height, edge) in data units
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
        settled.append((best_xy, w * data_per_px, h * data_per_px,
                        (x1, y1, x2, y2)))

    # A label pushed clear of the congestion no longer reads as belonging to any
    # particular edge -- the `vbusN` labels on a split-bus drawing end up in the
    # margin, since their edge is shorter than the label is wide. Give those a
    # leader back to the nearest point of the edge they name. Labels still
    # sitting on their edge get nothing, so the leaders stay rare enough to
    # read as pointers rather than as more network.
    if leaders:
        for (lx, ly), box_w, box_h, (x1, y1, x2, y2) in settled:
            tx, ty = nearest_on_segment(lx, ly, x1, y1, x2, y2)
            if math.hypot(tx - lx, ty - ly) < leader_thres * box_h:
                continue
            sx, sy = box_exit(lx, ly, box_w, box_h, tx, ty)
            ax.annotate(
                "",
                xy=(tx, ty),
                xytext=(sx, sy),
                arrowprops=dict(arrowstyle="->", color=LEADER_COLOR,
                                linewidth=0.7, shrinkA=0, shrinkB=0,
                                # Scale the head with the text, or it vanishes
                                # at the font sizes a dense network needs.
                                mutation_scale=label_fontsize * 1.4),
                annotation_clip=False,
                # Above the circles (2) but below the labels (3): a `vbusN` edge
                # lies entirely between its two circles, so a head drawn at the
                # circles' depth is hidden by them.
                zorder=2.6,
            )

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
    ax.legend(handles=handles, loc=legend_loc, frameon=True, framealpha=0.9,
              fontsize=label_fontsize * 1.8, borderpad=0.8, labelspacing=0.8)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# The two views cannot share one geometry. The collapsed view wants circles big
# enough to hold `bus10`; the split view must keep them under the 0.212 that
# separates a `busN`/`busN_int` pair, or the two merge into one blob. Each view
# therefore carries its own numbers, and any option given on the command line
# overrides that view's value.
VIEWS = {
    "graph_ids": dict(
        split_buses=False,
        figsize=11.0,
        label_fontsize=9.0,
        node_size=900.0,
        relax=1.5,
        legend_loc="upper left",
    ),
    "graph_ids_split": dict(
        split_buses=True,
        figsize=16.0,
        label_fontsize=11.0,
        node_size=500.0,
        # Circles this large would be pushed off their stored coordinates by
        # any relaxation, and the split view exists to show the real geometry.
        relax=0.0,
        legend_loc="upper left",
    ),
}
FORMATS = ("png", "svg")


if __name__ == "__main__":
    default_data = Path(__file__).resolve().parents[1] / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="write one file here instead of the default set; the view is "
             "chosen by --split-buses and the format by the suffix",
    )
    parser.add_argument(
        "--highlight", nargs="*", default=[],
        help="buses to fill in blue, as 21 or bus21",
    )
    parser.add_argument(
        "--bare-ids",
        dest="full_ids",
        action="store_false",
        help="label 7 / 3 instead of the full keys bus7 / br3; the bare numbers "
             "fit inside the circles, which the full keys only do at a large "
             "--node-size",
    )
    parser.add_argument(
        "--vbus-labels",
        action="store_true",
        help="label each circle with the failable component from edges.json "
             "(vbus7) rather than the node from nodes.json (bus7); ignored "
             "under --split-buses, where vbus7 labels its own edge",
    )
    parser.add_argument(
        "--no-leaders",
        dest="leaders",
        action="store_false",
        help="do not draw an arrow from a displaced edge label back to its edge",
    )
    parser.add_argument(
        "--leader-thres",
        type=float,
        default=1.2,
        help="how far a label must sit from its edge, in label heights, before "
             "it earns a leader arrow; lower draws more (default: 1.2)",
    )
    parser.add_argument(
        "--split-buses",
        action="store_true",
        help="draw busN and busN_int as two circles joined by the vbusN edge, "
             "so every node and edge in the data files gets its own mark. The "
             "pair is only 0.21 apart, so use a small --node-size and "
             "--relax 0, or the circles collide and get pushed off position",
    )
    # These default to None so an unset option leaves the per-view value in
    # VIEWS alone; anything actually passed applies to every view drawn.
    parser.add_argument(
        "--fontsize", type=float, default=None,
        help="label font size (pt); overrides the per-view default",
    )
    parser.add_argument(
        "--figsize", type=float, default=None,
        help="figure width/height (inches); overrides the per-view default",
    )
    parser.add_argument(
        "--node-size", type=float, default=None,
        help="node marker area in points^2 (matplotlib scatter s); overrides "
             "the per-view default",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--legend-loc", type=str, default=None,
        help="matplotlib legend location; the box is opaque and will hide any "
             "buses beneath it, so move it to whichever corner is empty",
    )
    parser.add_argument(
        "--relax", type=float, default=None,
        help="minimum bus separation as a multiple of the circle diameter; "
             "0 keeps the stored coordinates exactly",
    )
    args = parser.parse_args()
    highlight = tuple(
        h if str(h).startswith("bus") else f"bus{h}" for h in args.highlight
    )
    overrides = {
        name: value
        for name, value in (
            ("label_fontsize", args.fontsize),
            ("figsize", args.figsize),
            ("node_size", args.node_size),
            ("relax", args.relax),
            ("legend_loc", args.legend_loc),
        )
        if value is not None
    }

    if args.out is not None:
        view = "graph_ids_split" if args.split_buses else "graph_ids"
        targets = [(args.out, VIEWS[view])]
    else:
        # Both views in both formats: the PNG to look at, the SVG to zoom into
        # where the drawing is too dense to read at page size.
        targets = [
            (args.data_dir / f"{view}.{fmt}", preset)
            for view, preset in VIEWS.items()
            for fmt in FORMATS
        ]

    for out_path, preset in targets:
        out = draw_graph_ids(
            args.data_dir,
            out_path,
            highlight_nodes=highlight,
            full_ids=args.full_ids,
            vbus_labels=args.vbus_labels,
            dpi=args.dpi,
            leaders=args.leaders,
            leader_thres=args.leader_thres,
            **{**preset, **overrides},
        )
        print(f"Saved {out}")
