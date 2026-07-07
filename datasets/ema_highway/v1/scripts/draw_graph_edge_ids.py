"""Draw the ema_highway network with edge-ID labels.

Standalone: uses only the Python standard library and matplotlib (already a
core dependency of ndtools-duco). No Graphviz installation is required --
node positions are taken directly from the x/y coordinates in nodes.json.

Usage:
    python draw_graph_edge_ids.py [--data-dir DIR] [--out FILE]
                                  [--highlight NODE [NODE ...]]
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Keep SVG text editable (e.g., in Inkscape) instead of outlining it to paths.
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

EDGE_COLOR = "#b0b0b0"
NODE_FACE = "white"
NODE_EDGE = "#606060"
HIGHLIGHT_FACE = "#a8cee2"
LABEL_COLOR = "#1a1a1a"


def draw_graph_edge_ids(
    data_dir: Path,
    out_path: Path,
    highlight_nodes: tuple = (),
    figsize: float = 16.0,
    label_fontsize: float = 40.0,
    node_size: float = 30.0,
    dpi: int = 300,
    full_ids: bool = False,
) -> Path:
    with open(data_dir / "nodes.json", encoding="utf-8") as f:
        nodes = json.load(f)
    with open(data_dir / "edges.json", encoding="utf-8") as f:
        edges = json.load(f)

    pos = {nid: (float(a["x"]), float(a["y"])) for nid, a in nodes.items()}

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_aspect("equal")
    ax.axis("off")

    segments = [(pos[e["from"]], pos[e["to"]]) for e in edges.values()]
    ax.add_collection(
        LineCollection(segments, colors=EDGE_COLOR, linewidths=0.9, zorder=1)
    )

    # Bare numbers by default (44, not e0044) to keep labels compact; with
    # --full-ids, pad only to the width of the largest ID (e044).
    numbers = {
        eid: int(eid.lstrip("e"))
        for eid in edges
        if eid.lstrip("e").isdigit()
    }
    width = max((len(str(n)) for n in numbers.values()), default=0)

    def label_text(eid: str) -> str:
        if eid not in numbers:
            return eid
        if full_ids:
            return f"e{numbers[eid]:0{width}d}"
        return str(numbers[eid])

    texts = []
    for eid, e in edges.items():
        (x1, y1), (x2, y2) = pos[e["from"]], pos[e["to"]]

        # Rotate the label to run along its edge, kept upright for reading.
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if angle > 90:
            angle -= 180
        elif angle <= -90:
            angle += 180

        txt = ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            label_text(eid),
            fontsize=label_fontsize,
            fontweight="bold",
            color=LABEL_COLOR,
            ha="center",
            va="center",
            rotation=angle,
            rotation_mode="anchor",
            zorder=3,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.85),
        )
        texts.append((txt, (x1, y1), (x2, y2)))

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    colors = [
        HIGHLIGHT_FACE if nid in highlight_nodes else NODE_FACE for nid in pos
    ]
    # clip_on=False: markers may poke past the axes box (margins are computed
    # from point positions, ignoring marker radius); let savefig include them.
    ax.scatter(
        xs,
        ys,
        s=node_size,
        c=colors,
        edgecolors=NODE_EDGE,
        linewidths=0.8,
        zorder=2,
        clip_on=False,
    )
    ax.margins(0.02)
    ax.autoscale_view()

    # Short edges in dense areas put several labels on top of each other.
    # For each label, try positions along its edge (with a small perpendicular
    # offset as a fallback) and keep the first that clears every label already
    # placed; if none does, keep the least-overlapping one. The axes limits
    # must be final before this: extents are measured in display space.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def overlap_area(box, others):
        total = 0.0
        for other in others:
            w = min(box.x1, other.x1) - max(box.x0, other.x0)
            h = min(box.y1, other.y1) - max(box.y0, other.y0)
            if w > 0 and h > 0:
                total += w * h
        return total

    # Uniform because the aspect ratio is equal.
    data_per_px = abs(
        ax.transData.inverted().transform((1.0, 0.0))[0]
        - ax.transData.inverted().transform((0.0, 0.0))[0]
    )

    placed = []
    for txt, (x1, y1), (x2, y2) in texts:
        dx, dy = x2 - x1, y2 - y1
        norm = math.hypot(dx, dy) or 1.0
        nx_, ny_ = -dy / norm, dx / norm  # unit normal to the edge
        # Perpendicular step of one label height, so colliding labels can sit
        # beside their edge rather than on top of each other.
        step = txt.get_window_extent(renderer).height * data_per_px
        best_xy, best_score = None, math.inf
        # Prefer sliding along the edge; move off it only as a last resort.
        for off in (0.0, step, -step, 2 * step, -2 * step):
            for t in (0.5, 0.4, 0.6, 0.3, 0.7, 0.22, 0.78):
                xy = (x1 + t * dx + off * nx_, y1 + t * dy + off * ny_)
                txt.set_position(xy)
                box = txt.get_window_extent(renderer).expanded(1.1, 1.1)
                score = overlap_area(box, placed)
                if score < best_score:
                    best_xy, best_score, best_box = xy, score, box
                if score == 0.0:
                    break
            if best_score == 0.0:
                break
        txt.set_position(best_xy)
        placed.append(best_box)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    default_data = Path(__file__).resolve().parents[1] / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument(
        "--out", type=Path, default=default_data / "graph_edge_ids.png"
    )
    parser.add_argument(
        "--highlight", nargs="*", default=[], help="node IDs to fill in blue"
    )
    parser.add_argument(
        "--full-ids",
        action="store_true",
        help="label edges e044 instead of bare numbers (44)",
    )
    parser.add_argument(
        "--fontsize",
        type=float,
        default=18.0,  # keep in sync with label_fontsize in draw_graph_edge_ids
        help="edge label font size (pt)",
    )
    parser.add_argument(
        "--figsize", type=float, default=16.0, help="figure width/height (inches)"
    )
    args = parser.parse_args()
    out = draw_graph_edge_ids(
        args.data_dir,
        args.out,
        highlight_nodes=tuple(args.highlight),
        full_ids=args.full_ids,
        label_fontsize=args.fontsize,
        figsize=args.figsize,
    )
    print(f"Saved {out}")
