"""Draw the ema_highway network with node sizes scaled by population.

Standalone: uses only the Python standard library and matplotlib. Node
positions come directly from the x/y coordinates in nodes.json. Node marker
*diameters* are measured in data units -- i.e. the same units as the edge
lengths -- so nodes stay proportionate to the graph geometry at any figure
size. A size key relates marker size to population. The origin nodes n22 and
n66 are marked with red stars. No IDs are drawn.

Usage:
    python draw_graph_population.py [--data-dir DIR] [--out FILE]
                                    [--origins NODE [NODE ...]]
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
from matplotlib.lines import Line2D

EDGE_COLOR = "#9aa4ad"
NODE_FILL = "#4c78a8"
ORIGIN_COLOR = "#e01010"

# Node *diameters* as a fraction of the median edge length. Area still grows
# with population (diameter ~ sqrt(pop)); the floor keeps zero-pop nodes visible.
MIN_DIAM_FRAC = 0.20
MAX_DIAM_FRAC = 0.85
# Origin-star diameter as a fraction of the median edge length.
STAR_DIAM_FRAC = 1.10
# Population values shown in the size key.
LEGEND_POPS = (500, 1500, 3000)


def _points_per_data_unit(fig, ax) -> float:
    """Length of one data unit expressed in typographic points.

    Marker sizes (scatter ``s``) are areas in points**2, independent of the
    data scale, so to size nodes in data units we convert here. Requires the
    axes limits to be final and the canvas drawn.
    """
    trans = ax.transData
    (x0, _), (x1, _) = trans.transform((0.0, 0.0)), trans.transform((1.0, 0.0))
    px_per_data = abs(x1 - x0)
    return px_per_data * 72.0 / fig.dpi


def draw_graph_population(
    data_dir: Path,
    out_path: Path,
    origin_nodes: tuple = ("n22", "n66"),
    figsize: tuple = (2.2, 2.6),
    dpi: int = 300,
) -> Path:
    with open(data_dir / "nodes.json", encoding="utf-8") as f:
        nodes = json.load(f)
    with open(data_dir / "edges.json", encoding="utf-8") as f:
        edges = json.load(f)

    pos = {nid: (float(a["x"]), float(a["y"])) for nid, a in nodes.items()}
    pop = {nid: float(a.get("population", 0.0)) for nid, a in nodes.items()}

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    segments = [(pos[e["from"]], pos[e["to"]]) for e in edges.values()]
    edge_lengths = [math.dist(a, b) for a, b in segments]
    median_edge = sorted(edge_lengths)[len(edge_lengths) // 2]
    ax.add_collection(
        LineCollection(segments, colors=EDGE_COLOR, linewidths=0.6, zorder=1)
    )

    # Diameter in data units: sqrt scaling so *area* tracks population, with a
    # floor so the smallest nodes stay visible. Reference = median edge length.
    max_pop = max(pop.values()) or 1.0
    min_d = MIN_DIAM_FRAC * median_edge
    max_d = MAX_DIAM_FRAC * median_edge

    def diam_for(p: float) -> float:
        return min_d + (max_d - min_d) * math.sqrt(p / max_pop)

    ax.margins(0.05)
    ax.autoscale_view()

    # Limits are final now; convert data-unit diameters to marker areas (pt**2).
    fig.canvas.draw()
    ppd = _points_per_data_unit(fig, ax)

    def area(diam_units: float) -> float:
        return (diam_units * ppd) ** 2

    plain = [nid for nid in pos if nid not in origin_nodes]
    ax.scatter(
        [pos[n][0] for n in plain],
        [pos[n][1] for n in plain],
        s=[area(diam_for(pop[n])) for n in plain],
        c=NODE_FILL,
        edgecolors="none",
        zorder=2,
        clip_on=False,
    )

    # Origin nodes as red stars, sized to stand out regardless of population.
    origins = [n for n in origin_nodes if n in pos]
    if origins:
        ax.scatter(
            [pos[n][0] for n in origins],
            [pos[n][1] for n in origins],
            s=area(STAR_DIAM_FRAC * median_edge),
            marker="*",
            c=ORIGIN_COLOR,
            edgecolors="none",
            zorder=3,
            clip_on=False,
        )

    # Size key: reference circles at known populations, drawn with the exact
    # same size mapping as the nodes (markersize is a diameter in points, so
    # it equals the data-unit diameter times points-per-data-unit).
    handles = [
        Line2D(
            [], [],
            marker="o",
            linestyle="none",
            markerfacecolor=NODE_FILL,
            markeredgecolor="none",
            markersize=diam_for(p) * ppd,
            label=f"{p:,}",
        )
        for p in LEGEND_POPS
    ]
    legend = ax.legend(
        handles=handles,
        title="Population",
        loc="lower left",
        frameon=False,
        labelspacing=1.0,
        handletextpad=0.8,
        borderpad=0.0,
        fontsize=5,
        title_fontsize=6,
    )
    legend.get_title().set_ha("left")

    # Drop the axes clip so the SVG has no clip-path on every element -- that
    # clip is what makes elements vanish when dragged in Inkscape. Nothing is
    # drawn outside the data extent here, so removing it changes no pixels.
    for artist in (*ax.collections, *ax.texts, *ax.get_children()):
        artist.set_clip_on(False)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    # Also emit a vector copy (SVG) alongside, unless the primary output is
    # already a vector format.
    outputs = [out_path]
    if out_path.suffix.lower() not in (".svg", ".pdf", ".eps"):
        vector_path = out_path.with_suffix(".svg")
        fig.savefig(vector_path, bbox_inches="tight")
        outputs.append(vector_path)
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    default_data = Path(__file__).resolve().parents[1] / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument(
        "--out", type=Path, default=default_data / "graph_population.png"
    )
    parser.add_argument(
        "--origins",
        nargs="*",
        default=["n22", "n66"],
        help="node IDs to mark with red stars",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(2.2, 2.6),
        metavar=("W", "H"),
        help="figure width and height in inches",
    )
    args = parser.parse_args()
    outputs = draw_graph_population(
        args.data_dir,
        args.out,
        origin_nodes=tuple(args.origins),
        figsize=tuple(args.figsize),
    )
    for out in outputs:
        print(f"Saved {out}")
