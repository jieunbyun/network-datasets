"""Generate connected spatial graphs that tolerate any single edge failure."""

import argparse
import json
import random
from itertools import combinations
from math import ceil, cos, exp, hypot, isfinite, sin, sqrt
from pathlib import Path

NODE_COUNTS = (20, 40, 80, 160, 320)
BASE_NODE_COUNT = 20
X_RANGE = (0.0, 1.0)  # Domain at BASE_NODE_COUNT.
Y_RANGE = (0.0, 1.0)  # Domain at BASE_NODE_COUNT.
CORRELATION_LENGTH = 0.18  # Fixed length in x/y coordinate units.
CLUSTER_STRENGTH = 1.0     # Log-density standard deviation; zero gives uniform density.
GRID_SIZE = 48
FIELD_MODES = 48
PROB_RANK_2_TO_5 = 0.05
PROB_RANK_6_TO_9 = 0.01
SEED = 7
OUTPUT_DIR = Path("random_graph_generated")


def spatial_density(node_count, seed=SEED):
    """Sample a smooth log-density field and return its grid sampling weights."""
    width = X_RANGE[1] - X_RANGE[0]
    height = Y_RANGE[1] - Y_RANGE[0]
    if min(width, height) <= 0:
        raise ValueError("X_RANGE and Y_RANGE must have increasing endpoints.")
    if not isfinite(CORRELATION_LENGTH) or CORRELATION_LENGTH <= 0:
        raise ValueError("CORRELATION_LENGTH must be positive and finite.")
    if not isfinite(CLUSTER_STRENGTH) or CLUSTER_STRENGTH < 0:
        raise ValueError("CLUSTER_STRENGTH must be nonnegative and finite.")

    if BASE_NODE_COUNT <= 0 or node_count <= 0:
        raise ValueError("Node counts must be positive.")

    # Scale the area in proportion to node count to keep mean density fixed.
    domain_scale = sqrt(node_count / BASE_NODE_COUNT)
    grid_size = ceil(GRID_SIZE * domain_scale)
    cell_width = width * domain_scale / grid_size
    cell_height = height * domain_scale / grid_size

    # Reuse the same realization of the log-density field at every domain size.
    rng = random.Random(seed)
    # Gaussian Fourier frequencies approximate a squared-exponential correlation.
    modes = [
        (rng.gauss(0, 1 / CORRELATION_LENGTH), rng.gauss(0, 1 / CORRELATION_LENGTH),
         rng.gauss(0, 1), rng.gauss(0, 1))
        for _ in range(FIELD_MODES)
    ]
    scale = CLUSTER_STRENGTH / sqrt(FIELD_MODES)
    log_density = []
    for row in range(grid_size):
        y = (row + 0.5) * cell_height
        for column in range(grid_size):
            x = (column + 0.5) * cell_width
            value = 0.0
            for fx, fy, a, b in modes:
                phase = fx * x + fy * y
                value += a * cos(phase) + b * sin(phase)
            log_density.append(scale * value)

    peak = max(log_density)
    weights = [exp(value - peak) for value in log_density]
    return weights, grid_size, cell_width, cell_height


def component_labels(adjacency, excluded=frozenset()):
    labels = [-1] * len(adjacency)
    for root, label in enumerate(labels):
        if label != -1:
            continue
        labels[root] = root
        stack = [root]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                pair = (min(node, neighbor), max(node, neighbor))
                if labels[neighbor] == -1 and pair not in excluded:
                    labels[neighbor] = root
                    stack.append(neighbor)
    return labels


def add_redundant_edges(points, initial_pairs):
    """Join components, then close cycles until no single edge can disconnect the graph."""
    if len(points) < 3:
        raise ValueError("At least three nodes are needed to tolerate a single edge failure.")

    pairs = set(initial_pairs)
    adjacency = [set() for _ in points]
    for source, target in pairs:
        adjacency[source].add(target)
        adjacency[target].add(source)

    candidates = sorted(
        (hypot(points[u][0] - points[v][0], points[u][1] - points[v][1]), u, v)
        for u, v in combinations(range(len(points)), 2)
        if (u, v) not in pairs
    )

    added = []
    while True:
        labels = component_labels(adjacency)
        if len(set(labels)) == 1:
            bridges = set()
            for u, v in pairs:
                labels_without_edge = component_labels(adjacency, {(u, v)})
                if labels_without_edge[u] != labels_without_edge[v]:
                    bridges.add((u, v))
            if not bridges:
                return added
            # Contract the parts that already have alternate paths. A link
            # between two such parts bypasses one or more existing bridges.
            labels = component_labels(adjacency, bridges)

        source, target = next(
            (u, v) for _, u, v in candidates
            if (u, v) not in pairs and labels[u] != labels[v]
        )
        pairs.add((source, target))
        adjacency[source].add(target)
        adjacency[target].add(source)
        added.append((source, target))


def generate_graph(node_count, density_grid, seed=SEED):
    rng = random.Random(seed + node_count)
    weights, grid_size, cell_width, cell_height = density_grid
    cells = rng.choices(range(len(weights)), weights=weights, k=node_count)
    points = []
    for cell in cells:
        row, column = divmod(cell, grid_size)
        x = X_RANGE[0] + (column + rng.random()) * cell_width
        y = Y_RANGE[0] + (row + rng.random()) * cell_height
        points.append((x, y))
    nodes = {f"n{i}": {"x": x, "y": y} for i, (x, y) in enumerate(points)}
    selected_pairs = set()

    for source, (x, y) in enumerate(points):
        candidates = sorted(
            (hypot(x - other_x, y - other_y), target)
            for target, (other_x, other_y) in enumerate(points)
            if target != source
        )[:9]

        for rank, (_, target) in enumerate(candidates, start=1):
            if rank > 1:
                probability = PROB_RANK_2_TO_5 if rank <= 5 else PROB_RANK_6_TO_9
                if rng.random() >= probability:
                    continue

            selected_pairs.add((min(source, target), max(source, target)))

    ordered_pairs = sorted(selected_pairs) + add_redundant_edges(points, selected_pairs)
    edges = {
        f"e{i}": {"from": f"n{source}", "to": f"n{target}", "directed": False}
        for i, (source, target) in enumerate(ordered_pairs)
    }
    return nodes, edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Root directory for the generated datasets (default: %(default)s)")
    args = parser.parse_args()

    for node_count in NODE_COUNTS:
        density_grid = spatial_density(node_count)
        nodes, edges = generate_graph(node_count, density_grid)
        data_dir = args.output_dir / f"spatial_corr_scaled_redundant_n{node_count}" / "v1" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "nodes.json").write_text(json.dumps(nodes, indent=2), encoding="utf-8")
        (data_dir / "edges.json").write_text(json.dumps(edges, indent=2), encoding="utf-8")
        print(f"n={len(nodes)}, edges={len(edges)} -> {data_dir}")


if __name__ == "__main__":
    main()
