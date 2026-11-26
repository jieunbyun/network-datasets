"""Find the nearest distinct origin and destination nodes for each location pair."""

import argparse
import json
from heapq import heappop, heappush
from math import dist, inf, isfinite, sqrt
from pathlib import Path

from .generate import BASE_NODE_COUNT, NODE_COUNTS, OUTPUT_DIR, X_RANGE, Y_RANGE

# Each entry is ((origin_x, origin_y), (destination_x, destination_y)).
LOCATION_PAIRS = {
    "west_east": ((0.15, 0.50), (0.85, 0.50)),
    "south_north": ((0.50, 0.15), (0.50, 0.85)),
    "southwest_northeast": ((0.15, 0.15), (0.85, 0.85)),
}


def target_position(target, domain, absolute):
    if any(not isfinite(value) for value in target):
        raise ValueError(f"Target {target} must contain finite coordinates.")
    if absolute:
        if any(not low <= value <= high for value, (low, high) in zip(target, domain)):
            raise ValueError(f"Target {target} is outside the domain {domain}.")
        return target
    if any(not 0 <= value <= 1 for value in target):
        raise ValueError(f"Relative target {target} must lie in [0, 1] × [0, 1].")
    return tuple(low + value * (high - low) for value, (low, high) in zip(target, domain))


def nearest_distinct_pair(points, origin_target, destination_target):
    origin_distances = {node: dist(xy, origin_target) for node, xy in points.items()}
    destination_distances = {node: dist(xy, destination_target) for node, xy in points.items()}
    return min(
        ((origin, destination) for origin in points for destination in points
         if origin != destination),
        key=lambda pair: origin_distances[pair[0]] + destination_distances[pair[1]],
    )


def shortest_path_length(adjacency, origin, destination):
    distances = {origin: 0.0}
    queue = [(0.0, origin)]
    while queue:
        distance, node = heappop(queue)
        if distance != distances[node]:
            continue
        if node == destination:
            return distance
        for neighbor, length in adjacency[node]:
            new_distance = distance + length
            if new_distance < distances.get(neighbor, inf):
                distances[neighbor] = new_distance
                heappush(queue, (new_distance, neighbor))
    return None


def find_pairs(data_dir, absolute):
    nodes = json.loads((data_dir / "nodes.json").read_text(encoding="utf-8"))
    edges = json.loads((data_dir / "edges.json").read_text(encoding="utf-8"))
    return select_pairs(nodes, edges, absolute=absolute)


def select_pairs(nodes, edges, absolute=False):
    """Select OD pairs and compute path lengths from node and edge mappings."""
    if len(nodes) < 2:
        raise ValueError("At least two nodes are required to select OD pairs.")
    points = {node: (float(attrs["x"]), float(attrs["y"])) for node, attrs in nodes.items()}
    adjacency = {node: [] for node in points}
    for edge in edges.values():
        source, target = edge["from"], edge["to"]
        length = dist(points[source], points[target])
        adjacency[source].append((target, length))
        if not edge.get("directed", False):
            adjacency[target].append((source, length))

    scale = sqrt(len(nodes) / BASE_NODE_COUNT)
    domain = tuple((low, low + (high - low) * scale) for low, high in (X_RANGE, Y_RANGE))
    results = {}
    for name, (origin_reference, destination_reference) in LOCATION_PAIRS.items():
        origin_target = target_position(origin_reference, domain, absolute)
        destination_target = target_position(destination_reference, domain, absolute)
        origin, destination = nearest_distinct_pair(points, origin_target, destination_target)
        results[name] = {
            "origin": origin,
            "destination": destination,
            "origin_target_xy": origin_target,
            "destination_target_xy": destination_target,
            "origin_xy": points[origin],
            "destination_xy": points[destination],
            "origin_snap_distance": dist(points[origin], origin_target),
            "destination_snap_distance": dist(points[destination], destination_target),
            "straight_line_distance": dist(points[origin], points[destination]),
            "shortest_path_length": shortest_path_length(adjacency, origin, destination),
        }
    return {
        "node_count": len(nodes),
        "coordinate_mode": "absolute" if absolute else "relative",
        "domain": {"x": domain[0], "y": domain[1]},
        "pairs": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs="?", type=Path,
                        help="Optional folder containing nodes.json and edges.json")
    parser.add_argument("--absolute", action="store_true",
                        help="Interpret LOCATION_PAIRS as absolute x/y coordinates")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Root directory to scan when data_dir is omitted (default: %(default)s)")
    args = parser.parse_args()

    if args.data_dir is None:
        candidates = [
            args.output_dir / f"spatial_corr_scaled_redundant_n{count}" / "v1" / "data"
            for count in NODE_COUNTS
        ]
        folders = [
            folder for folder in candidates
            if all((folder / filename).is_file() for filename in ("nodes.json", "edges.json"))
        ]
        if not folders:
            parser.exit(1, f"No generated graph datasets found under {args.output_dir}.\n")
    else:
        folders = [args.data_dir]

    for folder in folders:
        result = find_pairs(folder, args.absolute)
        path = folder / "od_pairs.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        pairs = ", ".join(f"{name}={pair['origin']}→{pair['destination']}"
                          for name, pair in result["pairs"].items())
        print(f"n={result['node_count']}: {pairs} -> {path}")


if __name__ == "__main__":
    main()
