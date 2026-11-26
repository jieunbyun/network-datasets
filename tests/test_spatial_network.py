from __future__ import annotations

import json
from math import hypot
from pathlib import Path

import networkx as nx
import pytest

from ndtools import network_generator as ng


def test_spatial_network(tmp_path: Path):
    n_nodes = 20
    p_fail = 0.2
    cfg = ng.GenConfig(
        name="spatial_n20",
        version="v1",
        generator="spatial_network",
        description="test spatial network",
        generator_params={"n_nodes": n_nodes, "p_fail": p_fail},
        seed=7,
    )
    schema_dir = Path(__file__).resolve().parents[1] / "schema"
    ds_root = ng.generate_and_save(
        tmp_path / "generated", config=cfg, schema_dir=schema_dir, draw_graph=False
    )
    assert (ds_root / "README.md").is_file()
    assert (ds_root / "metadata.json").is_file()

    data_dir = ds_root / "data"
    nodes = json.loads((data_dir / "nodes.json").read_text(encoding="utf-8"))
    edges = json.loads((data_dir / "edges.json").read_text(encoding="utf-8"))
    probs = json.loads((data_dir / "probs.json").read_text(encoding="utf-8"))
    od = json.loads((data_dir / "od_pairs.json").read_text(encoding="utf-8"))

    assert len(nodes) == n_nodes
    assert set(probs) == set(edges)
    for states in probs.values():
        assert states["0"]["p"] == pytest.approx(p_fail)
        assert states["1"]["p"] == pytest.approx(1 - p_fail)

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for edge in edges.values():
        origin, destination = edge["from"], edge["to"]
        assert origin in nodes and destination in nodes and origin != destination
        assert edge["directed"] is False
        length = hypot(
            nodes[origin]["x"] - nodes[destination]["x"],
            nodes[origin]["y"] - nodes[destination]["y"],
        )
        graph.add_edge(origin, destination, length=length)
    assert nx.is_connected(graph)
    assert not list(nx.bridges(graph))

    assert od["node_count"] == n_nodes
    assert od["coordinate_mode"] == "relative"
    assert set(od["pairs"]) == {"west_east", "south_north", "southwest_northeast"}
    for pair in od["pairs"].values():
        origin, destination = pair["origin"], pair["destination"]
        assert origin in nodes and destination in nodes and origin != destination
        expected = nx.shortest_path_length(graph, origin, destination, weight="length")
        assert pair["shortest_path_length"] == pytest.approx(expected)
