import networkx as nx
from typing import Dict, Any, Optional

import json
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import math

def build_graph(
    nodes: Dict[str, Dict[str, Any]],
    edges: Dict[str, Dict[str, Any]],
    probs: Optional[Dict[str, Any]] = None,
) -> nx.Graph:
    G = nx.Graph()
    for nid, attrs in nodes.items():
        G.add_node(nid, **attrs)
    for eid, e in edges.items():
        u, v = e["from"], e["to"]
        attr = {"eid": eid, **{k: v for k, v in e.items() if k not in ("from","to")}}
        if probs is not None:
            attr["p_active"] = probs.get(eid, {}).get("1", {}).get("p", None)
        G.add_edge(u, v, **attr)
    return G

def draw_graph_from_data(
    data_dir: str | Path,
    *,
    layout: str = "spring",
    node_color: str = "skyblue",
    node_size: int = 500,
    edge_color: str = "gray",
    with_node_labels: bool = True,
    with_edge_labels: bool = False,
    title: Optional[str] = None,
    layout_kwargs: Optional[Dict[str, Any]] = None,
    output_name: str = "graph.png",
) -> Path:
    """
    Load nodes/edges from JSON files in `data_dir`, draw the graph, and save to the same dir.

    Expects (preferred, current repo format):
        - nodes.json : {"n0": {"x": null, "y": null, ...}, ...}
        - edges.json : [{"id": "e0","from":"n0","to":"n1",...}, ...]
                       or {"e0":{"from":"n0","to":"n1",...}, ...}

    Auto-chooses a layout if x/y are missing or null on any node.
    """
    def _is_number(x) -> bool:
        try:
            return isinstance(x, (int, float)) and not math.isnan(float(x))
        except Exception:
            return False

    def _extract_positions(G: nx.Graph, x_key="x", y_key="y") -> Dict[Any, tuple]:
        """Return {node: (x,y)} only if ALL nodes have numeric x,y."""
        pos = {}
        for n, d in G.nodes(data=True):
            x = d.get(x_key, None)
            y = d.get(y_key, None)
            if _is_number(x) and _is_number(y):
                pos[n] = (float(x), float(y))
            else:
                # As soon as we see an invalid coord, bail to force auto-layout
                return {}
        return pos

    def _normalize_edges(edges_raw):
        """Return (edge_list, is_directed) from dict or list edges."""
        if isinstance(edges_raw, dict):
            items = list(edges_raw.values())
        elif isinstance(edges_raw, list):
            items = edges_raw
        else:
            raise TypeError(f"edges.json must be list or dict, got {type(edges_raw)}")

        is_directed = False
        if items:
            is_directed = bool(items[0].get("directed", False))
        edge_list = []
        for ev in items:
            u, v = ev.get("from"), ev.get("to")
            if u is None or v is None:
                raise ValueError("Each edge must have 'from' and 'to' keys.")
            attrs = {k: val for k, val in ev.items() if k not in ("from", "to")}
            edge_list.append((u, v, attrs))
        return edge_list, is_directed

    data_dir = Path(data_dir)
    layout_kwargs = layout_kwargs or {}

    # --- Load nodes & edges ---
    nodes_path = data_dir / "nodes.json"
    edges_path = data_dir / "edges.json"

    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)  # expect dict {node_id: {x,y,...}}

    with open(edges_path, "r", encoding="utf-8") as f:
        edges_raw = json.load(f)  # list[dict] or dict[str, dict]

    # --- Build graph ---
    edge_list, is_directed = _normalize_edges(edges_raw)
    G = nx.DiGraph() if is_directed else nx.Graph()

    if isinstance(nodes, dict):
        for nid, nv in nodes.items():
            G.add_node(nid, **(nv if isinstance(nv, dict) else {}))
    else:
        # fallback if nodes came as a list of {"id":..., ...}
        for nd in nodes:
            nid = nd.get("id")
            if nid is None:
                raise ValueError("Node entries must have an 'id' when given as a list.")
            attrs = {k: v for k, v in nd.items() if k != "id"}
            G.add_node(nid, **attrs)

    for u, v, attrs in edge_list:
        G.add_edge(u, v, **attrs)

    # --- Determine positions ---
    pos = _extract_positions(G)  # only returns non-empty if ALL nodes have numeric x,y
    if not pos:
        # Fallback to algorithmic layout
        if layout == "spring":
            pos = nx.spring_layout(G, **layout_kwargs)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G, **layout_kwargs)
        elif layout == "circular":
            pos = nx.circular_layout(G, **layout_kwargs)
        elif layout == "shell":
            pos = nx.shell_layout(G, **layout_kwargs)
        else:
            raise ValueError(f"Unknown layout: {layout}")

    # --- Draw ---
    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        node_color=node_color,
        node_size=node_size,
        edge_color=edge_color,
        with_labels=with_node_labels,
        font_size=9,
        font_color="black",
    )

    if with_edge_labels:
        edge_labels = {(u, v): d.get("eid", "") for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    if title:
        plt.title(title)

    # --- Save ---
    out_path = data_dir / output_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path

def compute_edge_lengths(nodes_dict, edges_dict):
    """
    Compute Euclidean length in km for each edge.
      nodes_dict: {node_id: {"x": float(km), "y": float(km)}}
      edges_dict: {edge_id: {"from": str, "to": str, "directed": bool}}
    Returns:
      {edge_id: float}
    """
    lengths = {}
    for eid, e in edges_dict.items():
        u, v = e["from"], e["to"]
        x1, y1 = nodes_dict[u]["x"], nodes_dict[u]["y"]
        x2, y2 = nodes_dict[v]["x"], nodes_dict[v]["y"]
        lengths[eid] = math.hypot(x2 - x1, y2 - y1)
    return lengths

if __name__ == "__main__":
    output_path = draw_graph_from_data("datasets/toynet_11edges/v1/data")
