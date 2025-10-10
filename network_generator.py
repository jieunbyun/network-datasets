# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 11:57:59 2025

@author: caol
"""

# network_generator.py
from __future__ import annotations
import math
import random
from typing import Dict, Tuple, Iterable, Optional, List, Any
import networkx as nx
import matplotlib.pyplot as plt
import cmcrameri.cm as cm
import os
import json as json


# ---------------------------
# Core API
# ---------------------------

def generate_network(
    family: str,
    n: int = 50, # Nodes
    *,
    directed: bool = False,
    seed: Optional[int] = 42,
    # Family-specific knobs (sensible defaults provided)
    p_er: float = 0.05,                # ER: edge prob
    k_ws: int = 4, p_ws: float = 0.1,  # WS: k-nearest neighbors, rewiring prob
    m_ba: int = 2,                     # BA: edges per new node
    deg_sequence: Optional[Iterable[int]] = None,  # Configuration model
    radius_rg: float = 0.2,            # Random geometric radius
    grid_shape: Tuple[int, int] = (7, 7),          # Lattice/grid shape (rows, cols)
    # Edge binary state probabilities (0 = failed, 1 = working)
    edge_survival_p: float = 0.8,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, float]]], nx.Graph]:
    """
    Returns (nodes_dict, edges_dict, probs_dict, G).
    - nodes_dict: {"n1": {"x": ..., "y": ...}, ...}
    - edges_dict: {"e01": {"from": "n1", "to": "n2", "directed": bool, "length": float}, ...}
    - probs_dict: {"e01": {"0": {"p": q}, "1": {"p": p}}, ...}
    - G: the NetworkX graph with node positions in G.nodes[i]["pos"]
    """
    rng = random.Random(seed)
    nx_seed = seed

    fam = family.strip().lower()
    if fam not in {"er", "erdos-renyi", "ws", "watts-strogatz", "ba", "barabasi-albert",
                   "config", "configuration", "rg", "random-geometric", "grid", "lattice"}:
        raise ValueError(f"Unknown family '{family}'. Use one of: ER, WS, BA, Config, RG, Grid/Lattice.")

    # ---------------------------
    # Build topology
    # ---------------------------
    if fam in {"er", "erdos-renyi"}:
        G_base = nx.gnp_random_graph(n, p_er, seed=nx_seed, directed=directed)
    elif fam in {"ws", "watts-strogatz"}:
        # WS requires even k and k < n
        if k_ws >= n:
            k_ws = max(2, min(n - 1 - (n % 2), k_ws))
        if k_ws % 2 == 1:
            k_ws += 1
        G_base = nx.watts_strogatz_graph(n, k_ws, p_ws, seed=nx_seed)
        if directed:
            G_base = G_base.to_directed()
    elif fam in {"ba", "barabasi-albert"}:
        m_ba = max(1, min(m_ba, n - 1))
        # BA is inherently undirected; convert to directed if requested by symmetrizing
        G_base = nx.barabasi_albert_graph(n, m_ba, seed=nx_seed)
        if directed:
            G_base = G_base.to_directed()
    elif fam in {"config", "configuration"}:
        if deg_sequence is None:
            # Make a simple graphical degree sequence (e.g., truncated Poisson-like)
            target_avg = 4
            degs = []
            for _ in range(n):
                k = max(0, int(rng.gauss(target_avg, 1.5)))
                degs.append(k)
            # Ensure even sum for graphicality
            if sum(degs) % 2 == 1:
                degs[0] += 1
            deg_sequence = degs
        G_base = nx.configuration_model(list(deg_sequence), seed=nx_seed, create_using=nx.Graph)
        # Remove parallel edges/self-loops artifacts safely (configuration_model can create them)
        G_base = nx.Graph(G_base)  # drops parallel edges
        G_base.remove_edges_from(nx.selfloop_edges(G_base))
        if directed:
            G_base = G_base.to_directed()
    elif fam in {"rg", "random-geometric"}:
        # Note: RG built undirected; convert if requested
        G_base = nx.random_geometric_graph(n, radius_rg, seed=nx_seed)
        if directed:
            G_base = G_base.to_directed()
    elif fam in {"grid", "lattice"}:
        rows, cols = grid_shape
        G_base = nx.grid_2d_graph(rows, cols, create_using=nx.DiGraph if directed else nx.Graph)
        # Relabel nodes to 0..N-1
        mapping = {xy: i for i, xy in enumerate(G_base.nodes())}
        G_base = nx.relabel_nodes(G_base, mapping)
    else:
        raise RuntimeError("Unreachable")

    # If graph is empty / too sparse, add a tiny nudge: ensure at least one edge if n>1
    if G_base.number_of_nodes() == 0:
        G_base.add_nodes_from(range(n))
    if G_base.number_of_edges() == 0 and G_base.number_of_nodes() > 1:
        # connect a random pair just to avoid empty edge set
        u, v = 0, min(1, G_base.number_of_nodes() - 1)
        G_base.add_edge(u, v)

    # ---------------------------
    # 2D positions per family
    # ---------------------------
    if fam in {"rg", "random-geometric"}:
        # Already has 'pos' from generator
        pos = nx.get_node_attributes(G_base, "pos")
        # Ensure dict covers all nodes
        if len(pos) < G_base.number_of_nodes():
            # fill missing with random
            for i in G_base.nodes():
                pos.setdefault(i, (rng.random(), rng.random()))
    elif fam in {"grid", "lattice"}:
        rows, cols = grid_shape
        # arrange by (row, col) -> normalize to [0,1]
        pos = {}
        for i, (row, col) in enumerate(_grid_coords(rows, cols)):
            # nodes were relabelled 0..N-1 in creation
            pos[i] = (col / max(cols - 1, 1), row / max(rows - 1, 1))
    else:
        # Use spring layout for general cases
        pos = nx.spring_layout(G_base, seed=nx_seed, dim=2)

    nx.set_node_attributes(G_base, pos, "pos")

    # ---------------------------
    # Build dictionaries
    # ---------------------------
    nodes_dict: Dict[str, Dict[str, float]] = {}
    for i in G_base.nodes():
        x, y = G_base.nodes[i]["pos"]
        nodes_dict[f"n{i+1}"] = {"x": float(x), "y": float(y)}

    edges_dict: Dict[str, Dict[str, Any]] = {}
    probs_dict: Dict[str, Dict[str, Dict[str, float]]] = {}

    # Deterministic label order for reproducibility
    edge_list = list(G_base.edges())
    edge_list.sort(key=lambda e: (min(e[0], e[1]), max(e[0], e[1])))

    e_counter = 1
    for u, v in edge_list:
        n_from = f"n{u+1}"
        n_to = f"n{v+1}"
        length = _euclid(nodes_dict[n_from]["x"], nodes_dict[n_from]["y"],
                         nodes_dict[n_to]["x"], nodes_dict[n_to]["y"])

        eid = f"e{e_counter:02d}"
        edges_dict[eid] = {
            "from": n_from,
            "to": n_to,
            "directed": bool(directed),
            "length": float(length),
        }
        # Binary edge state probabilities
        p1 = float(edge_survival_p)
        p0 = float(max(0.0, 1.0 - p1))
        
        # Normalize and round to avoid drift
        s = p0 + p1
        if s == 0:
            p0 = 0.5
            p1 = 0.5
        else:
            p0 /= s
            p1 /= s
        
        probs_dict[eid] = {
            "0": {"p": round(p0, 6)},
            "1": {"p": round(p1, 6)},
        }
        e_counter += 1
        
    # Save to json files
    nodes = nx.number_of_nodes(G_base)
    edges = nx.number_of_edges(G_base)
    JSON_DIR = 'network_' + str(family) + '_nodes' + str(nodes) + '_edges' + str(edges)
    os.makedirs(JSON_DIR, exist_ok = True)
    
    filename = 'probs.json'
    path = os.path.join(JSON_DIR, filename)
    with open(path,'w') as thisdict_file:
        json.dump(probs_dict,thisdict_file)
        
    filename = 'nodes.json'
    path = os.path.join(JSON_DIR, filename)
    with open(path,'w') as thisdict_file:
        json.dump(nodes_dict,thisdict_file)
        
    filename = 'edges.json'
    path = os.path.join(JSON_DIR, filename)
    with open(path,'w') as thisdict_file:
        json.dump(edges_dict,thisdict_file)


    return nodes_dict, edges_dict, probs_dict, G_base


def plot_network(
    G: nx.Graph,
    *,
    node_size: int = 50,
    edge_width: float = 0.5,
    with_labels: bool = False,
    title: Optional[str] = None,
    family: Optional[str] = None
) -> None:
    """Simple matplotlib plot honoring node 'pos'."""
    
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 16
    plt.rc('font', family='times')
    
    pos = nx.get_node_attributes(G, "pos")
    nodes = nx.number_of_nodes(G)
    edges = nx.number_of_edges(G)
    
    if not pos:
        pos = nx.spring_layout(G)
    plt.figure(figsize=(4, 3))
    if G.is_directed == True:
        nx.draw_networkx_edges(G, pos, 
                               width=edge_width, 
                               arrows=G.is_directed(), 
                               arrowstyle='-|>', 
                               arrowsize=12)
    else:
        nx.draw_networkx_edges(G, pos, 
                               width=edge_width)
    nx.draw_networkx_nodes(G, pos, 
                           node_size=node_size, 
                           node_color=[cm.batlowS(4)], 
                           edgecolors='k', 
                           linewidths=0.5)
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=8)
    if title:
        plt.title(title)
    
    plt.axis("off")
    plt.tight_layout()
    FIGURES_DIR = 'network_' + str(family) + '_nodes' + str(nodes) + '_edges' + str(edges)
    os.makedirs(FIGURES_DIR, exist_ok = True)
    filename = 'network_' + str(family) + '_nodes' + str(nodes) + '_edges' + str(edges) + '.pdf'
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path,bbox_inches='tight',dpi=40)
    plt.show()


# ---------------------------
# Helpers
# ---------------------------

def _euclid(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)

def _grid_coords(rows: int, cols: int) -> Iterable[Tuple[int, int]]:
    for r in range(rows):
        for c in range(cols):
            yield (r, c)


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Choose a family: "ER", "WS", "BA", "Config", "RG", "Grid"
    family = "BA"
    nodes, edges, probs, G = generate_network(
        family,
        n=60, # Number of nodes
        directed=False, # Directed edges?
        seed=7, # Random seed
        p_er=0.05,
        k_ws=6, p_ws=0.15,
        m_ba=3,
        radius_rg=0.18,
        grid_shape=(8, 8),
        edge_survival_p=0.8, # Edge probabilities
    )

    # Quick peek
    print("N nodes:", len(nodes))
    print("N edges:", len(edges))
    first_node = next(iter(nodes.items()))
    first_edge = next(iter(edges.items()))
    print("Sample node:", first_node)
    print("Sample edge:", first_edge)
    print("Sample probs for that edge:", probs[first_edge[0]])

    # Plot
    plot_network(G, with_labels=False, family=family)
    
