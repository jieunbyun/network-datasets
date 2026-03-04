from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import networkx as nx
import pytest
import numpy as np

# Import the function under test
from ndtools import fun_binary_graph

# ---------- helpers ----------

def load_dataset_any(data_dir: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load nodes/edges[/probs] from a dataset directory and NORMALISE shapes.

    Accepted node formats:
      1) Dict: { "n1": {...}, "n2": {...}, ... }
      2) List: [ {"id":"n1", ...}, {"id":"n2", ...}, ... ]

    Accepted edge formats:
      1) Dict: { "e1": {"from":"n1","to":"n2", ...}, ... }
      2) List: [ {"eid":"e1","source":"n1","target":"n2", ...}, ... ]

    Returns:
      nodes: {node_id: attrs}
      edges: {eid: {"from": u, "to": v, ...}}
      probs: {} if probs.json missing
    """
    data_dir = Path(data_dir)

    # ---- nodes ----
    nodes_raw = json.loads((data_dir / "nodes.json").read_text(encoding="utf-8"))
    if isinstance(nodes_raw, dict):
        # already {id: attrs}
        nodes = nodes_raw
    elif isinstance(nodes_raw, list):
        # list of {"id": "...", ...}
        nodes = {n["id"]: {k: v for k, v in n.items() if k != "id"} for n in nodes_raw}
    else:
        raise TypeError("nodes.json must be dict or list")

    # ---- edges ----
    edges_raw = json.loads((data_dir / "edges.json").read_text(encoding="utf-8"))
    edges: Dict[str, Any] = {}

    if isinstance(edges_raw, dict):
        # already keyed by eid; ensure 'from'/'to' exist (or map source/target)
        for eid, e in edges_raw.items():
            if "from" in e and "to" in e:
                edges[eid] = e
            elif "source" in e and "target" in e:
                edges[eid] = {"from": e["source"], "to": e["target"],
                              **{k: v for k, v in e.items() if k not in ("source", "target")}}
            else:
                raise KeyError(f"Edge {eid} missing 'from'/'to' or 'source'/'target'")
    elif isinstance(edges_raw, list):
        # list of {"eid": "...", "source": "...", "target": "...", ...}
        for e in edges_raw:
            eid = e.get("eid")
            if not eid:
                raise KeyError("Edge entry in list missing 'eid'")
            if "from" in e and "to" in e:
                edges[eid] = {k: v for k, v in e.items() if k != "eid"}
            elif "source" in e and "target" in e:
                edges[eid] = {"from": e["source"], "to": e["target"],
                              **{k: v for k, v in e.items() if k not in ("eid", "source", "target")}}
            else:
                raise KeyError(f"Edge {eid} missing 'from'/'to' or 'source'/'target'")
    else:
        raise TypeError("edges.json must be dict or list")

    # ---- probs ----
    probs_path = data_dir / "probs.json"
    if probs_path.exists():
        probs = json.loads(probs_path.read_text(encoding="utf-8"))
    else:
        probs = {}

    return nodes, edges, probs

def build_base_graph(nodes: Dict[str, Dict[str, Any]],
                     edges: Dict[str, Dict[str, Any]]) -> nx.Graph:
    G = nx.Graph()
    for nid, attrs in nodes.items():
        G.add_node(nid, **attrs)
    for eid, e in edges.items():
        u, v = e["from"], e["to"]
        attr = {"eid": eid, **{k: v for k, v in e.items() if k not in ("from", "to")}}
        G.add_edge(u, v, **attr)
    return G

# ---------- tests ----------

def test_eval_global_conn_k1():
    nodes, edges, probs = load_dataset_any("datasets/toynet_11edges/v1/data")
    G_base = build_base_graph(nodes, edges)

    comps_st = {eid: 1 for eid in edges}  # all components survive
    k_val, sys_st, _ = fun_binary_graph.eval_global_conn_k(comps_st, G_base)

    assert sys_st == 2, f"Expected system state 2, got '{sys_st}'"
    assert k_val == 2, f"Expected k_val 2, got {k_val}"

def test_eval_global_conn_k2():
    nodes, edges, probs = load_dataset_any("datasets/toynet_11edges/v1/data")
    G_base = build_base_graph(nodes, edges)

    comps_st = {eid: 1 for eid in edges}  
    comps_st['e01'], comps_st['e02'] = 0, 0  # fail two edges
    k_val, sys_st, _ = fun_binary_graph.eval_global_conn_k(comps_st, G_base)

    assert sys_st == 1, f"Expected system state 1, got '{sys_st}'"
    assert k_val == 1, f"Expected k_val 1, got {k_val}"

def test_eval_global_conn_k3():
    nodes, edges, probs = load_dataset_any("datasets/toynet_11edges/v1/data")
    G_base = build_base_graph(nodes, edges)

    comps_st = {eid: 1 for eid in edges}  
    comps_st['e01'], comps_st['e02'], comps_st['e03'] = 0, 0, 0  # fail three edges
    k_val, sys_st, _ = fun_binary_graph.eval_global_conn_k(comps_st, G_base)

    assert sys_st == 0, f"Expected system state 0, got '{sys_st}'"
    assert k_val == 0, f"Expected k_val 0, got {k_val}"

def test_eval_global_conn_k4():
    nodes, edges, probs = load_dataset_any("datasets/toynet_11edges/v1/data")
    G_base = build_base_graph(nodes, edges)

    comps_st = {eid: 1 for eid in edges}  
    comps_st['e05'], comps_st['e06'] = 0, 0  # fail two edges
    k_val, sys_st, _ = fun_binary_graph.eval_global_conn_k(comps_st, G_base)

    assert sys_st == 1, f"Expected system state 1, got '{sys_st}'"
    assert k_val == 1, f"Expected k_val 1, got {k_val}"

def test_eval_travel_time_to_nearest1():
    nodes, edges, probs = load_dataset_any("datasets/toynet_11edges/v1/data")
    G_base = build_base_graph(nodes, edges)

    comps_st = {eid: 1 for eid in edges}  # all components survive
    origin = 'n1'
    destinations = ['n5', 'n7']

    travel_time, sys_st, details = fun_binary_graph.eval_travel_time_to_nearest(
        comps_st, G_base, origin, destinations, avg_speed=1.0,
        target_max=0.5, #  it shouldn't take longer than 0.5 hours more than original shortest time
        length_attr="length"
    )

    assert np.isclose(travel_time, np.sqrt(2) + 1.0), f"Expected travel_time {np.sqrt(2) + 1.0}, got {travel_time}"
    assert sys_st == 1, f"Expected system state 1, got '{sys_st}'"

def test_eval_travel_time_to_nearest2():
    nodes, edges, probs = load_dataset_any("datasets/toynet_11edges/v1/data")
    G_base = build_base_graph(nodes, edges)

    comps_st = {eid: 1 for eid in edges}  
    comps_st['e03'] = 0  # failed components

    origin = 'n1'
    destinations = ['n5', 'n7']

    travel_time, sys_st, details = fun_binary_graph.eval_travel_time_to_nearest(
        comps_st, G_base, origin, destinations, avg_speed=1.0,
        target_max=0.5, #  it shouldn't take longer than 0.5 hours more than original shortest time
        length_attr="length"
    )

    assert np.isclose(travel_time, 2*np.sqrt(2) + 1.0), f"Expected travel_time {2*np.sqrt(2) + 1.0}, got {travel_time}"
    assert sys_st == 0, f"Expected system state 0, got '{sys_st}'"

def test_eval_travel_time_to_nearest3():
    nodes, edges, probs = load_dataset_any("datasets/toynet_11edges/v1/data")
    G_base = build_base_graph(nodes, edges)

    comps_st = {eid: 1 for eid in edges}  
    comps_st['e03'] = 0  # failed components

    origin = 'n1'
    destinations = ['n5', 'n7']

    travel_time, sys_st, details = fun_binary_graph.eval_travel_time_to_nearest(
        comps_st, G_base, origin, destinations, avg_speed=1.0,
        target_max=1.5, #  it shouldn't take longer than 0.5 hours more than original shortest time
        length_attr="length"
    )

    assert np.isclose(travel_time, 2*np.sqrt(2) + 1.0), f"Expected travel_time {2*np.sqrt(2) + 1.0}, got {travel_time}"
    assert sys_st == 1, f"Expected system state 1, got '{sys_st}'"

def test_eval_travel_time_to_nearest4():
    nodes, edges, probs = load_dataset_any("datasets/toynet_11edges/v1/data")
    G_base = build_base_graph(nodes, edges)

    comps_st = {eid: 1 for eid in edges}  
    comps_st['e01'], comps_st['e02'], comps_st['e03'] = 0, 0, 0  # failed components

    origin = 'n1'
    destinations = ['n5', 'n7']

    travel_time, sys_st, details = fun_binary_graph.eval_travel_time_to_nearest(
        comps_st, G_base, origin, destinations, avg_speed=1.0,
        target_max=1.5, #  it shouldn't take longer than 0.5 hours more than original shortest time
        length_attr="length"
    )

    assert travel_time==None, f"Expected travel_time None, got {travel_time}"
    assert sys_st == 0, f"Expected system state 0, got '{sys_st}'"

def test_eval_travel_time_to_nearest5():
    nodes, edges, probs = load_dataset_any("datasets/toynet_11edges/v1/data")
    G_base = build_base_graph(nodes, edges)

    comps_st = {eid: 1 for eid in edges}  
    comps_st['e03'] = 0  # failed components

    origin = 'n1'
    destinations = ['n5', 'n7']

    travel_time, sys_st, details = fun_binary_graph.eval_travel_time_to_nearest(
        comps_st, G_base, origin, destinations, avg_speed=1.0,
        target_max=[3.0, 1.5], # > 3.0 hours = state 0, >1.5 hours = state 1, else state 2
        length_attr="length"
    )

    assert np.isclose(travel_time, 2*np.sqrt(2) + 1.0), f"Expected travel_time {2*np.sqrt(2) + 1.0}, got {travel_time}"
    assert sys_st == 2, f"Expected system state 2, got '{sys_st}'"

def test_eval_population_accessibility1():
    nodes, edges, probs = load_dataset_any("datasets/toynet_11edges/v1/data")
    G_base = build_base_graph(nodes, edges)

    comps_st = {eid: 1 for eid in edges}  # all components survive
    destinations = ['n5', 'n7']

    connected_ratio, sys_st, details = fun_binary_graph.eval_population_accessibility(
        comps_st, G_base, destinations, avg_speed=1.0,
        target_time_max=0.5, #  it shouldn't take longer than 0.5 hours more than original shortest time
        length_attr="length",
        target_pop_max=0.7, # at least 70% of population should reach destination within target time
        population_attr="population"
    )

    assert np.isclose(connected_ratio, 1.0), f"Expected connected_ratio 1.0, got {connected_ratio}"
    assert sys_st == 1, f"Expected system state 1, got '{sys_st}'"

def test_eval_population_accessibility2():
    nodes, edges, probs = load_dataset_any("datasets/toynet_11edges/v1/data")
    G_base = build_base_graph(nodes, edges)

    comps_st = {eid: 1 for eid in edges}  
    comps_st['e03'] = 0  # failed components

    destinations = ['n5', 'n7']

    connected_ratio, sys_st, details = fun_binary_graph.eval_population_accessibility(
        comps_st, G_base, destinations, avg_speed=1.0,
        target_time_max=0.5, #  it shouldn't take longer than 0.5 hours more than original shortest time
        length_attr="length",
        target_pop_max=0.7, # at least 70% of population should reach destination within target time
        population_attr="population"
    )

    assert np.isclose(connected_ratio, (110.0-10.0)/110.0), f"Expected connected_ratio {(110.0-10.0)/110.0}, got {connected_ratio}"
    assert sys_st == 1, f"Expected system state 1, got '{sys_st}'"

def test_eval_population_accessibility3():
    nodes, edges, probs = load_dataset_any("datasets/toynet_11edges/v1/data")
    G_base = build_base_graph(nodes, edges)

    comps_st = {eid: 1 for eid in edges}  
    comps_st['e05'], comps_st['e07'] = 0, 0  # failed components

    destinations = ['n5', 'n7']

    connected_ratio, sys_st, details = fun_binary_graph.eval_population_accessibility(
        comps_st, G_base, destinations, avg_speed=1.0,
        target_time_max=0.5, #  it shouldn't take longer than 0.5 hours more than original shortest time
        target_pop_max=0.7, # at least 70% of population should reach destination within target time
    )

    expected = (110.0 - 10.0 - 20.0 - 20.0 - 20.0)/110.0 # disconnected: n1-n4, connected: n6, n8
    assert np.isclose(connected_ratio, expected), f"Expected connected_ratio {expected}, got {connected_ratio}"
    assert sys_st == 0, f"Expected system state 0, got '{sys_st}'"
    
    for n in ['n1', 'n2', 'n3', 'n4']:
        assert details['node_details'][n]['within_threshold'] == False, f"Expected node {n} to be unreachable"
    for n in ['n6', 'n8']:
        assert details['node_details'][n]['within_threshold'] == True, f"Expected node {n} to be reachable"
    for n in ['n5', 'n7']:
        assert n not in details['node_details'], f"Expected destination node {n} to be skipped"

def test_eval_population_accessibility4():
    # multi-state system; rest remains the same as test_eval_population_accessibility3

    nodes, edges, probs = load_dataset_any("datasets/toynet_11edges/v1/data")
    G_base = build_base_graph(nodes, edges)

    comps_st = {eid: 1 for eid in edges}  
    comps_st['e05'], comps_st['e07'] = 0, 0  # failed components

    destinations = ['n5', 'n7']

    connected_ratio, sys_st, details = fun_binary_graph.eval_population_accessibility(
        comps_st, G_base, destinations, avg_speed=1.0,
        target_time_max=0.5, #  it shouldn't take longer than 0.5 hours more than original shortest time
        target_pop_max=[0.3, 0.7], # system state 0: if less than 30% population reaches destination within target time
        # state 1: if 30%<= <70% population reaches destination within target time; else state 2
    )

    expected = (110.0 - 10.0 - 20.0 - 20.0 - 20.0)/110.0 # disconnected: n1-n4, connected: n6, n8
    assert np.isclose(connected_ratio, expected), f"Expected connected_ratio {expected}, got {connected_ratio}"
    assert sys_st == 1, f"Expected system state 1, got '{sys_st}'"

def test_eval_population_accessibility5():
    # another check for multi-state systems; rest remains the same as test_eval_population_accessibility1
    nodes, edges, probs = load_dataset_any("datasets/toynet_11edges/v1/data")
    G_base = build_base_graph(nodes, edges)

    comps_st = {eid: 1 for eid in edges}  # all components survive
    destinations = ['n5', 'n7']

    connected_ratio, sys_st, details = fun_binary_graph.eval_population_accessibility(
        comps_st, G_base, destinations, avg_speed=1.0,
        target_time_max=0.5, #  it shouldn't take longer than 0.5 hours more than original shortest time
        length_attr="length",
        target_pop_max=[0.7, 1.0], # at least 70% of population should reach destination within target time
        population_attr="population"
    )

    assert np.isclose(connected_ratio, 1.0), f"Expected connected_ratio 1.0, got {connected_ratio}"
    assert sys_st == 2, f"Expected system state 2, got '{sys_st}'"

def test_eval_1od_connectivity1():
    nodes, edges, probs = load_dataset_any("datasets/toynet_11edges/v1/data")
    G_base = build_base_graph(nodes, edges)

    comps_st = {eid: 1 for eid in edges}  # all components survive
    origin = 'n1'
    destinations = 'n8'

    _, sys_st, info = fun_binary_graph.eval_1od_connectivity(
        comps_st, G_base, origin, destinations) 

    path_nodes_cands = [['n1', 'n2', 'n6', 'n8'],
                        ['n1', 'n4', 'n6', 'n8'],
                        ['n1', 'n4', 'n7', 'n8']]
    path_eids_cands = [['e01', 'e05', 'e10'],
                       ['e03', 'e06', 'e10'],
                       ['e03', 'e07', 'e11']]

    assert sys_st == 1, f"Expected system state 1, got '{sys_st}'"
    assert info['path_nodes'] in path_nodes_cands, f"Got {info['path_nodes']}"
    assert info['path_edge_ids'] in path_eids_cands, f"Got {info['path_edge_ids']}"

def test_eval_1od_connectivity2():
    nodes, edges, probs = load_dataset_any("datasets/toynet_11edges/v1/data")
    G_base = build_base_graph(nodes, edges)

    comps_st = {eid: 1 for eid in edges}  
    comps_st['e03'] = 0  # failed components

    origin = 'n1'
    destinations = 'n8'

    _, sys_st, info = fun_binary_graph.eval_1od_connectivity(
        comps_st, G_base, origin, destinations
    )

    path_nodes_expected = ['n1', 'n2', 'n6', 'n8']
    path_eids_expected = ['e01', 'e05', 'e10']

    assert sys_st == 1, f"Expected system state 1, got '{sys_st}'"
    assert info['path_nodes'] == path_nodes_expected, f"Expected path_nodes {path_nodes_expected}, got {info['path_nodes']}"
    assert info['path_edge_ids'] == path_eids_expected, f"Expected path_edge_ids {path_eids_expected}, got {info['path_edge_ids']}"

def test_eval_1od_connectivity3():
    nodes, edges, probs = load_dataset_any("datasets/toynet_11edges/v1/data")
    G_base = build_base_graph(nodes, edges)

    comps_st = {eid: 1 for eid in edges}  
    comps_st['e01'], comps_st['e02'], comps_st['e03'] = 0, 0, 0  # failed components

    origin = 'n1'
    destinations = 'n8'

    _, sys_st, info = fun_binary_graph.eval_1od_connectivity(
        comps_st, G_base, origin, destinations
    )

    assert sys_st == 0, f"Expected system state 1, got '{sys_st}'"
    assert info['path_nodes'] == None, f"Expected path_nodes None, got {info['path_nodes']}"
    assert info['path_edge_ids'] == None, f"Expected path_edge_ids None, got {info['path_edge_ids']}"