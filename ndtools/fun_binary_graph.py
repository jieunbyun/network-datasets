from typing import Dict, Tuple, Any, Iterable, Optional, List
import networkx as nx

def _pairwise(seq: List[str]):
    for i in range(len(seq) - 1):
        yield seq[i], seq[i + 1]

def _edge_ids_on_path(G: nx.Graph, node_path: Optional[List[str]]) -> Optional[List[Optional[Any]]]:
    """Return [eid1, eid2, ...] along the node_path; None if no path or eid missing."""
    if not node_path:
        return None
    is_multi = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    eids: List[Optional[Any]] = []
    for u, v in _pairwise(node_path):
        eid = None
        if is_multi:
            data_dict = G.get_edge_data(u, v)  # {key: attrdict}
            if data_dict:
                _, d = next(iter(data_dict.items()))
                eid = d.get("eid")
        else:
            d = G.get_edge_data(u, v)
            if d:
                eid = d.get("eid")
        eids.append(eid)
    return eids

def _node_edge_chain(G: nx.Graph, node_path: Optional[List[str]]) -> Optional[List[Any]]:
    """['n1','n3','n6'] -> ['n1', eid(n1,n3), 'n3', eid(n3,n6), 'n6'] (uses 'eid', falls back to 'u->v')."""
    if not node_path:
        return None
    chain: List[Any] = [node_path[0]]
    is_multi = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    for u, v in _pairwise(node_path):
        eid = None
        if is_multi:
            data_dict = G.get_edge_data(u, v)
            if data_dict:
                _, d = next(iter(data_dict.items()))
                eid = d.get("eid")
        else:
            d = G.get_edge_data(u, v)
            if d:
                eid = d.get("eid")
        chain.append(eid if eid is not None else f"{u}->{v}")
        chain.append(v)
    return chain



def eval_global_conn_k(
    comps_state: Dict[str, int],
    G_base: nx.Graph
) -> Tuple[int, str, None]:
    """
    Build subgraph H from G_base according to component states:
      - If comps_state[node_id] == 0: remove all edges incident to that node.
      - Only include edges whose comps_state[eid] == 1.
      - Nodes remain present; connectivity is determined by remaining edges.

    Returns:
        (k_value, k_value, None)
    """
    # Collect node-off set and the set of edge IDs marked on/off
    node_off = {cid for cid, st in comps_state.items() if st == 0 and cid in G_base.nodes}
    edge_on  = {cid for cid, st in comps_state.items() if st == 1}  # keep only these edges
    # (unknown IDs in comps_state are ignored)

    # Build subgraph
    H = nx.Graph()
    H.add_nodes_from(G_base.nodes(data=True))

    for u, v, data in G_base.edges(data=True):
        eid = data.get("eid")
        # skip if incident to a node that's off
        if u in node_off or v in node_off:
            continue
        # include only edges explicitly set to 1
        if eid is not None and eid in edge_on:
            H.add_edge(u, v, **data)

    # Compute global vertex connectivity
    k_val = nx.node_connectivity(H) if H.number_of_nodes() > 1 else 0
    return k_val, k_val, None

from typing import Dict, Tuple, Any, Iterable, Optional, List
import networkx as nx

def _pairwise(seq: List[str]):
    for i in range(len(seq) - 1):
        yield seq[i], seq[i + 1]

def _edge_ids_on_path(G: nx.Graph, node_path: Optional[List[str]]) -> Optional[List[Optional[Any]]]:
    """Return [eid1, eid2, ...] along the node_path; None if no path or eid missing."""
    if not node_path:
        return None
    is_multi = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    eids: List[Optional[Any]] = []
    for u, v in _pairwise(node_path):
        eid = None
        if is_multi:
            data_dict = G.get_edge_data(u, v)  # {key: attrdict}
            if data_dict:
                _, d = next(iter(data_dict.items()))
                eid = d.get("eid")
        else:
            d = G.get_edge_data(u, v)
            if d:
                eid = d.get("eid")
        eids.append(eid)
    return eids

def _node_edge_chain(G: nx.Graph, node_path: Optional[List[str]]) -> Optional[List[Any]]:
    """['n1','n3','n6'] -> ['n1', eid(n1,n3), 'n3', eid(n3,n6), 'n6'] (uses 'eid', falls back to 'u->v')."""
    if not node_path:
        return None
    chain: List[Any] = [node_path[0]]
    is_multi = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    for u, v in _pairwise(node_path):
        eid = None
        if is_multi:
            data_dict = G.get_edge_data(u, v)
            if data_dict:
                _, d = next(iter(data_dict.items()))
                eid = d.get("eid")
        else:
            d = G.get_edge_data(u, v)
            if d:
                eid = d.get("eid")
        chain.append(eid if eid is not None else f"{u}->{v}")
        chain.append(v)
    return chain

def eval_travel_time_to_nearest(
    comps_state: Dict[str, int],
    G_base: nx.Graph,
    origin: str,
    destinations: Iterable[str],
    *,
    avg_speed: float = 60.0,        # distance units per hour (e.g., km/h)
    target_max: float | list = 0.5,        # allowed extra time over baseline, in HOURS
    length_attr: str = "length",    # edge length attribute (e.g., km)
) -> Tuple[Optional[float], int, Dict[str, Any]]:
    """
    Evalute the shortest time from a specified origin to the nearest destination node
    under binary component states.

    A node is considered REACHABLE if:
        t_filtered(origin) <= t_baseline(origin) + target_max
    System state:
        1 if reachable else 0
    Returns:
        travel_time_filtered (float or None), system state (int), info (diagnostics dictionary)
    """
    dest_set = set(destinations)
    if not dest_set:
        return None, 0, {"reason": "no destinations provided"}

    # ----- Baseline graph (all edges that have length_attr) -----
    Hb = G_base.__class__()
    Hb.add_nodes_from(G_base.nodes(data=True))
    for u, v, data in G_base.edges(data=True):
        if length_attr in data and data[length_attr] is not None:
            Hb.add_edge(u, v, **data)

    if not Hb.has_node(origin):
        return None, 0, {"reason": "origin_missing_in_baseline"}

    cand_b = [d for d in dest_set if Hb.has_node(d)]
    if not cand_b:
        return None, 0, {"reason": "no_destinations_in_baseline"}

    try:
        dist_b_map, paths_b = nx.single_source_dijkstra(Hb, source=origin, weight=length_attr)
    except nx.NetworkXNoPath:
        return None, 0, {"reason": "no_baseline_path"}

    reach_b = [(d, dist_b_map[d]) for d in cand_b if d in dist_b_map]
    if not reach_b:
        return None, 0, {"reason": "no_baseline_destination_reachable"}

    dest_b, dist_b = min(reach_b, key=lambda x: x[1])
    time_b = dist_b / float(avg_speed)  # hours
    path_b_nodes = paths_b.get(dest_b)
    path_b_chain = _node_edge_chain(Hb, path_b_nodes)
    path_b_edges = _edge_ids_on_path(Hb, path_b_nodes)

    # ----- Filtered graph (apply comps_state) -----
    node_off = {cid for cid, st in comps_state.items() if st == 0 and cid in G_base.nodes}
    edge_on  = {cid for cid, st in comps_state.items() if st == 1}

    if origin in node_off:
        return None, 0, {
            "reason": "origin_off",
            "baseline_time_hours": time_b,
            "baseline_path_nodes": path_b_nodes,
            "baseline_path_edges": path_b_edges,
            "baseline_path_chain": path_b_chain,
        }

    H = G_base.__class__()
    H.add_nodes_from(G_base.nodes(data=True))
    for u, v, data in G_base.edges(data=True):
        if u in node_off or v in node_off:
            continue
        eid = data.get("eid")
        if eid is not None and eid in edge_on:
            if length_attr in data and data[length_attr] is not None:
                H.add_edge(u, v, **data)

    if not H.has_node(origin):
        return None, 0, {
            "reason": "origin_missing_in_filtered",
            "baseline_time_hours": time_b,
            "baseline_path_nodes": path_b_nodes,
            "baseline_path_edges": path_b_edges,
            "baseline_path_chain": path_b_chain,
        }

    cand_f = [d for d in dest_set if H.has_node(d) and d not in node_off]
    if not cand_f:
        return None, 0, {
            "reason": "no_destinations_in_filtered",
            "baseline_time_hours": time_b,
            "baseline_path_nodes": path_b_nodes,
            "baseline_path_edges": path_b_edges,
            "baseline_path_chain": path_b_chain,
        }

    try:
        dist_f_map, paths_f = nx.single_source_dijkstra(H, source=origin, weight=length_attr)
    except nx.NetworkXNoPath:
        return None, 0, {
            "reason": "no_path_filtered",
            "baseline_time_hours": time_b,
            "baseline_path_nodes": path_b_nodes,
            "baseline_path_edges": path_b_edges,
            "baseline_path_chain": path_b_chain,
        }

    reach_f = [(d, dist_f_map[d]) for d in cand_f if d in dist_f_map]
    if not reach_f:
        return None, 0, {
            "reason": "no_destination_reachable_filtered",
            "baseline_time_hours": time_b,
            "baseline_path_nodes": path_b_nodes,
            "baseline_path_edges": path_b_edges,
            "baseline_path_chain": path_b_chain,
        }

    dest_f, dist_f = min(reach_f, key=lambda x: x[1])
    time_f = dist_f / float(avg_speed)  
    path_f_nodes = paths_f.get(dest_f)
    path_f_chain = _node_edge_chain(H, path_f_nodes)
    path_f_edges = _edge_ids_on_path(H, path_f_nodes)

    # ----- Threshold check -----
    if isinstance(target_max, (int, float)):
        target_max = [target_max]
    time_threshold = [time_b + float(tm) for tm in target_max]

    sys_st = next((i for i, t in enumerate(time_threshold) if t < time_f), len(time_threshold))

    info = {
        # filtered
        "dest_reached": dest_f,
        "dist_filtered": dist_f,
        "time_filtered_hours": time_f,
        "path_filtered_nodes": path_f_nodes,
        "path_filtered_edges": path_f_edges,  
        "path_filtered_chain": path_f_chain,
        # baseline
        "baseline_dest": dest_b,
        "baseline_dist": dist_b,
        "baseline_time_hours": time_b,
        "baseline_path_nodes": path_b_nodes,
        "baseline_path_edges": path_b_edges,   
        "baseline_path_chain": path_b_chain,
        # params
        "avg_speed_per_hour": avg_speed,
        "allowed_extra_time_hours": target_max,
        "time_threshold_hours": time_threshold,
        "reached_any": True,
    }
    return time_f, sys_st, info


def eval_population_accessibility(
    comps_state: Dict[str, int],
    G_base: nx.Graph,
    destinations: Iterable[str],
    *,
    target_time_max: float = 0.5,   # allowed extra time over baseline [hours]
    avg_speed: float = 60.0,        # distance units per hour (e.g. km/h)
    length_attr: str = "length",
    target_pop_max: float | list = 0.7,
    population_attr: str = "population",
) -> Tuple[Optional[float], int, Dict[str, Any]]:
    """
    Evaluate population accessibility to destination nodes under binary component states.

    A node is considered CONNECTED if:
        t_filtered(node) <= t_baseline(node) + target_time_max

    System state:
        1 if (connected_population / total_population) >= target_pop_max
        0 otherwise

    Returns:
        connected_population_ratio (float or None),
        system_state (1 or 0),
        info (diagnostics dictionary)
    """

    dest_set = set(destinations)
    if not dest_set:
        return None, 0, {"reason": "no_destinations_provided"}

    # --------------------------------------------------
    # Baseline graph (ignore component states)
    # --------------------------------------------------
    Hb = G_base.__class__()
    Hb.add_nodes_from(G_base.nodes(data=True))
    for u, v, data in G_base.edges(data=True):
        if length_attr in data and data[length_attr] is not None:
            Hb.add_edge(u, v, **data)

    cand_dest_b = [d for d in dest_set if Hb.has_node(d)]
    if not cand_dest_b:
        return None, 0, {"reason": "no_destinations_in_baseline"}

    # Reverse Dijkstra: shortest distance TO nearest destination
    dist_b = {}
    for d in cand_dest_b:
        try:
            dmap = nx.single_source_dijkstra_path_length(
                Hb, d, weight=length_attr
            )
            for n, val in dmap.items():
                dist_b[n] = min(dist_b.get(n, float("inf")), val)
        except nx.NetworkXNoPath:
            continue

    # --------------------------------------------------
    # Filtered graph (apply binary component states)
    # --------------------------------------------------
    node_off = {
        cid for cid, st in comps_state.items()
        if st == 0 and cid in G_base.nodes
    }
    edge_on = {
        cid for cid, st in comps_state.items() if st == 1
    }

    H = G_base.__class__()
    H.add_nodes_from(G_base.nodes(data=True))
    for u, v, data in G_base.edges(data=True):
        if u in node_off or v in node_off:
            continue
        eid = data.get("eid")
        if eid is not None and eid in edge_on:
            if length_attr in data and data[length_attr] is not None:
                H.add_edge(u, v, **data)

    cand_dest_f = [d for d in dest_set if H.has_node(d) and d not in node_off]
    if not cand_dest_f:
        return None, 0, {"reason": "no_destinations_in_filtered"}

    # Reverse Dijkstra on filtered graph
    dist_f = {}
    for d in cand_dest_f:
        try:
            dmap = nx.single_source_dijkstra_path_length(
                H, d, weight=length_attr
            )
            for n, val in dmap.items():
                dist_f[n] = min(dist_f.get(n, float("inf")), val)
        except nx.NetworkXNoPath:
            continue

    # --------------------------------------------------
    # Population accounting
    # --------------------------------------------------
    total_pop = 0.0
    connected_pop = 0.0
    node_details = {}

    for n, data in G_base.nodes(data=True):
        if n in dest_set:
            continue  # skip destination nodes
        pop = float(data.get(population_attr, 0.0))
        total_pop += pop

        if n in node_off:
            node_details[n] = {
                "population": pop,
                "reachable": False,
                "reason": "node_off",
            }
            continue

        if n not in dist_f:
            node_details[n] = {
                "population": pop,
                "reachable": False,
                "reason": "no_path_filtered",
            }
            continue

        time_f = dist_f[n] / avg_speed

        if n not in dist_b:
            ok = False
            time_b = None
        else:
            time_b = dist_b[n] / avg_speed
            ok = time_f <= (time_b + target_time_max)

        if ok:
            connected_pop += pop

        node_details[n] = {
            "population": pop,
            "travel_time_filtered_hours": time_f,
            "travel_time_baseline_hours": time_b,
            "extra_time_hours": None if time_b is None else time_f - time_b,
            "within_threshold": ok,
        }

    if total_pop <= 0.0:
        return None, 0, {"reason": "total_population_zero"}

    connected_ratio = connected_pop / total_pop

    # ----- Threshold check (scalar or multi-level) -----
    if isinstance(target_pop_max, float):
        target_pop_max = [target_pop_max]

    # ensure sorted thresholds (important)
    target_pop_max = sorted(float(p) for p in target_pop_max)

    system_state = next(
        (i for i, p in enumerate(target_pop_max) if connected_ratio < p),
        len(target_pop_max)
    )

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------
    time_f = {n: d / avg_speed for n, d in dist_f.items()}
    time_b = {n: d / avg_speed for n, d in dist_b.items()}
    info = {
        "connected_population": connected_pop,
        "total_population": total_pop,
        "connected_population_ratio": connected_ratio,
        "travel_time": time_f,
        "baseline_travel_time": time_b,
        "avg_speed_per_hour": avg_speed,
        "node_details": node_details,
    }

    return connected_ratio, system_state, info

def eval_1od_connectivity(
    comps_st: Dict[str, int],
    G: nx.Graph,
    orig_node: str,
    dest_node: str,
    *,
    edge_id_attr: str = "eid",
) -> Tuple[str, int, Dict[str, Any]]:
    """
    Apply component states to G and check connectivity of a single origin-destination pair.

    The node is considered CONNECTED if there exists a path from orig_node to dest_node.

    System state:
        1 if path exists else 0

    Returns:
      ('disconnected' or 'connected', 0 or 1, info dict)
    """
    # --- node/edge states interpreted the same way as your original function ---
    node_off = {cid for cid, st in comps_st.items() if st == 0 and cid in G.nodes}
    edge_on  = {cid for cid, st in comps_st.items() if st == 1}

    # quick failures
    if orig_node in node_off or dest_node in node_off:
        return 0, {"reason": "origin_or_destination_off", "node_off": True}

    if not G.has_node(orig_node) or not G.has_node(dest_node):
        return 0, {"reason": "origin_or_destination_missing_in_base"}

    # --- build filtered graph ---
    H = G.__class__()
    H.add_nodes_from(G.nodes(data=True))

    kept_edges = 0
    skipped_no_eid = 0

    for u, v, data in G.edges(data=True):
        # drop edges incident to removed nodes
        if u in node_off or v in node_off:
            continue

        eid = data.get(edge_id_attr)
        if eid is None:
            skipped_no_eid += 1
            continue

        # keep only "on" edges
        if eid in edge_on:
            H.add_edge(u, v, **data)
            kept_edges += 1

    # If origin/destination got isolated, has_path will return False anyway,
    # but keeping this check makes the reason clearer.
    if not H.has_node(orig_node) or not H.has_node(dest_node):
        return 0, {"reason": "origin_or_destination_missing_in_filtered"}

    try:
        path_nodes = nx.shortest_path(H, orig_node, dest_node)
        connected = True
    except nx.NetworkXNoPath:
        path_nodes = None
        connected = False

    # Get the path edges and nodes
    if path_nodes is not None:
        path_eids = [
            H.edges[u, v]["eid"]
            for u, v in zip(path_nodes[:-1], path_nodes[1:])
        ]
    else:
        path_eids = None


    info = {
        "connected": connected,
        "path_nodes": path_nodes,
        "path_edge_ids": path_eids
    }
    return ("connected" if connected else "disconnected"), (1 if connected else 0), info