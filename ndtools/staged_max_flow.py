from scipy.optimize import linprog
import numpy as np
import copy
from typing import Dict

"""
The reference data are available at datasets/process_plants/

This module implements the staged maximum flow algorithm for e.g. process plant systems.
"""

def eval_edge_caps( nodes: Dict[str, Dict], edges: Dict[str, Dict], probs: Dict[str, Dict], comps_st: Dict[str, int] ):
    """
    Evaluate the capacities of the edges based on the capacities of
    the nodes and the capacities of the edges.

    Args:
    nodes: Dict[str, Dict]
        name: {"x": float, "y": float, "depot": int or null, "capacity": float or null, "comp_id": str}
    edges: Dict[str, Dict]
        name: {"from": str, "to": str, "directed": bool, "capacity": float, "comp_id": str}
    probs: Dict[str, Dict]
        name: {str (comp_id): {"p": float [0,1], "remaining_capacity_ratio": float [0,1]}}
    comps_st: Dict[str, int]
        str (comp_id): int (state)

    Note:
    - probs and comps_st should have the same keys.
    - comps_st's values should be valid keys in probs' values.
    
    Returns:
    e_caps: Dict[str, float]
        name (edge): capacity
    """

    # Check inputs
    assert set(probs.keys()) == set(comps_st.keys()), "probs and comps_st should have the same keys."
    for comp_id, st in comps_st.items():
        assert str(st) in probs[comp_id], f"State {st} not found in probs for component {comp_id}."

    # Get node capacities from nodes, probs, and comps_st
    n_caps_dict = {}
    for n, n_info in nodes.items():
        if n_info["comp_id"] in comps_st:
            comp_id = n_info["comp_id"]
            st = comps_st[comp_id]

            if n_info.get("capacity") is not None:
                base_capacity = n_info["capacity"]
                n_caps_dict[n] = base_capacity * probs[comp_id][str(st)]['remaining_capacity_ratio']
            else:
                n_caps_dict[n] = float('inf')

    # Get edge capacities from edges, probs, and comps_st
    edge_caps_dict = {}
    for e, e_info in edges.items():
        if e_info["comp_id"] in comps_st:
            comp_id = e_info["comp_id"]
            st = comps_st[comp_id]

            if e_info.get("capacity") is not None:
                base_capacity = e_info["capacity"]
                edge_caps_dict[e] = base_capacity * probs[comp_id][str(st)]['remaining_capacity_ratio']
            else:
                edge_caps_dict[e] = float('inf')
            
    # Evaluate edge capacities from n_caps_dict and edge_caps_dict
    # Minimum of the capacities of the two end nodes and the edge itself
    e_caps = {}
    for e, e_info in edges.items():
        from_node = e_info["from"]
        to_node = e_info["to"]

        node_cap_from = n_caps_dict.get(from_node, float('inf'))
        node_cap_to = n_caps_dict.get(to_node, float('inf'))
        edge_cap = edge_caps_dict.get(e, float('inf'))

        e_caps[e] = min(node_cap_from, node_cap_to, edge_cap)        

    return e_caps

def get_decision_var_details(edges: Dict[str, Dict]):
    """
    Get details of decision variables.

    Args:
    edges: Dict[str, Dict]
        name: {"from": str, "to": str, "directed": bool, "capacity": float, "comp_id": str}

    Returns:
    decision_var_details: List[Dict]

    """

    decision_var_details = []
    for e, e_info in edges.items():
        for stage in e_info.get("transition_start_stage", []):
            detail = {
                "edge": e,
                "from": e_info["from"],
                "to": e_info["to"],
                "transition_start_stage": stage
            }
            decision_var_details.append(detail)

    return decision_var_details


def create_lp(e_caps: Dict[str, float], nodes: Dict[str, Dict], edges: Dict[str, Dict]):
    """
    Create a linear programming problem for the pipe network optimisation.

    Args:
    e_caps: Dict[str, float]
        name (edge): capacity
        Computed by eval_edge_caps().
    nodes: Dict[str, Dict]
        name: {"x": float, "y": float, "depot": int or null, "capacity": float or null, "comp_id": str}
    edges: Dict[str, Dict]
        name: {"from": str, "to": str, "directed": bool, "capacity": float, "comp_id": str}

    Returns:
    c: array
        Coefficients of the objective function.
    A_eq: array
        Coefficients of the equality constraints.
    b_eq: array
        Right-hand side of the equality constraints.
    bounds: list of tuples
        Bounds of the variables.
    """
    # prepare inputs
    ## station nodes list [[stage1_nodes], [stage2_nodes], ...]
    stations = []
    for n, n_info in nodes.items():
        if n_info.get("station_stage") is not None:
            station_idx = n_info["station_stage"]
            while len(stations) <= station_idx-1:
                stations.append([])
            stations[station_idx-1].append(n)

    n_station = len(stations)

    decision_var_details = get_decision_var_details(edges)


    # cost matrix c on decision variables ({x}, u)
    n_dec = len(decision_var_details)
    c = np.zeros((n_dec+1,))
    c[-1] = -1  


    # constraint matrices A_eq and b_eq
    A_eq = np.empty(shape=(0, n_dec+1))
    b_eq = np.empty(shape=(0,))

    # constraint 1: balance in-flows to station nodes
    # Stage 1 -> 2
    A_d_ = np.zeros((1, n_dec+1))
    for idx, dec in enumerate(decision_var_details):
        i, j, m = dec["from"], dec["to"], dec["transition_start_stage"]
        if i in stations[0] and m == 1:
            A_d_[0, idx] = 1.0
    A_d_[0, -1] = -1.0
    A_eq = np.vstack((A_eq, A_d_))
    b_eq = np.append(b_eq, 0.0)

    # constraint 2: balance out-flows of station nodes
    # Stage 2 -> 3, ..., M-1 -> M
    for s_idx in range(1, n_station):
        A_d_ = np.zeros((1, n_dec+1))
        for idx, dec in enumerate(decision_var_details):
            i, j, m = dec["from"], dec["to"], dec["transition_start_stage"]
            if j in stations[s_idx] and m == s_idx:
                A_d_[0, idx] = 1.0
        A_d_[0, -1] = -1.0
        A_eq = np.vstack((A_eq, A_d_))
        b_eq = np.append(b_eq, 0.0)

    # constraint 3: balance in- and out-flows of station nodes,
    # except for the first and the last stages
    for n, n_info in nodes.items():
        station_stage = n_info.get("station_stage", None)
        if (station_stage is not None) and (station_stage != 1) and (station_stage != n_station):
            A_n_ = np.zeros((1, n_dec+1))
                
            for idx, dec in enumerate(decision_var_details):
                i, j, m = dec["from"], dec["to"], dec["transition_start_stage"]
                if j == n and m == station_stage - 1:
                    A_n_[0, idx] = 1.0
                elif i == n and m == station_stage:
                    A_n_[0, idx] = -1.0

            A_eq = np.vstack((A_eq, A_n_))
            b_eq = np.append(b_eq, 0.0)

    # contraint 4: balance flows at non-station nodes
    for n, n_info in nodes.items(): 
        if all( n not in d for d in stations ):
            ms_n = []
            for e_, e_info in edges.items():
                if e_info["from"] == n or e_info["to"] == n:
                    ms_n += e_info.get("transition_start_stage", []) 
            ms_n = np.unique(ms_n) # unique stage transitions (start stages) that use node n

            for s_idx in ms_n:
                A_n_ = np.zeros((1, n_dec+1))

                for e_idx, dec in enumerate(decision_var_details):
                    i, j, m = dec["from"], dec["to"], dec["transition_start_stage"]
                    if j == n and m == s_idx:
                        A_n_[0, e_idx] = 1.0
                    elif i == n and m == s_idx:
                        A_n_[0, e_idx] = -1.0

                A_eq = np.vstack((A_eq, A_n_))
                b_eq = np.append(b_eq, 0.0)

    # Bounds: capacity of the edges
    bounds = []
    for dec in decision_var_details:
        bounds.append( (0.0, e_caps[dec["edge"]]) )
    bounds.append( (0.0, np.inf) )

    return c, A_eq, b_eq, bounds, decision_var_details

# TODO: fix this function--it does not return the minimal component states correctly
def get_min_surv_comps_st(results, nodes, edges, probs):
    """
    Get the minimal component states to ensure system survival.
    components not in the dictionary do not affect the system state.
    NOTE: the returned state is not necessarily minimal. It is minimal when the max flow (i.e. results.x[-1]) equals the target flow; if max flow > target flow, the returned state may not be minimal.

    Args:
    results: OptimizeResult
        The result of the linear programming optimisation:
        results.x: array of decision variable values.
        results.decision_var_details: List[Dict] - details of decision variables.
    nodes: Dict[str, Dict]
        name: {"x": float, "y": float, "depot": int or null, "capacity": float or null, "comp_id": str}
    edges: Dict[str, Dict]
        name: {"from": str, "to": str, "directed": bool, "capacity": float, "comp_id": str}
    probs: Dict[str, Dict]
        name: {str (comp_id): {"p": float [0,1], "remaining_capacity_ratio": float [0,1]}}

    Returns:
    min_surv_comps_st_dict: Dict[str, int] - minimal component states to ensure system survival
        component_name: minimum survival state
        
    """
    # Create min_surv_comps_st_dict with minimum state
    min_surv_comps_st_dict = {}

    # Loop over each decision variable (edge flow)
    for f, dec in zip(results.x[:-1], results.decision_var_details):

        if f > 0:
            # Edge
            x_e = edges[dec["edge"]]["comp_id"]
            if x_e is not None:
                min_st_e = min((int(k) for k, v in probs[x_e].items() if v["remaining_capacity_ratio"] * edges[dec["edge"]]["capacity"] >= f - 1e-12), default=None)
                assert min_st_e is not None, f"No valid state found for edge {dec["edge"]} to meet flow {f:1.2e}."
                if x_e not in min_surv_comps_st_dict:
                    min_surv_comps_st_dict[x_e] = ('>=', min_st_e)
                else:
                    min_surv_comps_st_dict[x_e] = ('>=', max(min_surv_comps_st_dict[x_e][1], min_st_e))

            # From node
            x_from = nodes[dec["from"]]["comp_id"]
            if x_from is not None:
                min_st_from = min((int(k) for k, v in probs[x_from].items() if v["remaining_capacity_ratio"] * nodes[dec["from"]]["capacity"] >= f - 1e-12), default=None)
                assert min_st_from is not None, f"No valid state found for node {dec['from']} to meet flow {f:1.2e}."
                if x_from not in min_surv_comps_st_dict:
                    min_surv_comps_st_dict[x_from] = ('>=', min_st_from)
                else:
                    min_surv_comps_st_dict[x_from] = ('>=', max(min_surv_comps_st_dict[x_from][1], min_st_from))

            # To node
            x_to = nodes[dec["to"]]["comp_id"]
            if x_to is not None:
                min_st_to = min((int(k) for k, v in probs[x_to].items() if v["remaining_capacity_ratio"] * nodes[dec["to"]]["capacity"] >= f - 1e-12), default=None)
                assert min_st_to is not None, f"No valid state found for node {dec['to']} to meet flow {f:1.2e}."
                if x_to not in min_surv_comps_st_dict:
                    min_surv_comps_st_dict[x_to] = ('>=', min_st_to)
                else:
                    min_surv_comps_st_dict[x_to] = ('>=', max(min_surv_comps_st_dict[x_to][1], min_st_to))
    
    min_surv_comps_st_dict_sorted = {} # in the same order as probs
    for comp_id in probs.keys():
        if comp_id in min_surv_comps_st_dict:
            min_surv_comps_st_dict_sorted[comp_id] = min_surv_comps_st_dict[comp_id]

    return min_surv_comps_st_dict_sorted

def run_staged_max_flow( comps_st: Dict[str, int], nodes: Dict[str, Dict], edges: Dict[str, Dict], probs: Dict[str, Dict] ):
    """
    Run the staged maximum flow algorithm to evaluate the system state.
    """

    e_caps = eval_edge_caps( nodes, edges, probs, comps_st )
    c_, A_eq_, b_eq_, bounds_, decision_var_details = create_lp(e_caps, nodes, edges)
    result = linprog(c_, A_eq=A_eq_, b_eq=b_eq_, bounds=bounds_)
    result.decision_var_details = decision_var_details

    return result

def sys_fun(comps_st: Dict[str, int], nodes: Dict[str, Dict], edges: Dict[str, Dict], probs: Dict[str, Dict], target_flow: float):

    """
    The function to evaluate the system state using the staged maximum flow algorim,
    for a given component states (ref: TBC).

    Args:
    comps_st: Dict[str, int]
        str (comp_id): int (state)
    nodes: Dict[str, Dict]
        name: {"x": float, "y": float, "depot": int or null, "capacity": float or null, "comp_id": str}
    edges: Dict[str, Dict]
        name: {"from": str, "to": str, "directed": bool, "capacity": float, "comp_id": str}
    probs: Dict[str, Dict]
        name: {str (comp_id): {"p": float [0,1], "remaining_capacity_ratio": float [0,1]}}
    target_flow: float
        The target flow to determine system survival.
    """

    result = run_staged_max_flow( comps_st, nodes, edges, probs )    

    flow = result.x[-1]
    if flow >= target_flow - 1e-12:
        sys_st = 's'
        min_comp_state = get_min_surv_comps_st(result, nodes, edges, probs) # TODO: the function doesn't run correctly
        #min_comp_state = None               

    else:
        sys_st = 'f'
        min_comp_state = None

    return flow, sys_st, min_comp_state

def add_a_component( rv_name_to_copy, nodes: Dict[str, Dict], edges: Dict[str, Dict], probs: Dict[str, Dict] ):
    """
    Return a copy of the input data with an additional random variable added.
    This function is to calculate addition importance measure (AIM) of a random variable (or component event).

    Args:
    rv_name_to_copy: str
        The name of the random variable to be copied.
        Must exist as a key in probs.
    nodes: Dict[str, Dict]
        name: {"x": float, "y": float, "depot": int or null, "capacity": float or null, "comp_id": str}
    edges: Dict[str, Dict]
        name: {"from": str, "to": str, "directed": bool, "capacity": float, "comp_id": str}
    probs: Dict[str, Dict]
        name: {str (comp_id): {"p": float [0,1], "remaining_capacity_ratio": float [0,1]}}

    Returns:
    new_nodes: Dict[str, Dict]
        The updated nodes dictionary with the new random variable added.
    new_edges: Dict[str, Dict]
        The updated edges dictionary with the new random variable added.
    new_probs: Dict[str, Dict]
        The updated probs dictionary with the new random variable added.
    """

    assert rv_name_to_copy in probs, f"Random variable {rv_name_to_copy} not found in probs."

    # Create deep copies of the input data
    new_nodes = copy.deepcopy(nodes)
    new_edges = copy.deepcopy(edges)
    new_probs = copy.deepcopy(probs)

    # Copy probs entry
    rv_copied_name = f"{rv_name_to_copy}_copy"
    new_probs[rv_copied_name] = copy.deepcopy(probs[rv_name_to_copy])

    # Copy nodes entries
    for n, n_info in nodes.items():
        if n_info["comp_id"] == rv_name_to_copy:
            new_n_info = copy.deepcopy(n_info)
            new_n_info["comp_id"] = rv_copied_name
            new_node_name = f"{n}_copy"
            new_nodes[new_node_name] = new_n_info

            # Update edges to connect to the new node copy
            for e, e_info in edges.items():
                if e_info["from"] == n:
                    new_e_info = copy.deepcopy(e_info)
                    new_e_info["from"] = new_node_name
                    new_edge_name = f"{e}_from_{new_node_name}"
                    new_edges[new_edge_name] = new_e_info # this new edge has the same comp_id as the original edge
                if e_info["to"] == n:
                    new_e_info = copy.deepcopy(e_info)
                    new_e_info["to"] = new_node_name
                    new_edge_name = f"{e}_to_{new_node_name}"
                    new_edges[new_edge_name] = new_e_info # this new edge has the same comp_id as the original edge

    # Copy edges entries
    for e, e_info in edges.items():
        if e_info["comp_id"] == rv_name_to_copy:
            new_e_info = copy.deepcopy(e_info)
            new_e_info["comp_id"] = rv_copied_name
            new_edge_name = f"{e}_copy"
            new_edges[new_edge_name] = new_e_info

    return new_nodes, new_edges, new_probs

def deactivate_a_component( rv_name_to_deactivate, nodes: Dict[str, Dict], edges: Dict[str, Dict], probs: Dict[str, Dict] ):
    """
    Return a copy of the input data with a random variable deactivated.
    This function is to calculate deactivation importance measure (DIM) of a random variable (or component event).

    Args:
    rv_name_to_deactivate: str
        The name of the random variable to be deactivated.
        Must exist as a key in probs.
    nodes: Dict[str, Dict]
        name: {"x": float, "y": float, "depot": int or null, "capacity": float or null, "comp_id": str}
    edges: Dict[str, Dict]
        name: {"from": str, "to": str, "directed": bool, "capacity": float, "comp_id": str}
    probs: Dict[str, Dict]
        name: {str (comp_id): {"p": float [0,1], "remaining_capacity_ratio": float [0,1]}}

    Returns:
    new_nodes: Dict[str, Dict]
        The updated nodes dictionary with the random variable deactivated.
    new_edges: Dict[str, Dict]
        The updated edges dictionary with the random variable deactivated.
    new_probs: Dict[str, Dict]
        The updated probs dictionary (same as input).
    """

    assert rv_name_to_deactivate in probs, f"Random variable {rv_name_to_deactivate} not found in probs."

    # Create deep copies of the input data
    new_nodes = copy.deepcopy(nodes)
    new_edges = copy.deepcopy(edges)
    new_probs = copy.deepcopy(probs)

    # Remove in nodes
    nodes_to_delete = [n for n, n_info in nodes.items() if n_info["comp_id"] == rv_name_to_deactivate]
    for n in nodes_to_delete:
        new_nodes[n]['capacity'] = 0.0
    if len(nodes_to_delete) > 0:
        # Also remove edges connected to the removed nodes
        edges_to_delete = [e for e, e_info in new_edges.items() if e_info["from"] in nodes_to_delete or e_info["to"] in nodes_to_delete]
        for e in edges_to_delete:
            new_edges[e]['capacity'] = 0.0

    # Remove in edges
    edges_to_delete = [e for e, e_info in edges.items() if e_info["comp_id"] == rv_name_to_deactivate]
    for e in edges_to_delete:
        new_edges[e]['capacity'] = 0.0

    return new_nodes, new_edges, probs