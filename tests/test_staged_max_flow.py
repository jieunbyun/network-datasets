from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import pytest

from ndtools import staged_max_flow as smf

@pytest.fixture
def load_toy_process_plant_dataset() -> tuple[dict, dict, dict]:
    """Load the toy process plant dataset for testing."""
    data_dir = Path(__file__).parent.parent / "datasets" / "process_plants" / "toy" / "v1" / "data"
    nodes = json.loads((data_dir / "nodes.json").read_text())
    edges = json.loads((data_dir / "edges.json").read_text())
    probs = json.loads((data_dir / "probs.json").read_text())
    return nodes, edges, probs

def test_eval_edge_caps1(load_toy_process_plant_dataset):
    
    nodes, edges, probs = load_toy_process_plant_dataset

    # All components have been survived
    comps_st = {comp_id: 1 for comp_id in probs.keys()}

    # Run eval_edge_caps
    edge_caps = smf.eval_edge_caps(nodes, edges, probs, comps_st)

    expected = {"e1": 0.5, "e2": 0.5, "e3": 1.0, "e4": 0.5,
                "e5": 0.5, "e6": 1.0, "e7": 1.0, "e8": 0.5,
                "e9": 0.5, "e10": 0.5, "e11": 1.0, "e12": 1.0,
                "e13": 0.5, "e14": 0.5, "e15": 0.5, "e16": 0.5,
                "e17": 1.0, "e18": 0.5, "e19": 1.0}
    
    # Check edge capacities
    assert edge_caps == expected

def test_eval_edge_caps2(load_toy_process_plant_dataset):

    nodes, edges, probs = load_toy_process_plant_dataset

    # Some components have failed (Case #2 in the didactic example in Byun and Lee 2026)
    comps_st = {c: 1 for c in probs.keys()}
    comps_st["x5"], comps_st["x17"] = 0, 0

    # Run eval_edge_caps
    edge_caps = smf.eval_edge_caps(nodes, edges, probs, comps_st)

    expected = {"e1": 0.5, "e2": 0.5, "e3": 1.0, "e4": 0.5,
                "e5": 0.5, "e6": 1.0, "e7": 1.0, "e8": 0.5,
                "e9": 0.5, "e10": 0.5, "e11": 1.0, "e12": 1.0,
                "e13": 0.0, "e14": 0.0, "e15": 0.5, "e16": 0.5,
                "e17": 1.0, "e18": 0.5, "e19": 1.0}
    
    # Check edge capacities
    assert edge_caps == expected

def test_eval_edge_caps3(load_toy_process_plant_dataset):

    nodes, edges, probs = load_toy_process_plant_dataset

    # Some components have failed (Case #3 in the didactic example in Byun and Lee 2026)
    comps_st = {c: 1 for c in probs.keys()}
    comps_st["x5"] = 0

    # Run eval_edge_caps
    edge_caps = smf.eval_edge_caps(nodes, edges, probs, comps_st)

    # Note: the result should be the same as test_eval_edge_caps2
    expected = {"e1": 0.5, "e2": 0.5, "e3": 1.0, "e4": 0.5,
                "e5": 0.5, "e6": 1.0, "e7": 1.0, "e8": 0.5,
                "e9": 0.5, "e10": 0.5, "e11": 1.0, "e12": 1.0,
                "e13": 0.0, "e14": 0.0, "e15": 0.5, "e16": 0.5,
                "e17": 1.0, "e18": 0.5, "e19": 1.0}
    
    # Check edge capacities
    assert edge_caps == expected

def test_eval_edge_caps4(load_toy_process_plant_dataset):

    nodes, edges, probs = load_toy_process_plant_dataset

    # Some components have failed 
    comps_st = {c: 1 for c in probs.keys()}
    comps_st["x8"] = 0

    # Run eval_edge_caps
    edge_caps = smf.eval_edge_caps(nodes, edges, probs, comps_st)

    expected = {"e1": 0.5, "e2": 0.5, "e3": 1.0, "e4": 0.5,
                "e5": 0.5, "e6": 1.0, "e7": 1.0, "e8": 0.5,
                "e9": 0.5, "e10": 0.5, "e11": 1.0, "e12": 1.0,
                "e13": 0.5, "e14": 0.5, "e15": 0.5, "e16": 0.5,
                "e17": 1.0, "e18": 0.5, "e19": 0.0}
    
    # Check edge capacities
    assert edge_caps == expected

def test_eval_edge_caps5(load_toy_process_plant_dataset):

    nodes, edges, probs = load_toy_process_plant_dataset

    # Some components have failed 
    comps_st = {c: 1 for c in probs.keys()}
    comps_st["x8"] = 0
    # Change a node capacity
    nodes["n7"]["capacity"] = 0.25

    # Run eval_edge_caps
    edge_caps = smf.eval_edge_caps(nodes, edges, probs, comps_st)

    expected = {"e1": 0.5, "e2": 0.5, "e3": 1.0, "e4": 0.5,
                "e5": 0.5, "e6": 1.0, "e7": 1.0, "e8": 0.5,
                "e9": 0.25, "e10": 0.25, "e11": 1.0, "e12": 1.0,
                "e13": 0.5, "e14": 0.5, "e15": 0.5, "e16": 0.5,
                "e17": 1.0, "e18": 0.5, "e19": 0.0}
    
    # Check edge capacities
    assert edge_caps == expected

def test_get_decision_var_details1(load_toy_process_plant_dataset):

    # Load data
    nodes, edges, probs = load_toy_process_plant_dataset

    # Run function
    decision_var_details = smf.get_decision_var_details(edges)

    # Expected
    assert len(decision_var_details) == 19+2 # e6 and e11 are responsible for 2 transitions
    assert decision_var_details[0] =={"edge": "e1", "from": "n1", "to": "n3", "transition_start_stage": 1}
    assert decision_var_details[1] =={"edge": "e2", "from": "n2", "to": "n3", "transition_start_stage": 1}
    assert decision_var_details[5] =={"edge": "e6", "from": "n4", "to": "n6", "transition_start_stage": 1}
    assert decision_var_details[6] =={"edge": "e6", "from": "n4", "to": "n6", "transition_start_stage": 2}
    assert decision_var_details[10] =={"edge": "e10", "from": "n7", "to": "n6", "transition_start_stage": 2}
    assert decision_var_details[11] =={"edge": "e11", "from": "n6", "to": "n8", "transition_start_stage": 1}
    assert decision_var_details[12] =={"edge": "e11", "from": "n6", "to": "n8", "transition_start_stage": 2}
    assert decision_var_details[-1] =={"edge": "e19", "from": "n13", "to": "n14", "transition_start_stage": 3}


@pytest.fixture
def expected_toy_lp():
    n_edge_stage = 19 + 2 # e6 and e11 are responsible for 2 transitions
    c_expected = [0.0] * n_edge_stage + [-1.0]
    
    A_eq_expected = [
        # constraint 1: balance in-flows to station nodes - Stage 1 -> 2
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], 
        # constraint 2: balance in-flows to station nodes - Stage 2 -> 3, ..., M-1 -> M
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], 
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
        # constraint 3: balance in- and out-flows of station nodes, except for the first and the last stages
        [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # stage 2, n5
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # stage 2, n7
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # stage 2, n9
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], # stage 3, n10
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0], # stage 3, n12
        # constraint 4: balance flows at non-station nodes
        [1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # stage 1, n3
        [0.0, 0.0, 1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # stage 1, n4
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # stage 2, n4
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # stage 1, n6
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # stage 2, n6
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # stage 1, n8
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], # stage 2, n8
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0], # stage 3, n11
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0] # stage 3, n13
    ]
    b_eq_expected = [0.0] * 18
    bounds_expected = [(0.0, 0.5), (0.0, 0.5), (0.0, 1.0), (0.0, 0.5),
                       (0.0, 0.5), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 0.5),
                       (0.0, 0.5), (0.0, 0.5), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
                       (0.0, 0.5), (0.0, 0.5), (0.0, 0.5), (0.0, 0.5),
                       (0.0, 1.0), (0.0, 0.5), (0.0, 1.0), (0.0, np.inf)]
    
    c_expected = np.array(c_expected)
    A_eq_expected = np.array(A_eq_expected)
    b_eq_expected = np.array(b_eq_expected)
    return c_expected, A_eq_expected, b_eq_expected, bounds_expected

def test_create_lp1(load_toy_process_plant_dataset, expected_toy_lp):

    nodes, edges, probs = load_toy_process_plant_dataset

    # Prepare input
    comps_st = {c: 1 for c in probs.keys()}
    e_caps = smf.eval_edge_caps(nodes, edges, probs, comps_st)

    # Run function
    c, A_eq, b_eq, bounds, decision_var_details = smf.create_lp(e_caps, nodes, edges)

    c_expected, A_eq_expected, b_eq_expected, bounds_expected = expected_toy_lp

    # Check results
    np.testing.assert_array_almost_equal(c, c_expected)

    try:
        np.testing.assert_array_almost_equal(A_eq, A_eq_expected)
    except AssertionError as e: # find locations of differences
        diff_mask = ~np.isclose(A_eq, A_eq_expected)
        rows, cols = np.where(diff_mask)
        for r, c in zip(rows, cols):
            print(f"({r}, {c}): A_eq={A_eq[r, c]}, expected={A_eq_expected[r, c]}")

    np.testing.assert_array_almost_equal(b_eq, b_eq_expected)
    assert bounds == bounds_expected

def test_create_lp2(load_toy_process_plant_dataset, expected_toy_lp):

    nodes, edges, probs = load_toy_process_plant_dataset

    # Prepare input
    comps_st = {c: 1 for c in probs.keys()}
    # Some components have failed
    comps_st["x5"], comps_st["x17"] = 0, 0

    e_caps = smf.eval_edge_caps(nodes, edges, probs, comps_st)

    # Run function
    c, A_eq, b_eq, bounds, decision_var_details = smf.create_lp(e_caps, nodes, edges)

    # Expected values
    c_expected, A_eq_expected, b_eq_expected, bounds_expected = expected_toy_lp
    # The assumed failures affect only the bounds
    bounds_expected[15-1] = (0.0, 0.0)  # e13
    bounds_expected[16-1] = (0.0, 0.0)  # e14

    # Check results
    np.testing.assert_array_almost_equal(c, c_expected)

    try:
        np.testing.assert_array_almost_equal(A_eq, A_eq_expected)
    except AssertionError as e: # find locations of differences
        diff_mask = ~np.isclose(A_eq, A_eq_expected)
        rows, cols = np.where(diff_mask)
        for r, c in zip(rows, cols):
            print(f"({r}, {c}): A_eq={A_eq[r, c]}, expected={A_eq_expected[r, c]}")

    np.testing.assert_array_almost_equal(b_eq, b_eq_expected)
    assert bounds == bounds_expected

def test_run_staged_max_flow1(load_toy_process_plant_dataset):

    nodes, edges, probs = load_toy_process_plant_dataset

    # Prepare input
    comps_st = {c: 1 for c in probs.keys()}

    # Run function
    result = smf.run_staged_max_flow( comps_st, nodes, edges, probs )

    # Check the result
    assert result.x[-1] == 1.0 # system's maximum flow

    # NOTE: the solution is not unique; just check one possible solution
    candidates = [
        # when n5 and n9 are used for stage 2's deposits 
        np.array([0.5, 0.5, 1.0, 0.5, 0.5, # e1 to e5
                  0.5, 0.0, 0.0, 0.5, 0.0, 0.0, # e6-1, e6-2, e7, e8, e9 , e10
                  0.5, 0.0, 0.0, 0.5, 0.5, 0.5, # e11-1, e11-2, e12, e13, e14, e15
                  0.5, 0.5, 0.5, 1.0]), # e16, e17, e18, e19 

        # when n7 and n9 are used for stage 2's deposits 
        np.array([0.5, 0.5, 1.0, 0.0, 0.0, # e1 to e5
                  1.0, 0.0, 0.5, 0.5, 0.5, 0.5, # e6-1, e6-2, e7, e8, e9 , e10
                  0.5, 0.0, 0.0, 0.5, 0.5, 0.5, # e11-1, e11-2, e12, e13, e14, e15
                  0.5, 0.5, 0.5, 1.0]), # e16, e17, e18, e19

        # when n5 and n7 are used for stage 2's deposits
        np.array([0.5, 0.5, 1.0, 0.5, 0.5, # e1 to e5
                  0.5, 0.0, 0.0, 0.5, 0.5, 0.5, # e6-1, e6-2, e7, e8, e9, e10
                  0.0, 0.5, 0.0, 0.0, 0.0, 0.5, # e11-1, e11-2, e12, e13, e14, e15
                  0.5, 0.5, 0.5, 1.0]) # e16, e17, e18, e19 
    ]

    assert any( np.allclose(result.x[:-1], candidate) for candidate in candidates )

def test_run_staged_max_flow2(load_toy_process_plant_dataset):

    nodes, edges, probs = load_toy_process_plant_dataset

    # Prepare input
    comps_st = {c: 1 for c in probs.keys()}
    comps_st['x5'], comps_st['x17'] = 0, 0

    # Run function
    result = smf.run_staged_max_flow( comps_st, nodes, edges, probs )

    # Check the result
    assert result.x[-1] == 1.0 # system's maximum flow

    # NOTE: the solution is not unique; just check one possible solution
    candidates = [
        # when n5 and n7 are used for stage 2's deposits
        # n9 is not available due to the failure of x5 and x17
        np.array([0.5, 0.5, 1.0, 0.5, 0.5, # e1 to e5
                  0.5, 0.0, 0.0, 0.5, 0.5, 0.5, # e6-1, e6-2, e7, e8, e9, e10
                  0.0, 0.5, 0.0, 0.0, 0.0, 0.5, # e11-1, e11-2, e12, e13, e14, e15
                  0.5, 0.5, 0.5, 1.0]) # e16, e17, e18, e19 
    ]

    assert any( np.allclose(result.x[:-1], candidate) for candidate in candidates )

def test_run_staged_max_flow3(load_toy_process_plant_dataset):

    nodes, edges, probs = load_toy_process_plant_dataset

    # Prepare input
    comps_st = {c: 1 for c in probs.keys()}
    comps_st['x5'], comps_st['x12'] = 0, 0

    # Run function
    result = smf.run_staged_max_flow( comps_st, nodes, edges, probs )

    # Check the result
    assert result.x[-1] == 0.5 # system's maximum flow

    # NOTE: the solution is not unique; just check one possible solution
    candidates = [
        # when only n7 is used for stage 2's deposits
        np.array([0.0, 0.5, 0.5, 0.0, 0.0, # e1 to e5
                  0.5, 0.0, 0.0, 0.0, 0.5, 0.5, # e6-1, e6-2, e7, e8, e9, e10
                  0.0, 0.5, 0.0, 0.0, 0.0, 0.5, # e11-1, e11-2, e12, e13, e14, e15
                  0.0, 0.0, 0.5, 0.5]) # e16, e17, e18, e19 
    ]

    assert any( np.allclose(result.x[:-1], candidate) for candidate in candidates )

@pytest.mark.skip(reason="get_min_surv_comps_st is incorrect")
def test_get_min_surv_comps_st1(load_toy_process_plant_dataset):
    nodes, edges, probs = load_toy_process_plant_dataset

    # Prepare input
    comps_st = {c: 1 for c in probs.keys()}

    result = smf.run_staged_max_flow( comps_st, nodes, edges, probs )

    # Run function
    min_comp_state = smf.get_min_surv_comps_st( result, nodes, edges, probs )

    # Check the result
    candidates = [
        # when n5 and n9 are used for stage 2's deposits
        {'x1': ('>=', 1), 'x2': ('>=', 1), 'x3': ('>=', 1), 'x4': ('>=', 1), 'x6': ('>=', 1), 'x7': ('>=', 1), 'x8': ('>=', 1), # nodes
         'x9': ('>=', 1), 'x10': ('>=', 1), 'x11': ('>=', 1), 'x12': ('>=', 1), 'x13': ('>=', 1), 'x14': ('>=', 1), 'x16': ('>=', 1), 'x17': ('>=', 1), 'x18': ('>=', 1), 'x19': ('>=', 1), 'x20': ('>=', 1), 'x21': ('>=', 1), 'x22': ('>=', 1)}, # edges
        # when n5 and n7 are used for stage 2's deposits
        {'x1': ('>=', 1), 'x2': ('>=', 1), 'x3': ('>=', 1), 'x5': ('>=', 1), 'x6': ('>=', 1), 'x7': ('>=', 1), 'x8': ('>=', 1), # nodes
         'x9': ('>=', 1), 'x10': ('>=', 1), 'x11': ('>=', 1), 'x12': ('>=', 1), 'x13': ('>=', 1), 'x14': ('>=', 1), 'x15': ('>=', 1), 'x16': ('>=', 1), 'x18': ('>=', 1), 'x19': ('>=', 1), 'x20': ('>=', 1), 'x21': ('>=', 1), 'x22': ('>=', 1)}, # edges
        # when n7 and n9 are used for stage 2's deposits
        {'x1': ('>=', 1), 'x2': ('>=', 1), 'x4': ('>=', 1), 'x5': ('>=', 1), 'x6': ('>=', 1), 'x7': ('>=', 1), 'x8': ('>=', 1), # nodes
         'x9': ('>=', 1), 'x10': ('>=', 1), 'x11': ('>=', 1), 'x13': ('>=', 1), 'x14': ('>=', 1), 'x15': ('>=', 1), 'x16': ('>=', 1), 'x17': ('>=', 1), 'x18': ('>=', 1), 'x19': ('>=', 1), 'x20': ('>=', 1), 'x21': ('>=', 1), 'x22': ('>=', 1)} # edges       
    ]
    assert any( min_comp_state == candidate for candidate in candidates )

@pytest.mark.skip(reason="get_min_surv_comps_st is incorrect")
def test_get_min_surv_comps_st2(load_toy_process_plant_dataset):
    nodes, edges, probs = load_toy_process_plant_dataset

    # Prepare input
    comps_st = {c: 1 for c in probs.keys()}
    comps_st['x5'], comps_st['x17'] = 0, 0

    result = smf.run_staged_max_flow( comps_st, nodes, edges, probs )

    # Run function
    min_comp_state = smf.get_min_surv_comps_st( result, nodes, edges, probs )

    # Check the result
    candidates = [
        # when n5 and n7 are used for stage 2's deposits
        {'x1': ('>=', 1), 'x2': ('>=', 1), 'x3': ('>=', 1), 'x4': ('>=', 1), 'x6': ('>=', 1), 'x7': ('>=', 1), 'x8': ('>=', 1), # nodes
         'x9': ('>=', 1), 'x10': ('>=', 1), 'x11': ('>=', 1), 'x12': ('>=', 1), 'x13': ('>=', 1), 'x14': ('>=', 1), 'x15': ('>=', 1), 'x16': ('>=', 1), 'x18': ('>=', 1), 'x19': ('>=', 1), 'x20': ('>=', 1), 'x21': ('>=', 1), 'x22': ('>=', 1)}, # edges 
    ]
    assert any( min_comp_state == candidate for candidate in candidates )

@pytest.mark.skip(reason="get_min_surv_comps_st is incorrect")
def test_get_min_surv_comps_st3(load_toy_process_plant_dataset):
    nodes, edges, probs = load_toy_process_plant_dataset

    # Prepare input
    comps_st = {c: 1 for c in probs.keys()}
    comps_st['x5'], comps_st['x12'] = 0, 0

    result = smf.run_staged_max_flow( comps_st, nodes, edges, probs )

    # Run function
    min_comp_state = smf.get_min_surv_comps_st( result, nodes, edges, probs )

    # Check the result
    candidates = [
        # when n2 is used for stage 1, n7 is used for stage 2's deposits and n12 for stage 3's deposit
        {'x2': ('>=', 1), 'x4': ('>=', 1), 'x7': ('>=', 1), 'x8': ('>=', 1), # nodes
         'x10': ('>=', 1), 'x11': ('>=', 1), 'x13': ('>=', 1), 'x15': ('>=', 1), 'x16': ('>=', 1), 'x18': ('>=', 1), 'x21': ('>=', 1), 'x22': ('>=', 1)}, # edges 
        # when n1 is used for stage 1, n7 is used for stage 2's deposits and n12 for stage 3's deposit
        {'x1': ('>=', 1), 'x4': ('>=', 1), 'x7': ('>=', 1), 'x8': ('>=', 1), # nodes
         'x9': ('>=', 1), 'x11': ('>=', 1), 'x13': ('>=', 1), 'x15': ('>=', 1), 'x16': ('>=', 1), 'x18': ('>=', 1), 'x21': ('>=', 1), 'x22': ('>=', 1)}, # edges 
        # when n2 is used for stage 1, n7 is used for stage 2's deposits and n10 for stage 3's deposit
        {'x2': ('>=', 1), 'x4': ('>=', 1), 'x6': ('>=', 1), 'x8': ('>=', 1), # nodes
         'x10': ('>=', 1), 'x11': ('>=', 1), 'x13': ('>=', 1), 'x14': ('>=', 1), 'x15': ('>=', 1), 'x19': ('>=', 1), 'x20': ('>=', 1), 'x22': ('>=', 1)}, # edges 
        # when n1 is used for stage 1, n7 is used for stage 2's deposits and n10 for stage 3's deposit
        {'x1': ('>=', 1), 'x4': ('>=', 1), 'x7': ('>=', 1), 'x8': ('>=', 1), # nodes
         'x9': ('>=', 1), 'x11': ('>=', 1), 'x13': ('>=', 1), 'x14': ('>=', 1), 'x15': ('>=', 1), 'x19': ('>=', 1), 'x20': ('>=', 1), 'x22': ('>=', 1)}, # edges 
    ]
    assert any( min_comp_state == candidate for candidate in candidates )

def test_sys_fun1(load_toy_process_plant_dataset):

    nodes, edges, probs = load_toy_process_plant_dataset

    # Prepare input
    comps_st = {c: 1 for c in probs.keys()}

    # Run function
    max_flow, sys_st, min_comp_state = smf.sys_fun( comps_st, nodes, edges, probs, target_flow=1.0 )

    # Check the result
    assert max_flow == 1.0
    assert sys_st == 's'

def test_sys_fun2(load_toy_process_plant_dataset):

    nodes, edges, probs = load_toy_process_plant_dataset

    # Prepare input
    comps_st = {c: 1 for c in probs.keys()}
    comps_st['x5'], comps_st['x17'] = 0, 0

    # Run function
    max_flow, sys_st, min_comp_state = smf.sys_fun( comps_st, nodes, edges, probs, target_flow=1.0 )

    # Check the result
    assert max_flow == 1.0
    assert sys_st == 's'
    assert isinstance(min_comp_state, dict)

def test_sys_fun3(load_toy_process_plant_dataset):

    nodes, edges, probs = load_toy_process_plant_dataset

    # Prepare input
    comps_st = {c: 1 for c in probs.keys()}
    comps_st['x5'], comps_st['x17'] = 0, 0

    # Run function
    max_flow, sys_st, min_comp_state = smf.sys_fun( comps_st, nodes, edges, probs, target_flow=2.0 )

    # Check the result
    assert max_flow == 1.0
    assert sys_st == 'f'
    assert min_comp_state is None

def test_sys_fun4(load_toy_process_plant_dataset):

    nodes, edges, probs = load_toy_process_plant_dataset

    # Prepare input
    comps_st = {c: 1 for c in probs.keys()}
    comps_st['x5'], comps_st['x12'] = 0, 0

    # Run function
    max_flow, sys_st, min_comp_state = smf.sys_fun( comps_st, nodes, edges, probs, target_flow=1.0 )

    # Check the result
    assert max_flow == 0.5
    assert sys_st == 'f'
    assert min_comp_state is None

def test_sys_fun5(load_toy_process_plant_dataset):

    nodes, edges, probs = load_toy_process_plant_dataset

    # Prepare input
    comps_st = {c: 1 for c in probs.keys()}
    comps_st['x5'], comps_st['x12'] = 0, 0

    # Run function
    max_flow, sys_st, min_comp_state = smf.sys_fun( comps_st, nodes, edges, probs, target_flow=0.5 )

    # Check the result
    assert max_flow == 0.5
    assert sys_st == 's'
    assert isinstance(min_comp_state, dict)

def test_add_a_component1(load_toy_process_plant_dataset):

    nodes, edges, probs = load_toy_process_plant_dataset

    # Run function
    new_nodes, new_edges, new_probs = smf.add_a_component(
        'x1', nodes, edges, probs
    )

    # Check the result
    assert "x1_copy" in new_probs
    assert new_probs["x1_copy"] == probs['x1']
    assert len(new_probs) == len(probs) + 1

    assert len(new_edges) == len(edges) + 1 # one edge is connected to n1  
    assert all(
        new_nodes["n1_copy"][k] == nodes["n1"][k]
        for k in nodes["n1"]
        if k != "comp_id"
    )
    assert len(new_nodes) == len(nodes) + 1


    # Run staged max flow on the new system
    comps_st = {c: 1 for c in new_probs.keys()}
    max_flow, sys_st, min_comp_state = smf.sys_fun( comps_st, new_nodes, new_edges, new_probs, target_flow=1.0 )

    # Check the result
    assert max_flow == 1.0
    assert sys_st == 's'


    # Run staged max flow on the new system
    comps_st = {c: 1 for c in new_probs.keys()}
    comps_st['x1'] = 0  # original component fails
    max_flow, sys_st, min_comp_state = smf.sys_fun( comps_st, new_nodes, new_edges, new_probs, target_flow=1.0 )

    # Check the result: the result should be the same as the original system because of the added same component
    assert max_flow == 1.0
    assert sys_st == 's'


    # Run staged max flow on the new system
    comps_st = {c: 1 for c in new_probs.keys()}
    comps_st['x1'], comps_st['x1_copy'] = 0, 0  # both original and copied components fail
    max_flow, sys_st, min_comp_state = smf.sys_fun( comps_st, new_nodes, new_edges, new_probs, target_flow=1.0 )
    assert max_flow == 0.5
    assert sys_st == 'f'


def test_add_a_component2(load_toy_process_plant_dataset):

    nodes, edges, probs = load_toy_process_plant_dataset

    # Run function
    new_nodes, new_edges, new_probs = smf.add_a_component(
        'x11', nodes, edges, probs
    )

    # Check the result
    assert "x11_copy" in new_probs
    assert new_probs["x11_copy"] == probs['x11']
    assert len(new_probs) == len(probs) + 1

    assert len(new_edges) == len(edges) + 1 
    assert all(
        new_edges["e3_copy"][k] == edges["e3"][k]
        for k in edges["e3"]
        if k != "comp_id"
    )
    assert new_edges["e3_copy"]["comp_id"] == "x11_copy"

    assert new_nodes == nodes  # no new node is added


    # Run staged max flow on the new system
    comps_st = {c: 1 for c in new_probs.keys()}
    max_flow, sys_st, min_comp_state = smf.sys_fun( comps_st, new_nodes, new_edges, new_probs, target_flow=1.0 )
    # Check the result
    assert max_flow == 1.0
    assert sys_st == 's'


    # Run staged max flow on the new system
    comps_st = {c: 1 for c in new_probs.keys()}
    comps_st['x11'] = 0  # original component fails
    max_flow, sys_st, min_comp_state = smf.sys_fun( comps_st, new_nodes, new_edges, new_probs, target_flow=1.0 )
    # Check the result: the result should be the same as the original system because of the added same component
    assert max_flow == 1.0
    assert sys_st == 's'


    # Run staged max flow on the new system
    comps_st = {c: 1 for c in new_probs.keys()}
    comps_st['x11'], comps_st['x11_copy'] = 0, 0  # both original and copied components fail
    max_flow, sys_st, min_comp_state = smf.sys_fun( comps_st, new_nodes, new_edges, new_probs, target_flow=1.0 )
    assert max_flow == 0.0
    assert sys_st == 'f'


    # Run staged max flow on the new system
    comps_st = {c: 1 for c in new_probs.keys()}
    comps_st['x2'] = 0
    max_flow, sys_st, min_comp_state = smf.sys_fun( comps_st, new_nodes, new_edges, new_probs, target_flow=1.0 )
    # Check the result
    assert max_flow == 0.5
    assert sys_st == 'f'

def test_deactivate_a_component1( load_toy_process_plant_dataset):

    nodes, edges, probs = load_toy_process_plant_dataset

    # Run function
    new_nodes, new_edges, new_probs = smf.deactivate_a_component(
        'x1', nodes, edges, probs
    )

    # Check the result
    assert new_probs == probs

    connected_edges = ['e1']
    for eid in connected_edges:
        assert new_edges[eid]['capacity'] == 0.0
    assert len(new_edges) == len(edges) 

    assert new_nodes['n1']['capacity'] == 0.0   


    # Run staged max flow on the new system
    comps_st = {c: 1 for c in new_probs.keys()}
    max_flow, sys_st, min_comp_state = smf.sys_fun( comps_st, new_nodes, new_edges, new_probs, target_flow=1.0 )

    # Check the result
    assert max_flow == 0.5
    assert sys_st == 'f'


    # Run staged max flow on the new system
    comps_st = {c: 1 for c in new_probs.keys()}
    comps_st['x2'] = 0  # another component fails
    max_flow, sys_st, min_comp_state = smf.sys_fun( comps_st, new_nodes, new_edges, new_probs, target_flow=0.5 )
    assert max_flow == 0.0
    assert sys_st == 'f'

def test_deactivate_a_component2( load_toy_process_plant_dataset ):

    nodes, edges, probs = load_toy_process_plant_dataset

    # Run function
    new_nodes, new_edges, new_probs = smf.deactivate_a_component(
        'x11', nodes, edges, probs
    )

    # Check the result
    assert new_probs == probs

    assert new_edges["e3"]["capacity"] == 0.0

    assert nodes == new_nodes  # no node is removed


    # Run staged max flow on the new system
    comps_st = {c: 1 for c in new_probs.keys()}
    max_flow, sys_st, min_comp_state = smf.sys_fun( comps_st, new_nodes, new_edges, new_probs, target_flow=0.5 )

    # Check the result
    assert max_flow == 0.0
    assert sys_st == 'f'
