from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import pytest

from ndtools import staged_max_flow as smf

@pytest.fixture
def load_plant_dataset() -> tuple[dict, dict, dict]:
    """Load the toy process plant dataset for testing."""
    data_dir = Path(__file__).parent.parent / "data"
    nodes = json.loads((data_dir / "nodes.json").read_text())
    edges = json.loads((data_dir / "edges.json").read_text())
    probs = json.loads((data_dir / "probs.json").read_text())
    return nodes, edges, probs

def test_component_vector_state1(load_plant_dataset):
    
    nodes, edges, probs = load_plant_dataset
    
    # Prepare input
    comps_st = {c: 1 for c in probs.keys()}

    # Run function
    result = smf.run_staged_max_flow( comps_st, nodes, edges, probs )

    # Check the result
    assert result.x[-1] == 145.0 # system's maximum flow

def test_component_vector_state2(load_plant_dataset):
    
    nodes, edges, probs = load_plant_dataset
    
    # Prepare input
    comps_st = {c: 1 for c in probs.keys()}
    comps_st['x42'], comps_st['x1'] = 0, 0 # component failures assumed

    # Run function
    result = smf.run_staged_max_flow( comps_st, nodes, edges, probs )

    # Check the result
    assert result.x[-1] == 90.0 # system's maximum flow

def test_component_vector_state3(load_plant_dataset):
    
    nodes, edges, probs = load_plant_dataset
    
    # Prepare input
    comps_st = {c: 1 for c in probs.keys()}
    comps_st['x42'], comps_st['x39'] = 0, 0 # component failures assumed

    # Run function
    result = smf.run_staged_max_flow( comps_st, nodes, edges, probs )

    # Check the result
    assert result.x[-1] == 0.0 # system's maximum flow

def test_component_vector_state4(load_plant_dataset):
    
    nodes, edges, probs = load_plant_dataset
    
    # Prepare input
    comps_st = {c: 1 for c in probs.keys()}
    comps_st['x3'], comps_st['x4'] = 0, 0 # component failures assumed

    # Run function
    result = smf.run_staged_max_flow( comps_st, nodes, edges, probs )

    # Check the result
    assert result.x[-1] == 55.0 # system's maximum flow