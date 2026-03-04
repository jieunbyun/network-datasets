# Dataset: process-plant-toy (v1.0.0)

## Summary
A small toy process plant network.

Reference: Byun, J. and Lee, S. (2026). "A New System Function for Maximum Processible Flow in Process Plants and Application to Reliability Assessment" (in preparation)

## Structure
- `data/nodes.json`  
  Dictionary of node IDs to schematic coordinates and other necessary properties for staged maximum flow analysis. Example:
  ```json
  "n1": { "x": 0.0, "y": 0.0, "station_stage": 1, "capacity": 0.5, "comp_id": "x1" }
  ```

- `data/edges.json`
  Dictionary of undirected links with fields `from`, `to`, and `directed`, plus other necessary properties for staged maximum flow analysis. Example:
  ```json
  "e1": { "from": "n1", "to": "n2", "directed": true, "transition_start_stage": [1], "capacity": 1.0, "comp_id": "x9" }
  ```

- `data/probs.json`
    Dictionary of component states and their probabilities. Example minimal shape (per component):
    ```json
    "x9": {"0": {"p": 0.1, "remaining_capacity_ratio": 0.0}, "1": {"p": 0.9, "remaining_capacity_ratio": 1.0}}
    ```

  - `data/network.png`
    Schematic diagram of the toy process plant network.

## Notes
- This example is used in Byun and Lee (2026) to illustrate the staged maximum flow method for process plant networks.
- The data is compatible with ndtools.staged_max_flow; for usage, refer to tests/test_staged_max_flow.py.