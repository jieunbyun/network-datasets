# Dataset: dist-sub-110-220kV-liang2022 (v1.0.0)

## Summary
This dataset represents a hypothetical network with 11 edges and 8 nodes, where edges can be failed. 
It can be useful for testing and demonstration purposes.

## Structure

- `data/nodes.json`  
  Dictionary of nodes with coordinates and demand.  
  Example entry:
  ```json
  "N1": { "x": 0, "y": 1, "population": 10 }
  ```
- `data/edges.json`  
  Dictionary of edges with from, to, and macrocomponent_type.  
  Example entry:
  ```json
  "E1": { "from": "N1", "to": "N2", "directed": false }
  ```
- `data/probs.json`  
  Dictionary of edge failure probabilities.  
  Example entry:
  ```json
  "E1": { "0": {"p": 0.2}, "1": {"p": 0.8} }
  ```

## Data Dictionary

### Nodes
- x, y — schematic coordinates (not geodetic).
- demand — population at the node.

### Edges
- from, to — node IDs.
- directed — whether the edge is directed.

## Usage

  ```python
  from pathlib import Path
  import json

  root = Path("toynet-11edges/v1")

  with open(root / "data" / "nodes.json", "r") as f:
      nodes = json.load(f)

  with open(root / "data" / "edges.json", "r") as f:
      edges = json.load(f)

  with open(root / "data" / "probs.json", "r") as f:
      probs = json.load(f)
  ```