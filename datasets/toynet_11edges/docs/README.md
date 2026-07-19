# Dataset: toynet-11edges (v1.0.0)

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

## Scripts

- `v1/scripts/draw_graph_ids.py`  
  Draws the network with **node-ID and edge-ID labels** and saves it as an image.
  Standalone: requires only the Python standard library and matplotlib (already a
  core dependency of ndtools-duco) — no Graphviz installation. Node positions are
  taken directly from the x/y coordinates in `nodes.json`. Labels are bold bare
  numbers (`4` for node `n4` / edge `e04`) to save space; node labels sit inside
  the node circles, edge labels on the edges with a collision-avoidance pass.
  ```bash
  # from the repo root; writes data/graph_ids.png by default
  python datasets/toynet_11edges/v1/scripts/draw_graph_ids.py

  # options
  python datasets/toynet_11edges/v1/scripts/draw_graph_ids.py \
      --full-ids           # label e04 / n4 instead of bare numbers (4)
      --highlight n1 n8    # fill given nodes in blue (e.g., origin/destination)
      --fontsize 18        # label font size in pt (default 18)
      --figsize 6          # figure width/height in inches (default 6)
      --node-size 720      # node marker area in points^2 (default 720)
      --data-dir DIR       # directory containing nodes.json / edges.json
      --out FILE           # output path; extension picks the format
  ```
  The output format follows the `--out` extension: `.png` (raster), or `.svg` /
  `.pdf` for vector output. SVG keeps the labels as editable text (e.g., for
  post-editing in Inkscape).