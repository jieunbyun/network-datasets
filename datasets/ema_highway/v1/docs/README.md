# Dataset: ema-highway (v1.0.0)

## Summary
**Eastern Massachusetts (EMA) highway benchmark network** with nodes, edges, and a probability file.
This package is intended for testing network reliability / resilience algorithms (e.g., MBN/BN-based)
and path-based analyses on a mid-sized, schematic road network.

## Structure
- `data/nodes.json`  
  Dictionary of node IDs to schematic coordinates (km). Example:
  ```json
  "n1": { "x": 110.245232, "y": 139.0 }
  ```

- `data/edges.json`  
  Dictionary of undirected links with fields `from`, `to`, and `directed` (here `false`). Example:
  ```json
  "e0001": { "from": "n1", "to": "n2", "directed": false }
  ```

- `data/probs_bin.json`  
  The exact semantics depend on your analysis. Example minimal shape (per edge):
  ```json
  "e0001": {"0": {"p": 0.05}, "1": {"p": 0.95}}
  ```

- `data/probs_mult.json`
  Example minimal shape (per edge):
  ```json
  "e0001": {"0": {"p": 0.05}, "1": {"p": 0.10}, "2": {"p": 0.85}}
  ```

## Data Dictionary
### Nodes (`nodes.json`)
- `x`, `y` — schematic coordinates in kilometers (not geodetic).

### Edges (`edges.json`)
- `from`, `to` — node IDs (strings matching `nodes.json` keys).
- `directed` — boolean (this dataset uses `false`).
- `length_km` — length of edge in km, computed from node coordinates.

### Probabilities (`probs_bin.json` and `probs_mult.json`)
- Keyed by edge ID, with state/probability entries (format may vary by method).

## Usage
```python
from pathlib import Path
import json

root = Path("ema-highway/v1/data")

nodes = json.loads((root / "nodes.json").read_text("utf-8"))
edges = json.loads((root / "edges.json").read_text("utf-8"))
probs = json.loads((root / "probs.json").read_text("utf-8"))  # if used
```

## Scripts
- `scripts/draw_graph_edge_ids.py`  
  Draws the network with **edge-ID labels** (rather than node IDs) and saves it as a PNG.
  Standalone: requires only the Python standard library and matplotlib (already a core
  dependency of ndtools-duco) — no Graphviz installation. Node positions are taken
  directly from the x/y coordinates in `nodes.json`, and a collision-avoidance pass
  keeps labels readable in dense areas. Labels are bold bare numbers (`44` for edge
  `e0044`) to save space.
  ```bash
  # from the repo root; writes data/graph_edge_ids.png by default
  python datasets/ema_highway/v1/scripts/draw_graph_edge_ids.py

  # options
  python datasets/ema_highway/v1/scripts/draw_graph_edge_ids.py \
      --full-ids           # label edges e044 instead of bare numbers (44)
      --highlight n22 n66  # fill given nodes in blue (e.g., origin/destination)
      --fontsize 10        # edge label font size in pt (default 10)
      --figsize 16         # figure width/height in inches (default 16)
      --data-dir DIR       # directory containing nodes.json / edges.json
      --out FILE           # output path; extension picks the format
  ```
  The output format follows the `--out` extension: `.png` (raster), or `.svg` /
  `.pdf` for vector output. SVG keeps the labels as editable text (e.g., for
  post-editing in Inkscape).

## Notes
- Coordinates in `nodes.json` are planar coordinates in kilometres (not geodetic).
