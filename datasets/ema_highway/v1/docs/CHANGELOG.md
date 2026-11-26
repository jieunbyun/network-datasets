# Changelog

## Unreleased
- Added `data/nodes_geo.json` (approximate WGS84 positions and source town/zone for all 74 nodes),
  `data/geo_control_points.json` (27 town centres from OpenStreetMap) and
  `scripts/georeference_nodes.py` (reproduces the fit).
- `PROVENANCE.md`: origin of the network (TransportationNetworks "Eastern-Massachusetts"),
  georeferencing method and accuracy, and the finding that x/y are ~1.81 km per unit, not km.
- Added `scripts/draw_graph_edge_ids.py`: standalone plot of the network with
  edge-ID labels (matplotlib only, no Graphviz; positions from `nodes.json`).

## v1.0.0 (2025-09-17)
- Initial release.
- Added:
  - `data/nodes.json` (schematic EMA nodes with x/y in km)
  - `data/edges.json` (undirected links, `directed=false`)
  - `data/probs_bin.json` (edge probability file - binary-state case)
  - `data/probs_mult.json` (edge probability file - multi-state case)
- Documentation:
  - `README.md`
  - `PROVENANCE.md`
