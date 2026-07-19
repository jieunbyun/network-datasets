# Changelog

## Unreleased
- Added `v1/scripts/draw_graph_ids.py`: standalone plot of the network with
  node-ID and edge-ID labels (matplotlib only, no Graphviz; positions from
  `nodes.json`).
- Fixed dataset name in the `README.md` title (was copied from another dataset).

## v1.0.0 (2025-09-02)
- Initial release.
- Added:
  - `nodes.json` 
  - `edges.json` 
  - `probs.json` with edge failure probabilities.
- Documentation:
  - Dataset card (`README.md`)
  - Provenance (`PROVENANCE.md`)

## v1.0.1 (2025-12-15)
- Updated `nodes.json` to include `population` attribute for each node.
- Updated documentation in `README.md` to reflect changes in `nodes.json`.