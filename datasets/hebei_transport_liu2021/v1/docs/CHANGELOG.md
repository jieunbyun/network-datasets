# Changelog

## Unreleased
- Added:
  - `data/nodes_geo.json` (approximate WGS84 positions for all 22 nodes)
  - `data/geo_control_points.json` (Wikidata coordinates of the 14 labelled cities)
  - `scripts/georeference_nodes.py` (reproduces the fit)
- Documentation:
  - `PROVENANCE.md`: georeferencing method and accuracy; full citation for Bai et al. (2021)

## v1.0.0 (2026-09-25)
- Initial release.
- Added:
  - `data/nodes.json` (22 nodes, schematic x/y digitised from Fig. 7)
  - `data/edges.json` (33 directed arcs, 3 capacity states each)
  - `data/probs.json` (i.i.d. arc distribution `p = (0.05, 0.70, 0.25)`)
  - `data/probs_case1.json` (Table 4, Case 1: `p = (0.05, 0.25, 0.70)`)
  - `data/probs_case3.json` (Table 4, Case 3: `p = (0.70, 0.05, 0.25)`)
- Documentation:
  - `README.md`
  - `PROVENANCE.md`
