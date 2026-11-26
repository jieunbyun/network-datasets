# Provenance

This dataset represents the EMA highway benchmark network data.

## Relevant publications
- Byun, J. E., Ryu, H. & Straub, D. (2024). Branch-and-bound algorithm for efficient reliability analysis of general coherent systems. arXiv preprint arXiv:2410.22363.

## Processing Notes
- Nodes and edges were manually digitised into `nodes.json` and `edges.json`.
- `probs_bin.json` and `probs_mult.json` were generated assuming simple probability distributions over edges.

## Notes
- Coordinates in `nodes.json` are planar coordinates in kilometres (not geodetic).

## Origin of the network
The 74 nodes and 129 undirected links are the Eastern Massachusetts highway
subnetwork of the TransportationNetworks benchmark
(https://github.com/bstabler/TransportationNetworks, folder `Eastern-Massachusetts`,
retrieved 2026-10-12). Its `EMA_net.tntp` lists 258 directed links. Collapsed to
undirected pairs, these are exactly the 129 links in `edges.json` under the same
node numbering, so `nK` here is node `K` there. The benchmark comes from Zhang,
Pourazarm, Cassandras & Paschalidis (2016–2017), via
https://github.com/jingzbu/InverseVIsTraffic (for example arXiv:1610.09580).

## Georeferencing (`data/nodes_geo.json`)
Approximate WGS84 positions, added so the network can be used with geodetic
hazard inputs such as USGS ShakeMap grids. `x`/`y` in `nodes.json` are unchanged.

- **Node names:** the caption of Fig. 3 in the benchmark's `EMA_intro.pdf` names
  the town or zone of every node, e.g. {22}: Boston, {59, 60, 72}: Worcester. It
  is reproduced in `scripts/georeference_nodes.py`. Each node's zone is stored as
  `zone` in `nodes_geo.json`.
- **Control points:** the 49 nodes whose zone is a single town, in 27 towns. Each
  uses the town's OpenStreetMap Nominatim administrative-boundary point, retrieved
  2026-10-12 (`data/geo_control_points.json`). Zones given only as a state (NH,
  RI, VT, CT) or as two towns (Marlborough/Framingham, Westwood/Quincy, Cape Cod)
  are placed by the fit instead.
- **Fit:** a least-squares similarity transform (uniform scale, rotation,
  translation) from `x`/`y` to a local equirectangular km frame centred on
  42.3°N 71.4°W. Rotation is −0.1°, so the source figure is north-up. An affine
  fit predicted held-out towns slightly worse (7.7 km vs 7.3 km RMS).
- **Accuracy:** fit RMS 6.8 km. Scored by leaving out one whole town at a time,
  the held-out RMS is **7.3 km**, maximum 27.4 km (n61, Amherst, the westernmost
  node, drawn at the edge of the source map). Town centres are not interchange
  locations, so part of this error is the control points' own imprecision.
- **Every node takes its fitted position**, including control nodes. Several
  nodes share one town (Waltham has six), and using town centres would stack them
  on one point.
- **`x`/`y` are not kilometres.** The fitted scale is **1.81 real km per unit**,
  so the network spans about 140 × 137 km, not 77 × 75. Anything that reads `x`/`y`
  as km, such as the reference scenario in `scripts/cal_prob_from_eq_scenario.py`,
  computes distances about 1.8× too short for the real geography. That is fine for
  a schematic benchmark, but it matters if the scenario is meant as a real
  Massachusetts earthquake.
- **Not provided:** bridge locations. A position on a link (e.g. its midpoint) is
  an assumption.

