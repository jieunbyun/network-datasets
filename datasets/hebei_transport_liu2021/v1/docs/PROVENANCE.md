# Provenance

This dataset is the Hebei transportation multistate network used as the third
(largest) benchmark in Liu et al. (2021).

## Relevant publications
- Liu, T., Bai, G., Tao, J., Zhang, Y.-A., & Fang, Y. (2021). An improved bounding
  algorithm for approximating multistate network reliability based on state-space
  decomposition method. *Reliability Engineering & System Safety*, 210, 107500.
  https://doi.org/10.1016/j.ress.2021.107500
- The network is originally attributed by that paper to its reference [37]:
  Bai, G., Xu, B., Chen, X., Zhang, Y., & Tao, J. (2021). Searching for d-MPs for
  all level d in multistate two-terminal networks without duplicates. *IEEE
  Transactions on Reliability*, 70(1), 319–330 (early access 2020).
  https://doi.org/10.1109/TR.2020.3004971 — the network is its Example 3, drawn
  on a map of Hebei in Fig. 12.

## Processing notes
- The paper distributes no machine-readable data. Nodes and arcs were digitised
  from **Fig. 7** of Liu et al. (2021) (p. 8), which draws the 22 numbered nodes
  and the 33 labelled arcs `a1`–`a33` with their directions.
- Node coordinates are the centres of the drawn node circles, taken from a
  high-resolution render of the figure and normalised to unitless figure
  coordinates (figure height = 100, aspect preserved, `y` increasing northwards).
  They are schematic positions on the provincial map, **not** geodetic coordinates.
- Arc directions follow the arrowheads in the figure. Arc IDs `e01`–`e33`
  correspond one-to-one to the paper's `a1`–`a33` and are also stored in the
  `label` field of each edge.
- State distributions are taken verbatim from the paper: Section 4.1 uses
  `p = (0.05, 0.70, 0.25)` for all i.i.d. components (`probs.json`), and Table 4
  lists two further cases used in the stability study, `(0.05, 0.25, 0.70)`
  (`probs_case1.json`) and `(0.70, 0.05, 0.25)` (`probs_case3.json`).

## Verification
- The digitised arc list gives **3596** d-MPs for `d = 3` and **5665** for `d = 4`,
  matching the counts stated in Section 4.1 of the paper exactly.
- The graph is a DAG, every node lies on an s–t path, and the maximum flow with
  all arcs at capacity 2 is 6 — consistent with the demands `d = 3` and `d = 4`
  studied in the paper.

## Georeferencing (`data/nodes_geo.json`)
Approximate WGS84 positions, added so the network can be used with geodetic
hazard inputs such as USGS ShakeMap grids. `x`/`y` in `nodes.json` are unchanged.

- Both source figures (Liu et al. 2021 Fig. 7; Bai et al. 2021 Fig. 12) draw the
  nodes on a map of Hebei's prefecture boundaries, without a graticule, scale
  bar or stated projection.
- **Control points:** the 14 labelled cities, each with the coordinate location
  (P625) of its Wikidata item, retrieved 2026-10-12 (`data/geo_control_points.json`).
  These are city-centre points, not specific facilities.
- **Fit:** a least-squares similarity transform (uniform scale, rotation,
  translation) from figure units to a local equirectangular km frame centred on
  38.8°N 116.5°E (`scripts/georeference_nodes.py`). Scale 7.47 km per figure
  unit, rotation −2.2°. An affine fit was tried and predicted held-out cities
  slightly worse (leave-one-out RMS 23.8 km vs 22.1 km), so the extra freedom
  was not used.
- **Accuracy:** fit RMS 19.0 km; leave-one-out RMS **22.1 km**, maximum 38.9 km
  (XT). No single city dominates, including the inferred BZ and HH, so the error
  is the drawing's placement scatter rather than a misidentified city.
- **Labelled nodes** take the city's own coordinate; **the 8 junction nodes**
  (6, 9, 13, 16, 17, 19, 20, 21) take the fitted position and should be treated
  as uncertain by about ±22 km. All 8 fall inside Hebei, in counties consistent
  with their arcs (e.g. n13 in Hejian, between Baoding and Hengshui). Every node
  also carries `fitted_lat`/`fitted_lon` for comparison.
- **Not provided:** road geometry and bridge locations. Arcs are straight lines in
  both figures, so any position on an arc (e.g. its midpoint) is an assumption.
