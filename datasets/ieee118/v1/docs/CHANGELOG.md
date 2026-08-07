# Changelog

## v1.1.0 — 2026-08-07

- `nodes.json`: added a `demand` attribute (MW, from the `PD` column of
  `ieee118.m`) to every bus with non-zero demand — 99 buses, 4242.0 MW total.
- `nodes.json`: renamed the node type `output` to **`output only`**, meaning
  *load and no generation*. **Breaking** for anything matching on the old
  string.
- `nodes.json`: `capacity` now means generation capacity and nothing else.
  It previously held `PMAX` on `source` nodes but `PD` on `output` nodes — one
  key with two meanings. The demand it used to carry on load buses now lives in
  `demand`; no value was lost.
- This fixes demand at buses that both generate and consume being unrecorded:
  45 of the 54 generator buses carry 2809.0 MW between them, 66% of system
  load, previously invisible in `nodes.json` (though always present in
  `ieee118.m`, which the DC-OPF reads directly, so no computed result changes).
- `scripts/draw_graph_ids.py`: `TYPE_FACE` / `TYPE_LABEL` follow the rename.

## v1.0.0 — 2026-06-10

- Initial release.
- Full branches+buses model (304 components: 118 buses + 186 branches) with
  4-state generator buses, 2-state ordinary buses, and 2-state branches.
- Bundled MATPOWER source `ieee118.m`, DC-OPF system function
  (`sfun_dcopt.py`, `func_dcopt_py.py`), and example runner
  (`run_case118.py`).
- Mirrored from `tsum/demos/case118`.
