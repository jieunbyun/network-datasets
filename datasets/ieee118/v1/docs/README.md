# IEEE 118-bus DC-OPF blackout model (v1)

DC optimal power flow blackout model for the IEEE 118-bus test system, as used
in Chan et al. (2024). The MATPOWER case file (`data/ieee118.m`) is the
authoritative source for line admittances, generator limits, and bus demand.

## Component model

304 failable components in total:

- **118 buses**, each modelled as a virtual edge `vbusX` joining `busX` to
  `busX_int`. Failing `vbusX` is equivalent to failing the bus itself.
- **186 branches** `br1`..`br186`.

### States

- **Generator buses (54)** — 4-state:
  - state 0: complete removal (p = 0.01)
  - state 1: 40% capacity (p = 0.19)
  - state 2: 80% capacity (p = 0.30)
  - state 3: full capacity (p = 0.50)
- **Ordinary buses (64)** — binary, p(fail) = 0.01.
- **Branches (186)** — `br1`..`br186`: binary, p(fail) = 0.01.

## Node attributes

Each outer `busX` node in `nodes.json` carries its role and its electrical
quantities, read from `data/ieee118.m`. The inner `busX_int` nodes carry only
`x`, `y` and `type: transmission`.

| `type` | generation | demand | count |
| --- | --- | --- | --- |
| `source` | yes (`capacity`) | often also yes (`demand`) | 54 |
| `output only` | no | yes (`demand`) | 54 |
| `transmission` | no | no | 10 |

- **`capacity`** — generation capacity in MW, summed over the bus's rows in
  `mpc.gen` (`PMAX`). Present on `source` nodes only.
- **`demand`** — nominal real power demand in MW, from the `PD` column of
  `mpc.bus`. Present on any bus with `PD != 0` (99 of the 118 buses).

`source` and `demand` are independent: 45 of the 54 generator buses also carry
load, totalling 2809.0 MW — 66% of the system's 4242.0 MW. `output only` is
named to make that asymmetry explicit; it means *load and no generation*, not
*the only buses with load*.

```json
"bus1":  { "x": 8.2258, "y": 1.1481, "type": "source",      "capacity": 100.0, "demand": 51.0, "unit": "MW" },
"bus2":  { "x": 8.9209, "y": 1.3787, "type": "output only", "demand": 20.0, "unit": "MW" },
"bus5":  { "x": 9.5203, "y": 2.9532, "type": "transmission" }
```

## System function

Blackout threshold is **13.8%** of total demand (Scenario 1 in Chan et al.).
A system state of 1 (survival) means served demand ≥ 100 − 13.8 = 86.2% of nominal load.

`scripts/sfun_dcopt.py` exposes `make_dcopt_sfun(case_path, blackout_threshold,
alpha)` which returns a callable `sfun(component_states) -> (blackout_pct,
sys_state, info)` using the MATPOWER case as the network model.

## How to load

```python
import json
from pathlib import Path

data = Path("v1/data")
edges = json.load(open(data / "edges.json"))
nodes = json.load(open(data / "nodes.json"))
probs = json.load(open(data / "probs.json"))
```

See `scripts/run_case118.py` for an end-to-end example that builds the
system function and runs reliability analysis.

## Reference result

Chan et al. (2024), Table 2: P(blackout) ≈ 1.0 × 10⁻⁴ at the 13.8% threshold.
