# IEEE 14-bus DC-OPF blackout model (v1)

DC optimal power flow blackout model for the IEEE 14-bus test system, as used
in Chan et al. (2024). The MATPOWER case file (`data/ieee14.m`) is the
authoritative source for line admittances, generator limits, and bus demand.

## Component model

34 failable components in total:

- **14 buses**, each modelled as a virtual edge `vbusX` joining `busX` to
  `busX_int`. Failing `vbusX` is equivalent to failing the bus itself.
- **20 branches** `br1`..`br20`.

### States

- **Generator buses (5)** — `vbus1`, `vbus2`, `vbus3`, `vbus6`, `vbus8`:
  - state 0: complete removal (p = 0.01)
  - state 1: 40% capacity (p = 0.19)
  - state 2: 80% capacity (p = 0.30)
  - state 3: full capacity (p = 0.50)
- **Ordinary buses (9)** — remaining `vbusX`: binary, p(fail) = 0.01.
- **Branches (20)** — `br1`..`br20`: binary, p(fail) = 0.01.

## Node attributes

Each outer `busX` node in `nodes.json` carries its role and its electrical
quantities, read from `data/ieee14.m`. The inner `busX_int` nodes carry only
`x`, `y` and `type: transmission`.

| `type` | generation | demand | count |
| --- | --- | --- | --- |
| `source` | yes (`capacity`) | often also yes (`demand`) | 5 |
| `output only` | no | yes (`demand`) | 8 |
| `transmission` | no | no | 1 |

- **`capacity`** — generation capacity in MW, summed over the bus's rows in
  `mpc.gen` (`PMAX`). Present on `source` nodes only.
- **`demand`** — nominal real power demand in MW, from the `PD` column of
  `mpc.bus`. Present on any bus with `PD != 0`.

`source` and `demand` are independent: 3 of the 5 generator buses (2, 3, 6)
also carry load, totalling 127.1 MW — 49% of the system's 259.0 MW. `output
only` is named to make that asymmetry explicit; it means *load and no
generation*, not *the only buses with load*.

```json
"bus2":  { "x": 7.4171, "y": 6.831,  "type": "source",      "capacity": 140.0, "demand": 21.7, "unit": "MW" },
"bus4":  { "x": 2.2939, "y": 3.6167, "type": "output only", "demand": 47.8, "unit": "MW" },
"bus7":  { "x": 6.376,  "y": 1.4211, "type": "transmission" }
```

## System function

Blackout threshold is **54.8%** of total demand (Scenario 1 in Chan et al.).
A system state of 1 (survival) means served demand ≥ 100 − 54.8 = 45.2% of nominal load.

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

See `scripts/run_case14.py` for an end-to-end example that builds the
system function and runs reliability analysis.

## Reference result

Chan et al. (2024), Table 2: P(blackout) ≈ 1.1 × 10⁻⁴ at the 54.8% threshold.
