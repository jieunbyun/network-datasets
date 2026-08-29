"""Build the ACTIVSg2000 blackout dataset (Chan et al. Scenario-1 model).

Source: MATPOWER case_ACTIVSg2000.m (2000-bus synthetic Texas grid). Two
preparation steps make it match the IEEE blackout demos:

  1. Aggregate generators to one row per bus. The case has 544 gens on 485
     buses; the blackout model treats a *generator bus* as one 4-state
     component, and `CertModel` (the certificate extractor) assumes one gen
     per bus. Summing same-bus PMAX is exact for DC-OPF with zero gen cost
     and PMIN 0. Bus / branch / baseMVA blocks are kept verbatim.

  2. Generate probs.json with the standard convention:
       - generator buses : 4-state {0:.01, 1:.19, 2:.30, 3:.50}
       - ordinary buses  : 2-state {0:.01, 1:.99}
       - branches        : 2-state {0:.01, 1:.99}
     Component ids: vbus{BUS_I} for buses, br{k+1} for branch row k.

Run:  python build_dataset.py
Writes data/case_ACTIVSg2000.m (aggregated) and data/probs.json.
"""
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC_M = Path("/mnt/c/Projects/matpower8.1/data/case_ACTIVSg2000.m")
DATA = HERE.parent / "data"

from func_dcopt_py import load_case  # noqa: E402
from pypower.idx_gen import GEN_BUS, PG, QG, QMAX, QMIN, PMAX, PMIN, GEN_STATUS  # noqa: E402
from pypower.idx_bus import BUS_I  # noqa: E402


def _fmt_matrix(rows):
    return "\n".join("\t" + "\t".join(f"{v:.10g}" for v in r) + ";" for r in rows)


def aggregate_gens(ppc):
    """One gen row per bus: sum the additive columns, keep first row's rest."""
    gen = ppc["gen"]
    add_cols = [PG, QG, QMAX, QMIN, PMAX, PMIN]
    out = []
    for bus in dict.fromkeys(gen[:, GEN_BUS].astype(int)):   # preserve order
        rows = gen[gen[:, GEN_BUS].astype(int) == bus]
        agg = rows[0].copy()
        for c in add_cols:
            agg[c] = rows[:, c].sum()
        agg[GEN_STATUS] = 1
        out.append(agg)
    return np.array(out)


def main():
    ppc = load_case(SRC_M)
    n_gen0 = ppc["gen"].shape[0]
    gen_agg = aggregate_gens(ppc)
    gen_buses = sorted(int(b) for b in gen_agg[:, GEN_BUS])
    print(f"gens {n_gen0} -> {len(gen_agg)} (one per bus)")

    # ---- write aggregated .m: keep original text, swap gen + gencost ----
    text = SRC_M.read_text()
    gen_block = _fmt_matrix(gen_agg)
    text = re.sub(r"mpc\.gen\s*=\s*\[.*?\];",
                  "mpc.gen = [\n" + gen_block + "\n];", text, flags=re.DOTALL)
    # gencost: one 2-term polynomial zero-cost row per aggregated gen
    gencost_rows = [[2, 0, 0, 2, 0.0, 0.0] for _ in gen_agg]
    text = re.sub(r"mpc\.gencost\s*=\s*\[.*?\];",
                  "mpc.gencost = [\n" + _fmt_matrix(gencost_rows) + "\n];",
                  text, flags=re.DOTALL)
    out_m = DATA / "case_ACTIVSg2000.m"
    out_m.write_text(text)

    # ---- sanity: re-parse the written case ----
    ppc2 = load_case(out_m)
    assert ppc2["gen"].shape[0] == len(gen_agg)
    assert ppc2["bus"].shape[0] == ppc["bus"].shape[0]
    assert ppc2["branch"].shape[0] == ppc["branch"].shape[0]

    # ---- probs.json ----
    bus_ids = ppc2["bus"][:, BUS_I].astype(int)
    gen_bus_set = set(gen_buses)
    probs = {}
    for b in bus_ids:
        if int(b) in gen_bus_set:
            probs[f"vbus{int(b)}"] = {"0": {"p": 0.01}, "1": {"p": 0.19},
                                      "2": {"p": 0.30}, "3": {"p": 0.50}}
        else:
            probs[f"vbus{int(b)}"] = {"0": {"p": 0.01}, "1": {"p": 0.99}}
    for k in range(ppc2["branch"].shape[0]):
        probs[f"br{k + 1}"] = {"0": {"p": 0.01}, "1": {"p": 0.99}}
    json.dump(probs, open(DATA / "probs.json", "w"))
    n4 = sum(1 for v in probs.values() if len(v) == 4)
    print(f"probs.json: {len(probs)} components "
          f"({n4} gen buses, {len(bus_ids) - n4} ordinary buses, "
          f"{ppc2['branch'].shape[0]} branches)")
    print(f"wrote {out_m}\n      {DATA / 'probs.json'}")


if __name__ == "__main__":
    main()
