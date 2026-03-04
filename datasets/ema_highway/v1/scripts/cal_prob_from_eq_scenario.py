import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
from pathlib import Path
import json

# === GMPE coefficients (T = 1.0 s) ===
# for Eastern North America (Reference: K. W. Campbell, Prediction of strong ground motion using the hybrid empirical method and its use in the development of ground-motion (attenuation) relations in eastern north america, Bulletin of the Seismological Society of America 93 (2003) 1012–1033. https://doi.org/10.1785/0120020002)
GMPE_COEFFS = dict(
    c1=-0.6104,
    c2=0.451,
    c3=-0.2090,
    c4=-1.158,
    c5=-0.00255,
    c6=0.000141,
    c7=0.299,
    c8=0.503,
)

R1 = 70.0
R2 = 130.0

def gmpe_lnY(Mw, rrup, coeffs=GMPE_COEFFS):
    """Mean ln PSA (g) at T=1.0s"""
    c = coeffs

    f1 = c["c2"] * Mw + c["c3"] * (8.5 - Mw) ** 2

    R = sqrt(rrup**2 + (c["c7"] * exp(c["c8"] * Mw))**2)
    f2 = c["c4"] * log(R) + (c["c5"] + c["c6"] * Mw) * rrup

    if rrup <= R1:
        f3 = 0.0
    elif rrup <= R2:
        f3 = c["c7"] * (log(rrup) - log(R1))
    else:
        f3 = (
            c["c7"] * (log(rrup) - log(R1))
            + c["c8"] * (log(rrup) - log(R2))
        )

    return c["c1"] + f1 + f2 + f3


# Aleatory sigma (Eq. 25)
def sigma_lnY(Mw):
    c11, c12, c13, M1 = 1.110, -0.0793, 0.543, 7.16
    return c11 + c12 * Mw if Mw < M1 else c13

# Epistemic sigma (constant placeholder – replace with Table A2)
def sigma_e_lnY(Mw, rrup):
    return 0.3

def edge_midpoint(edge, nodes):
    n1, n2 = edge["from"], edge["to"]
    x = 0.5 * (nodes[n1]["x"] + nodes[n2]["x"])
    y = 0.5 * (nodes[n1]["y"] + nodes[n2]["y"])
    return x, y


def rrup_point_source(xe, ye, xs, ys):
    return sqrt((xe - xs)**2 + (ye - ys)**2)

def edge_state_probabilities(
    nodes,
    edges,
    Mw,
    xs,
    ys,
):
    """
    Compute marginal failure / survival probabilities per edge
    using edge-specific fragility parameters stored in edges dict.
    """

    out = {}

    sig_a = sigma_lnY(Mw)  # same for all edges

    for eid, e in edges.items():
        # --- geometry ---
        xe, ye = edge_midpoint(e, nodes)
        rrup = rrup_point_source(xe, ye, xs, ys)

        # --- GMPE ---
        lnY = gmpe_lnY(Mw, rrup)
        sig_e = sigma_e_lnY(Mw, rrup)

        # --- edge-specific fragility ---
        lnR = e["fragility_ln_mean"]
        beta = e["fragility_std"]

        # --- total dispersion ---
        denom = sqrt(sig_a**2 + sig_e**2 + beta**2)

        # --- failure probability ---
        z = (lnY-lnR) / denom
        p_fail = norm.cdf(z)

        out[eid] = {
            "0": {"p": float(p_fail)},
            "1": {"p": float(1.0 - p_fail)},
        }

    return out

def gm_covariance_matrix(
    nodes,
    edges,
    Mw,
    xs,
    ys,
):
    """
    Build covariance matrix Σ for edge limit-state variables
    using edge-specific fragility dispersions.
    """

    edge_ids = list(edges.keys())
    M = len(edge_ids)

    sig_a = sigma_lnY(Mw)

    # --- epistemic sigmas per edge ---
    sig_e = {}

    for eid in edge_ids:
        e = edges[eid]
        xe, ye = edge_midpoint(e, nodes)
        rrup = rrup_point_source(xe, ye, xs, ys)
        sig_e[eid] = sigma_e_lnY(Mw, rrup)

    Sigma = np.zeros((M, M))

    for i, ei in enumerate(edge_ids):
        beta_i = edges[ei]["fragility_std"]

        for j, ej in enumerate(edge_ids):
            if i == j:
                Sigma[i, j] = (
                    sig_a**2
                    + sig_e[ei]**2
                    + beta_i**2
                )
            else:
                Sigma[i, j] = sig_e[ei] * sig_e[ej]

    return edge_ids, Sigma

if __name__ == "__main__":
    # --- paths ---
    nodes_path = Path(r"datasets/ema_highway/v1/data/nodes.json")
    edges_path = Path(r"datasets/ema_highway/v1/data/edges.json")
    out_path = Path(r"datasets/ema_highway/v1/data/probs_eq.json")

    # --- earthquake scenario ---
    Mw = 8.0
    xs, ys = 1.0, 10.0   # epicentre

    # --- load network ---
    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    with open(edges_path, "r", encoding="utf-8") as f:
        edges = json.load(f)

    # --- compute failure probabilities ---
    probs = edge_state_probabilities(
        nodes=nodes,
        edges=edges,
        Mw=Mw,
        xs=xs,
        ys=ys,
    )

    # --- write output ---
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(probs, f, indent=2)

    print(f"✓ Failure probabilities written to {out_path}")
    print(f"  Scenario: Mw={Mw}, epicentre=({xs}, {ys})")