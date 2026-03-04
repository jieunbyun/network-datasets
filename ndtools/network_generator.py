# ndtools/network_generator.py
from __future__ import annotations
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal, Optional, Any
import math
import networkx as nx

from ndtools.graphs import draw_graph_from_data

try:
    import jsonschema  # used in validate()
except Exception:
    jsonschema = None

DatasetVersion = Literal["v1"]

@dataclass
class GenConfig:
    name: str                   # e.g., "network_grid_5x5"
    version: DatasetVersion = "v1"
    seed: Optional[int] = 42
    description: str = ""
    generator: str = ""         # "grid" | "erdos_renyi" | ...
    generator_params: Dict = None

def _rng(seed: Optional[int]):
    r = random.Random()
    if seed is not None:
        r.seed(seed)
    return r

# Generators
def generate_grid(rows: int, cols: int) -> Tuple[List[Dict], List[Dict]]:
    """
    nodes: {"n0": {"x": int, "y": int}, ...}
    edges: {"e0": {"from": "n0", "to": "n1"}, ...}
    """
    nodes = {f"n{i}": {"x": i % cols, "y": i // cols} for i in range(rows * cols)}
    def idx(i, j): return i * cols + j
    edges = {}
    eid = 0
    for i in range(rows):
        for j in range(cols):
            u = f"n{idx(i, j)}"
            if j + 1 < cols:
                v = f"n{idx(i, j+1)}"
                edges[f"e{eid}"] = {"from": u, "to": v, "directed": False}
                eid += 1
            if i + 1 < rows:
                v = f"n{idx(i+1, j)}"
                edges[f"e{eid}"] = {"from": u, "to": v, "directed": False}
                eid += 1
    return nodes, edges

def generate_erdos_renyi(n_nodes: int, p: float, seed: Optional[int]=42) -> Tuple[List[Dict], List[Dict]]:
    r = _rng(seed)
    nodes = {f"n{i}": {"x": None, "y": None} for i in range(n_nodes)}
    edges = {}
    eid = 0
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if r.random() < p:
                edges[f"e{eid}"] = {"from": f"n{i}", "to": f"n{j}", "directed": False}
                eid += 1
    return nodes, edges

def _edges_from_nx(G: nx.Graph | nx.DiGraph) -> Dict[str, Dict[str, Any]]:
    """Convert a NetworkX graph to your edges dict format (undirected by default)."""
    edges: Dict[str, Dict[str, Any]] = {}
    eid = 0
    for u, v in G.edges():
        edges[f"e{eid}"] = {"from": f"n{u}", "to": f"n{v}", "directed": bool(G.is_directed())}
        eid += 1
    return edges

def generate_watts_strogatz(n_nodes: int, k: int, p_rewire: float, seed: Optional[int] = 42) -> Tuple[Dict, Dict]:
    """
    Watts–Strogatz small-world graph.
    nodes: {"n<i>": {"x": None, "y": None}}
    edges: {"e<i>": {"from": "n<i>", "to": "n<j>", "directed": False}}
    """
    # Ensure valid k
    if k >= n_nodes:
        k = max(2, n_nodes - (n_nodes % 2) - 1)
    if k % 2 == 1:
        k += 1
    G = nx.watts_strogatz_graph(n_nodes, k, p_rewire, seed=seed)
    nodes = {f"n{i}": {"x": None, "y": None} for i in range(n_nodes)}
    edges = _edges_from_nx(G)
    return nodes, edges

def generate_barabasi_albert(n_nodes: int, m: int, seed: Optional[int] = 42) -> Tuple[Dict, Dict]:
    """Barabási–Albert preferential attachment graph."""
    m = max(1, min(m, n_nodes - 1))
    G = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
    nodes = {f"n{i}": {"x": None, "y": None} for i in range(n_nodes)}
    edges = _edges_from_nx(G)
    return nodes, edges

def _load_deg_seq(deg_seq_arg: Optional[str], n_nodes: int) -> List[int]:
    """
    Accepts:
      - None  -> auto-generate a simple graphical sequence
      - "3,3,2,2,1" (comma-separated string)
      - "@path/to/file.txt" (one integer per line)
    """
    if deg_seq_arg is None:
        # simple Gaussian-ish, then fix parity to be graphical
        import random as _r
        _r.seed(1234)
        target_avg = 4
        degs = [max(0, int(_r.gauss(target_avg, 1.5))) for _ in range(n_nodes)]
        if sum(degs) % 2 == 1:
            degs[0] += 1
        return degs
    if deg_seq_arg.startswith("@"):
        p = Path(deg_seq_arg[1:])
        vals = [int(line.strip()) for line in p.read_text().splitlines() if line.strip()]
        return vals
    # comma-separated
    return [int(x.strip()) for x in deg_seq_arg.split(",") if x.strip()]

def _synthesize_degree_sequence(
    n: int,
    avg_deg: float,
    *,
    seed: Optional[int] = 42,
    max_tries: int = 200,
) -> List[int]:
    """
    Build a simple, graphical degree sequence near a target.
    """
    rng = random.Random(seed)

    S = max(0, int(round(n * float(avg_deg))))

    # sum must be even
    if S % 2 == 1:
        S += 1

    # aim degrees around mu, clipped to [0, n-1]
    mu = min(n - 1, max(0.0, S / n))
    for _ in range(max_tries):
        # gaussian-ish around mu
        degs = [max(0, min(n - 1, int(round(rng.gauss(mu, 1.5))))) for _ in range(n)]

        # adjust total sum to S
        cur = sum(degs)
        delta = S - cur
        # make parity even
        if delta % 2 != 0:
            # flip one degree by +/-1 within [0,n-1]
            i = rng.randrange(n)
            if degs[i] < n - 1:
                degs[i] += 1
            elif degs[i] > 0:
                degs[i] -= 1
            delta = S - sum(degs)

        # distribute the remaining difference in +/-2 steps
        step = 2 if delta > 0 else -2
        while delta != 0:
            i = rng.randrange(n)
            new_d = degs[i] + step
            if 0 <= new_d <= n - 1:
                degs[i] = new_d
                delta -= step
            # safeguard to avoid infinite loops
            if abs(delta) > 4 * n * (n - 1):
                break

        # final guards
        if sum(degs) != S:
            continue
        if any(d > n - 1 or d < 0 for d in degs):
            continue

        # use Erdős–Gallai test
        if nx.is_graphical(degs, method="eg"):
            return degs

    raise RuntimeError("Failed to synthesize a graphical degree sequence")

def generate_configuration(
    n_nodes: int,
    avg_deg: float,
    *,
    seed: Optional[int] = 42,
) -> Tuple[Dict, Dict]:
    """
    Generate a simple configuration-model graph WITHOUT passing a long deg_seq.
    Generate edges to follow avg_deg.
    Produces a simple graph (no multiedges/self-loops).
    """
    seq = _synthesize_degree_sequence(
        n_nodes, avg_deg=avg_deg, seed=seed
    )

    # Use Havel–Hakimi simple-graph generator (no multiedges/self-loops)
    G = nx.random_degree_sequence_graph(seq, seed=seed, tries=100)

    # Build nodes/edges dicts
    nodes = {f"n{i}": {"x": None, "y": None} for i in G.nodes()}
    edges: Dict[str, Dict[str, Any]] = {}
    eid = 0
    for u, v in G.edges():
        edges[f"e{eid}"] = {"from": f"n{u}", "to": f"n{v}", "directed": False}
        eid += 1

    return nodes, edges

def generate_random_geometric(n_nodes: int, radius: float, seed: Optional[int] = 42) -> Tuple[Dict, Dict]:
    """Random Geometric Graph in unit square; uses generator’s 'pos' if present."""
    G = nx.random_geometric_graph(n_nodes, radius, seed=seed)
    # Get positions; fill any missing with random
    pos = nx.get_node_attributes(G, "pos")
    if len(pos) < n_nodes:
        import random as _r
        _r.seed(seed)
        for i in range(n_nodes):
            pos.setdefault(i, (_r.random(), _r.random()))
    nodes = {f"n{i}": {"x": float(pos[i][0]), "y": float(pos[i][1])} for i in range(n_nodes)}
    edges = _edges_from_nx(G)
    return nodes, edges

# Post processing
def assign_edge_probs(edges: List[Dict], p_fail: float=0.1) -> Dict:
    """
    probs.json format example:
    {
      "components": {
        "edges": {
          "e0": {"failure": 0.1, "survival": 0.9},
          ...
        }
      }
    }
    """
    edge_probs = {eid: { "0": {"p": float(p_fail)}, "1": {"p": float(1.0 - p_fail)} } for eid in edges.keys()}
    return edge_probs

def _dataset_root(base: Path, name: str, version: DatasetVersion) -> Path:
    return base / name / version

def save_dataset(base_dir: Path, name: str, version: DatasetVersion, nodes: List[Dict], edges: List[Dict],
                 probs: Dict, description: str="", generator: str="", generator_params: Dict|None=None) -> Path:
    """
    Writes:
      <base_dir>/<name>/<version>/
        data/nodes.json
        data/edges.json
        data/probs.json
        README.md
        metadata.json
    """
    root = _dataset_root(base_dir, name, version)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    (data_dir / "nodes.json").write_text(json.dumps(nodes, indent=2))
    (data_dir / "edges.json").write_text(json.dumps(edges, indent=2))
    (data_dir / "probs.json").write_text(json.dumps(probs, indent=2))

    # README
    readme = f"""# {name}

Generated dataset ({version}).

## Description
{description}

## Files
- `data/nodes.json`
- `data/edges.json`
- `data/probs.json`

## Generator
- type: `{generator}`
- params: `{json.dumps(generator_params or {}, indent=2)}`
"""
    (root / "README.md").write_text(readme)

    # metadata
    metadata = {
        "name": name,
        "version": version,
        "description": description,
        "generator": {"type": generator, "params": generator_params or {}},
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return root

def _load_schema(schema_dir: Path, schema_name: str) -> Dict:
    with open(schema_dir / f"{schema_name}.json", "r") as f:
        return json.load(f)

def validate(dataset_root: Path, schema_dir: Path) -> None:
    """
    Validate nodes.json, edges.json, probs.json against repo schemas.
    Raises jsonschema.ValidationError if invalid.
    """
    if jsonschema is None:
        raise RuntimeError("jsonschema is not installed. `pip install jsonschema`")

    nodes = json.loads((dataset_root / "data" / "nodes.json").read_text())
    edges = json.loads((dataset_root / "data" / "edges.json").read_text())
    probs = json.loads((dataset_root / "data" / "probs.json").read_text())

    node_schema = _load_schema(schema_dir, "nodes.schema")
    edge_schema = _load_schema(schema_dir, "edges.schema")
    probs_schema = _load_schema(schema_dir, "probs.schema")

    jsonschema.validate(nodes, node_schema)
    jsonschema.validate(edges, edge_schema)
    jsonschema.validate(probs, probs_schema)

def update_registry(registry_path: Path, name: str, version: DatasetVersion, rel_path: str, meta: Dict) -> None:
    """
    Append/merge entry into registry.json.
    """
    if registry_path.exists():
        reg = json.loads(registry_path.read_text())
    else:
        reg = {"datasets": []}

    # overwrite if same name+version exists
    reg["datasets"] = [
        d for d in reg["datasets"] if not (d.get("name")==name and d.get("version")==version)
    ]
    reg["datasets"].append({
        "name": name,
        "version": version,
        "path": rel_path,
        "metadata": meta,
    })
    registry_path.write_text(json.dumps(reg, indent=2))

def generate_and_save(
    out_base: Path,
    config: GenConfig,
    update_registry_flag: bool = False,
    # Visualisation options:
    draw_graph: bool = True,
    graph_layout: str = "spring",
    graph_name: str = "graph.png",
    graph_kwargs: Optional[Dict] = None,
    # Schema validation:
    schema_dir: Optional[Path] = None,
) -> Path:
    """
    High-level: generate -> save -> validate -> registry update.
    Returns dataset root path.
    """
    params = config.generator_params or {}
    g = config.generator.lower()

    if g in ("grid", "lattice"):
        gen_params = {k: v for k, v in params.items() if k in ["rows", "cols"]}
        nodes, edges = generate_grid(**gen_params)
    elif g in ("erdos_renyi", "er"):
        gen_params = {k: v for k, v in params.items() if k in ["n_nodes", "p"]}
        nodes, edges = generate_erdos_renyi(**gen_params, seed=config.seed)
    elif g in ("watts_strogatz", "ws"):
        gen_params = {k: v for k, v in params.items() if k in ["n_nodes", "k", "p_ws"]}
        # rename p_ws -> p_rewire for function
        gen_params = {
            "n_nodes": gen_params["n_nodes"],
            "k": gen_params["k"],
            "p_rewire": gen_params["p_ws"],
        }
        nodes, edges = generate_watts_strogatz(**gen_params, seed=config.seed)

    elif g in ("barabasi_albert", "ba"):
        gen_params = {k: v for k, v in params.items() if k in ["n_nodes", "m"]}
        nodes, edges = generate_barabasi_albert(**gen_params, seed=config.seed)

    elif g in ("configuration", "config"):
        gen_params = {k: v for k, v in params.items() if k in ["n_nodes", "avg_deg"]}
        nodes, edges = generate_configuration(**gen_params, seed=config.seed)

    elif g in ("random_geometric", "rg"):
        gen_params = {k: v for k, v in params.items() if k in ["n_nodes", "radius"]}
        nodes, edges = generate_random_geometric(**gen_params, seed=config.seed)

    else:
        raise ValueError(f"Unknown generator: {config.generator}")

    if g not in ("grid","lattice"):
        params["seed"] = config.seed  # include seed in params for metadata

    probs = assign_edge_probs(edges, p_fail=float(params.get("p_fail", 0.1)))

    ds_root = save_dataset(
        out_base, config.name, config.version,
        nodes, edges, probs,
        description=config.description,
        generator=config.generator,
        generator_params=params,
    )

    if isinstance(schema_dir, Path):
        validate(ds_root, schema_dir)
    elif schema_dir is not None:
        raise Warning("schema_dir should be a Path or None. Skipping validation.")

    if draw_graph:
        data_dir = ds_root / "data"
        try:
            draw_graph_from_data(
                data_dir,
                layout=graph_layout,
                title=f"{config.name} ({config.generator})",
                output_name=graph_name,
                **(graph_kwargs or {})
            )
        except Exception as e:
            # don't fail generation just because plotting failed
            print(f"[warn] Failed to draw graph: {e}")

    if update_registry_flag:
        # Update registry.json at repo root
        repo_root = Path(__file__).resolve().parents[1]
        registry_path = repo_root / "registry.json"
        rel = str(ds_root.relative_to(repo_root))
        meta = {
            "description": config.description,
            "generator": {"type": config.generator, "params": params}
        }
        update_registry(registry_path, config.name, config.version, rel, meta)
    return ds_root

def run(argv: Optional[list[str]] = None):
    """Entry point for both CLI and programmatic calls (e.g. in Jupyter)."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate and register network datasets.")
    parser.add_argument("--type",
        choices=[
            "grid", "lattice",
            "erdos_renyi", "er",
            "watts_strogatz", "ws",
            "barabasi_albert", "ba",
            "configuration", "config",
            "random_geometric", "rg",
        ],
        required=True
    )
    parser.add_argument("--name", required=True, help="dataset folder name, e.g., network_grid_5x5")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--cols", type=int)
    parser.add_argument("--n_nodes", type=int)
    parser.add_argument("--p", type=float)
    parser.add_argument("--p_fail", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--description", default="")
    parser.add_argument("--outbase", default="generated")
    parser.add_argument("--draw_graph", type=bool, default=True)
    parser.add_argument("--k", type=int)
    parser.add_argument("--p_ws", type=float)
    parser.add_argument("--m", type=int)
    parser.add_argument("--avg_deg", type=float)
    parser.add_argument("--radius", type=float)

    args = parser.parse_args(argv)

    # --- rest of your logic unchanged ---
    g = args.type.lower()
    if g in ("grid", "lattice"):
        params = {"rows": args.rows, "cols": args.cols, "p_fail": args.p_fail}
        if params["rows"] is None or params["cols"] is None:
            raise SystemExit("--rows and --cols required for grid")
    elif g in ("erdos_renyi", "er"):
        params = {"n_nodes": args.n_nodes, "p": args.p, "p_fail": args.p_fail}
        if params["n_nodes"] is None or params["p"] is None:
            raise SystemExit("--n_nodes and --p required for erdos_renyi")
    elif g in ("watts_strogatz", "ws"):
        params = {"n_nodes": args.n_nodes, "k": args.k, "p_ws": args.p_ws, "p_fail": args.p_fail}
    elif g in ("barabasi_albert", "ba"):
        params = {"n_nodes": args.n_nodes, "m": args.m, "p_fail": args.p_fail}
    elif g in ("configuration", "config"):
        params = {"n_nodes": args.n_nodes, "avg_deg": args.avg_deg, "p_fail": args.p_fail}
    elif g in ("random_geometric", "rg"):
        params = {"n_nodes": args.n_nodes, "radius": args.radius, "p_fail": args.p_fail}
    else:
        raise SystemExit(f"Unknown --type {args.type}")

    cfg = GenConfig(
        name=args.name,
        version="v1",
        seed=args.seed,
        description=args.description,
        generator=args.type,
        generator_params=params,
    )

    repo_root = Path(__file__).resolve().parents[1]
    script_root = Path.cwd()
    out_base = (script_root / args.outbase)
    schema_dir = repo_root / "schema"

    ds_root = generate_and_save(out_base, schema_dir, cfg, draw_graph=args.draw_graph)
    print(f"Wrote dataset to: {ds_root}")


if __name__ == "__main__":
    run()
