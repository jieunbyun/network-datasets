# Network Datasets

A curated collection of example infrastructure network datasets for reliability and resilience research.

This repository provides structured datasets (nodes, edges, probabilities, and optional metadata) along with JSON Schemas for validation.  
The datasets are designed for use with [MBNpy](https://github.com/jieunbyun/mbnpy) but can also be loaded directly with Python.

---

## Repository structure
```
├─ registry.json # Index of available datasets
├─ schema/ # JSON Schemas for validation
├─ <dataset folders>/ # e.g. distribution-substation-liang2022/, ...
├─ ndtools/ # Utility functions for loading, graph building, and general network functions
├─ tests/ # Unit tests
└─ LICENSE # Licensing (MIT for code, CC-BY-4.0 for data)
```

---

## Installation

The `ndtools` utilities can be installed either as a **released package** from PyPI
or in **editable (developer) mode** from the source repository.

### Using pip (recommended for users)

Install the latest released version from PyPI:

```bash
pip install ndtools-duco
```

This installs a stable version of the tools suitable for general use.

Verify the installation:

```bash
python -c "import ndtools; print(ndtools.__version__)"
```

---

### Using pip (editable / developer install)

If you are developing `ndtools` or working directly with the source code,
install in editable mode from the repository root:

```bash
# Activate your environment first (if using conda)
conda activate <your-env>

# Install in editable mode
pip install -e .
```

---

### Run tests (developers)
```bash
pytest -q
```

---

## Documentation

📚 **Comprehensive documentation** is available at: [https://jieunbyun.github.io/network-datasets/](https://jieunbyun.github.io/network-datasets/)

The documentation includes:
- **Overview** - Introduction to the repository and its purpose
- **Installation Instructions** - Creating a Python environment and pip-installing ndtools
- **Quick Start Guide** - Get up and running in minutes
- **Available Datasets** - Detailed information about each dataset
- **ndtools Pacakage** - Complete documentation of the ndtools package
- **Examples** - Jupyter notebooks and code examples
- **Contributing Guide** - How to add new datasets and contribute

### Building Documentation Locally

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build the documentation
cd docs
make html

# View the documentation
open _build/html/index.html
```

Or use the provided build script:
```bash
# On Linux/Mac
python build_docs.py

# On Windows
build_docs.bat
```

## Usage

### Example load directly in Python
```python
import json
from pathlib import Path

# Example dataset
root = Path("toynet_11edges/v1") 

nodes = json.loads((root/"data/nodes.json").read_text())
edges = json.loads((root/"data/edges.json").read_text())
probs = json.loads((root/"data/probs.json").read_text())
```

### Using ndtools (recommended)
```python
from ndtools.io import dataset_paths, load_json
from ndtools.graphs import build_graph
from pathlib import Path

# Load dataset
nodes_path, edges_path, probs_path = dataset_paths(Path('.'), 'toynet_11edges', 'v1')
nodes = load_json(nodes_path)
edges = load_json(edges_path)
probs = load_json(probs_path)

# Build NetworkX graph
G = build_graph(nodes, edges, probs)
```

### Validate against schemas
```python
pip install jsonschema
python data_validate.py --root . # Check all repos
python data_validate.py --root . --dataset toynet_11edges # Check specific dataset
```

## License
- Code (scripts, validators): MIT License
- Data (datasets): CC-BY-4.0 License
