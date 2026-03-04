Network Datasets Documentation
==============================

A curated collection of example infrastructure network datasets to support research on network reliability, risk, resilience, and uncertainty quantification.

The repository serves two purposes (cf Figure 1): (1) to provide a standardised collection of benchmark network datasets, and (2) to offer ready-to-use system performance functions compatible with network data for probabilistic assessments.

.. figure:: _static/nd_repo_intro.png
   :alt: Network Datasets overview
   :align: center
   :width: 70%

   *Figure 1.* Overview of the Network Datasets repository.

The datasets are designed for use with `MBNpy <https://jieunbyun.github.io/MBNpy-docs>`_ but can also be loaded independently.

Overview
--------

The Network Datasets repository contains:

* **Structured datasets**:  
  Infrastructure networks with nodes, edges, probabilities, and optional metadata, each accompanied by illustrative examples 
  | **JSON Schemas**: Validation schemas that ensure data consistency
* **ndtools**: Python utilities for loading, graph building, and network analysis

Repository Structure
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ├─ registry.json              # Index of available datasets
   ├─ schema/                    # JSON Schemas for validation
   ├─ <dataset folders>/         # e.g. distribution-substation-liang2022/
   ├─ ndtools/                   # Utility functions for loading and analysis
   ├─ tests/                     # Unit tests
   └─ LICENSE                    # Licensing (MIT for code, CC-BY-4.0 for data)

Key Features
~~~~~~~~~~~~

* **Standardised Format**: All datasets follow consistent JSON schemas
* **NetworkX Integration**: Easy conversion to NetworkX graphs
* **Validation Tools**: Built-in data validation against schemas
* **Visualisation**: Graph drawing utilities with multiple layout options
* **Analysis Functions**: Connectivity, shortest path, substation capacity, and more
* **Multiple Domains**: Power grids, transportation, and other infrastructure

Quick Start
~~~~~~~~~~~

Install the package:

.. code-block:: bash

   pip install -e .

Load a dataset:

.. code-block:: python

   from ndtools.io import dataset_paths, load_json
   from ndtools.graphs import build_graph
   from pathlib import Path

   # Get dataset paths
   nodes_path, edges_path, probs_path = dataset_paths(Path('.'), 'toynet-11edges', 'v1')
   
   # Load data
   nodes = load_json(nodes_path)
   edges = load_json(edges_path)
   probs = load_json(probs_path)
   
   # Build NetworkX graph
   G = build_graph(nodes, edges, probs)

License
-------

* **Code** (scripts, validators): MIT License
* **Data** (datasets): CC-BY-4.0 License

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   datasets
   ndtools
   generating networks
   examples
   how to contribute
   contributors

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
