ndtools Package
===============

The ``ndtools`` package provides utilities for loading, processing, and analysing network datasets. It consists of several modules, each serving a specific purpose in the network analysis workflow.

Package Overview
----------------

.. code-block:: python

   import ndtools
   print(ndtools.__version__)  # 0.1.0

The package is organised into the following modules:

* :doc:`ndtools.io` - Data loading and I/O utilities
* :doc:`ndtools.graphs` - Graph construction and visualization
* :doc:`ndtools.fun_binary_graph` - Evaluation of systems with binary-state components

IO Module
---------

The ``io`` module provides functions for loading datasets and handling file I/O operations.

.. automodule:: ndtools.io
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
~~~~~~~~~~~~~

``load_json(path)``
   Load a JSON file and return its contents as a Python dictionary.

``load_yaml(path)``
   Load a YAML file and return its contents.

``dataset_paths(repo_root, dataset, version)``
   Get the file paths for a specific dataset's nodes, edges, and probabilities files.

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from ndtools.io import dataset_paths, load_json
   from pathlib import Path

   # Get dataset file paths
   nodes_path, edges_path, probs_path = dataset_paths(
       Path('.'), 'toynet-11edges', 'v1'
   )
   
   # Load the data
   nodes = load_json(nodes_path)
   edges = load_json(edges_path)
   probs = load_json(probs_path)

Graphs Module
-------------

The ``graphs`` module provides functions for constructing NetworkX graphs and visualizing networks.

.. automodule:: ndtools.graphs
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
~~~~~~~~~~~~~

``build_graph(nodes, edges, probs)``
   Construct a NetworkX graph from node, edge, and probability data.

``draw_graph_from_data(data_dir, ...)``
   Load data from files and create a visualization of the network.

``compute_edge_lengths(nodes_dict, edges_dict)``
   Calculate Euclidean distances for edges based on node coordinates.

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from ndtools.graphs import build_graph, draw_graph_from_data
   from ndtools.io import dataset_paths, load_json
   from pathlib import Path

   # Load data
   nodes_path, edges_path, probs_path = dataset_paths(Path('.'), 'toynet-11edges', 'v1')
   nodes = load_json(nodes_path)
   edges = load_json(edges_path)
   probs = load_json(probs_path)
   
   # Build graph
   G = build_graph(nodes, edges, probs)
   
   # Visualize
   output_path = draw_graph_from_data(
       "toynet-11edges/v1/data",
       layout="spring",
       with_node_labels=True,
       title="Toy Network"
   )

Binary Graph Functions Module
-----------------------------

The ``fun_binary_graph`` module provides functions for evaluating performance of system events, given binary states of components.

.. automodule:: ndtools.fun_binary_graph
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from ndtools.fun_binary_graph import eval_global_conn_k
   from ndtools.graphs import build_graph
   from ndtools.io import dataset_paths, load_json
   from pathlib import Path

   # Load and build graph
   nodes_path, edges_path, probs_path = dataset_paths(Path('.'), 'toynet-11edges', 'v1')
   nodes = load_json(nodes_path)
   edges = load_json(edges_path)
   probs = load_json(probs_path)
   G = build_graph(nodes, edges, probs)
   
   # Define component states (1 = working, 0 = failed)
   comps_state = {
       "n1": 1,  # Node n1 is working
       "n2": 1,  # Node n2 is working
       "e1": 0,  # Edge e1 has failed
       "e2": 1,  # Edge e2 is working
   }
   
   # Evaluate connectivity
   k_value, status, _ = eval_global_conn_k(comps_state, G)
   print(f"Connectivity: {k_value}, Status: {status}")

Dependencies
------------

The ndtools package requires the following dependencies:

* **networkx** (>=3.0): Graph analysis and manipulation
* **pyyaml** (>=6.0): YAML file parsing
* **jsonschema** (>=4.0): JSON schema validation

Optional dependencies:

* **matplotlib** (>=3.0): For graph visualization functions

Installation
~~~~~~~~~~~

Install the package and its dependencies:

.. code-block:: bash

   pip install -e .

Or install with optional visualization dependencies:

.. code-block:: bash

   pip install -e ".[viz]"

Testing
-------

Run the test suite to verify functionality:

.. code-block:: bash

   pytest tests/

The test suite includes:

* Unit tests for all major functions
* Integration tests for complete workflows
* Data validation tests

Performance Notes
-----------------

* Graph construction is optimized for large networks
* Visualisation functions use efficient layout algorithms
* System function evaluation supports both directed and undirected graphs
* Memory usage scales linearly with network size
