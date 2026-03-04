Quick Start Guide
=================

This guide will get you up and running with the Network Datasets package in just a few minutes.

Basic Usage
-----------

Loading a Dataset
~~~~~~~~~~~~~~~~~

The simplest way to load a dataset is using the provided utility functions:

.. code-block:: python

   from ndtools.io import dataset_paths, load_json
   from ndtools.graphs import build_graph
   from pathlib import Path

   # Get paths to dataset files
   nodes_path, edges_path, probs_path = dataset_paths(
       Path('.'), 'toynet_11edges', 'v1'
   )
   
   # Load the data
   nodes = load_json(nodes_path)
   edges = load_json(edges_path)
   probs = load_json(probs_path)
   
   # Build a NetworkX graph
   G = build_graph(nodes, edges, probs)
   
   print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

Direct Loading
~~~~~~~~~~~~~~

You can also load datasets directly using standard Python:

.. code-block:: python

   import json
   from pathlib import Path

   # Load a specific dataset
   root = Path("toynet_11edges/v1") 
   
   nodes = json.loads((root/"data/nodes.json").read_text())
   edges = json.loads((root/"data/edges.json").read_text())
   probs = json.loads((root/"data/probs.json").read_text())

Visualising Networks
--------------------

Draw a Network Graph
~~~~~~~~~~~~~~~~~~~~

Use the built-in visualization function to create network diagrams:

.. code-block:: python

   from ndtools.graphs import draw_graph_from_data
   from pathlib import Path

   # Draw the graph and save as PNG
   output_path = draw_graph_from_data(
       "toynet_11edges/v1/data",
       layout="spring",
       with_node_labels=True,
       title="Toy Network Example"
   )
   
   print(f"Graph saved to: {output_path}")

Customising Visualisation
~~~~~~~~~~~~~~~~~~~~~~~~~

The visualisation function offers many customisation options:

.. code-block:: python

   output_path = draw_graph_from_data(
       "ema_highway/v1/data",
       layout="kamada_kawai",        # Layout algorithm
       node_color="lightblue",       # Node color
       node_size=800,               # Node size
       edge_color="gray",           # Edge color
       with_node_labels=True,       # Show node labels
       with_edge_labels=True,       # Show edge labels
       title="Highway Network",     # Plot title
       output_name="highway_network.png"
   )

Available Layouts
~~~~~~~~~~~~~~~~~

* ``spring`` - Force-directed layout (default)
* ``kamada_kawai`` - Spring layout with better convergence
* ``circular`` - Nodes arranged in a circle
* ``shell`` - Nodes arranged in concentric circles

Data Validation
---------------

Validate Dataset Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure your datasets conform to the expected schemas:

.. code-block:: bash

   # Validate all datasets
   python data_validate.py --root .

   # Validate a specific dataset
   python data_validate.py --root . --dataset toynet_11edges

Programmatic Validation
~~~~~~~~~~~~~~~~~~~~~~~

You can also validate data programmatically:

.. code-block:: python

   import jsonschema
   from ndtools.io import load_json
   from pathlib import Path

   # Load schema
   with open("schema/nodes.schema.json") as f:
       nodes_schema = json.load(f)
   
   # Load and validate data
   nodes = load_json(Path("toynet-11edges/v1/data/nodes.json"))
   jsonschema.validate(nodes, nodes_schema)
   print("Nodes data is valid!")

Network Analysis
----------------

Basic Graph Properties
~~~~~~~~~~~~~~~~~~~~~~

Once you have a NetworkX graph, you can perform various analyses:

.. code-block:: python

   import networkx as nx

   # Basic properties
   print(f"Number of nodes: {G.number_of_nodes()}")
   print(f"Number of edges: {G.number_of_edges()}")
   print(f"Is connected: {nx.is_connected(G)}")
   print(f"Number of components: {nx.number_connected_components(G)}")

   # Node connectivity
   if nx.is_connected(G):
       print(f"Node connectivity: {nx.node_connectivity(G)}")

Edge Lengths
~~~~~~~~~~~

Calculate Euclidean distances for edges with spatial coordinates:

.. code-block:: python

   from ndtools.graphs import compute_edge_lengths

   # Compute edge lengths (requires x, y coordinates in nodes)
   lengths = compute_edge_lengths(nodes, edges)
   
   for edge_id, length in lengths.items():
       print(f"Edge {edge_id}: {length:.2f} km")

System Performance Evaluation
-------------------------------

The package includes functions for evaluating system performance under different failure scenarios:

.. code-block:: python

   from ndtools.fun_binary_graph import eval_global_conn_k

   # Define component states (1 = working, 0 = failed)
   comps_state = {
       "n1": 1,  # Node n1 is working
       "n2": 1,  # Node n2 is working
       "e1": 0,  # Edge e1 has failed
       "e2": 1,  # Edge e2 is working
   }
   
   # Evaluate global connectivity
   k_value, state, _ = eval_global_conn_k(comps_state, Guide)
   print(f"Connectivity: {k_value}, System state: {state}")

Next Steps
----------

* Explore the :doc:`datasets` page to see available datasets
* Check out the :doc:`ndtools` API reference for detailed function documentation
* Look at the :doc:`examples` for more complex usage scenarios
* Learn about :doc:`contributing` to add your own datasets
