"""Minimum spanning tree utilities for degraded water network operation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

from simulation.epanet_engine import NetworkModel, build_topology_graph


def _mst_weight(data: dict) -> float:
    length = float(data.get("length", 1.0))
    diameter = max(float(data.get("diameter", 1.0)), 1e-6)
    roughness = max(float(data.get("roughness", 130.0)), 1e-6)
    # Lower weights should correspond to cheaper structural pumping routes.
    return length / (diameter * roughness)


def build_active_graph(model_or_graph: NetworkModel | nx.Graph) -> nx.Graph:
    if isinstance(model_or_graph, NetworkModel):
        graph = build_topology_graph(model_or_graph)
    else:
        graph = model_or_graph.copy()

    active = nx.Graph()
    for node, attrs in graph.nodes(data=True):
        active.add_node(node, **attrs)

    for source, target, data in graph.edges(data=True):
        status = str(data.get("status", "Open")).lower()
        if status not in {"open", "1", "true", "yes"}:
            continue
        active.add_edge(source, target, weight=_mst_weight(data), **data)

    return active


def minimum_spanning_tree(model_or_graph: NetworkModel | nx.Graph) -> nx.Graph:
    """Return the MST of the active network using Prim's algorithm."""

    active_graph = build_active_graph(model_or_graph)
    if active_graph.number_of_edges() == 0:
        return active_graph
    return nx.minimum_spanning_tree(active_graph, algorithm="prim", weight="weight")


def mst_summary(model_or_graph: NetworkModel | nx.Graph) -> dict:
    tree = minimum_spanning_tree(model_or_graph)
    total_cost = 0.0
    edges: List[Tuple[str, str, float]] = []
    for u, v, data in tree.edges(data=True):
        weight = float(data.get("weight", 0.0))
        total_cost += weight
        edges.append((u, v, weight))
    return {
        "num_nodes": tree.number_of_nodes(),
        "num_edges": tree.number_of_edges(),
        "total_cost": total_cost,
        "edges": edges,
        "tree": tree,
    }
