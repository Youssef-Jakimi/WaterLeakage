"""Max-flow redirection utilities for LeakDB distribution networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx

from simulation.epanet_engine import NetworkModel, build_topology_graph


def _pipe_capacity(length: float, diameter: float, roughness: float) -> float:
    diameter_m = max(float(diameter), 1e-6) / 1000.0
    length_m = max(float(length), 1e-6)
    roughness_factor = max(float(roughness), 1e-6) / 100.0
    return (diameter_m**2) * roughness_factor * (1000.0 / length_m)


def build_flow_network(model_or_graph: NetworkModel | nx.Graph) -> nx.DiGraph:
    """Create a directed flow network with capacities derived from pipe metadata."""

    if isinstance(model_or_graph, NetworkModel):
        graph = build_topology_graph(model_or_graph)
    else:
        graph = model_or_graph

    flow_graph = nx.DiGraph()
    for node, attrs in graph.nodes(data=True):
        flow_graph.add_node(node, **attrs)

    for source, target, data in graph.edges(data=True):
        status = str(data.get("status", "Open")).lower()
        if status not in {"open", "1", "true", "yes"}:
            continue
        capacity = float(data.get("capacity", 0.0))
        if capacity <= 0:
            capacity = _pipe_capacity(
                data.get("length", 1.0),
                data.get("diameter", 1.0),
                data.get("roughness", 130.0),
            )
        flow_graph.add_edge(source, target, capacity=capacity, link_id=data.get("link_id"))
        flow_graph.add_edge(target, source, capacity=capacity, link_id=data.get("link_id"))

    return flow_graph


def _add_super_sink(graph: nx.DiGraph, sinks: Sequence[str], sink_name: str = "__super_sink__") -> str:
    graph = graph.copy()
    graph.add_node(sink_name)
    for sink in sinks:
        if sink not in graph:
            continue
        graph.add_edge(sink, sink_name, capacity=float("inf"), link_id="super_sink")
    return sink_name


def maximize_redirection(
    model_or_graph: NetworkModel | nx.Graph,
    source_node: str,
    critical_nodes: Sequence[str],
    closed_links: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Run Edmonds-Karp/Ford-Fulkerson from a source reservoir to critical sinks."""

    graph = build_flow_network(model_or_graph)
    closed_links = set(str(link) for link in (closed_links or []))

    if closed_links:
        removable_edges = [
            (u, v)
            for u, v, data in graph.edges(data=True)
            if str(data.get("link_id")) in closed_links
        ]
        graph.remove_edges_from(removable_edges)

    sink_name = "__super_sink__"
    graph.add_node(sink_name)
    for node in critical_nodes:
        if node in graph:
            graph.add_edge(node, sink_name, capacity=float("inf"))

    try:
        flow_value, flow_dict = nx.maximum_flow(graph, source_node, sink_name, flow_func=nx.algorithms.flow.edmonds_karp)
    except nx.NetworkXError:
        flow_value, flow_dict = 0.0, {}

    edge_flows: Dict[Tuple[str, str], float] = {}
    for u, adjacency in flow_dict.items():
        for v, flow in adjacency.items():
            if u == sink_name or v == sink_name:
                continue
            if flow > 0:
                edge_flows[(u, v)] = float(flow)

    return {
        "source": source_node,
        "sinks": list(critical_nodes),
        "closed_links": sorted(closed_links),
        "max_flow": float(flow_value),
        "flow_dict": flow_dict,
        "edge_flows": edge_flows,
    }


def reroute_when_isolated(
    model_or_graph: NetworkModel | nx.Graph,
    source_node: str,
    critical_nodes: Sequence[str],
    isolated_link_id: str,
) -> Dict[str, object]:
    """Convenience wrapper for a single pipe isolation event."""

    return maximize_redirection(
        model_or_graph=model_or_graph,
        source_node=source_node,
        critical_nodes=critical_nodes,
        closed_links=[isolated_link_id],
    )
