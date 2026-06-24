"""Lightweight report-to-isolation agent for water leakage operations."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx


PIPE_PATTERN = re.compile(r"\b(?:pipe|link|valve)\s*#?\s*(\d+)\b", re.IGNORECASE)
NODE_PATTERN = re.compile(r"\b(?:junction|node|near)\s*#?\s*(\d+)\b", re.IGNORECASE)


def build_prompt(report: str) -> str:
    return (
        "You are a water-network triage assistant. "
        "Given a citizen report, identify the most likely pipe or valve ID to isolate. "
        f"Report: {report}"
    )


def _parse_numbers(report: str) -> Tuple[Optional[int], Optional[int]]:
    pipe_match = PIPE_PATTERN.search(report)
    node_match = NODE_PATTERN.search(report)
    pipe_id = int(pipe_match.group(1)) if pipe_match else None
    node_id = int(node_match.group(1)) if node_match else None
    return pipe_id, node_id


def _incident_edges(graph: nx.Graph, node_id: int | str) -> List[Tuple[str, str, dict]]:
    node_key = str(node_id)
    if node_key not in graph:
        return []
    return [(u, v, data) for u, v, data in graph.edges(node_key, data=True)]


def recommend_isolation_action(
    report: str,
    topology: Optional[nx.Graph] = None,
    source_node: Optional[str] = None,
) -> Dict[str, object]:
    """Map a textual citizen report to the most likely isolation asset."""

    prompt = build_prompt(report)
    pipe_id, node_id = _parse_numbers(report)

    recommendation: Dict[str, object] = {
        "prompt": prompt,
        "report": report,
        "pipe_id": None,
        "node_id": node_id,
        "action": "inspect",
        "confidence": 0.35,
        "reason": "No explicit asset identifier detected in the report.",
    }

    if pipe_id is not None:
        recommendation.update(
            {
                "pipe_id": str(pipe_id),
                "action": "close_pipe",
                "confidence": 0.95,
                "reason": f"Direct pipe identifier extracted from the report: pipe {pipe_id}.",
            }
        )
        return recommendation

    if topology is not None and node_id is not None:
        node_key = str(node_id)
        candidates = _incident_edges(topology, node_key)
        if candidates:
            chosen = candidates[0]
            if source_node is not None and source_node in topology and node_key in topology:
                try:
                    source_dist = nx.shortest_path_length(topology, source_node, node_key, weight="length")
                except Exception:
                    source_dist = None
                if source_dist is not None:
                    downstream = []
                    for candidate in candidates:
                        u, v, data = candidate
                        neighbor = v if u == node_key else u
                        try:
                            neighbor_dist = nx.shortest_path_length(topology, source_node, neighbor, weight="length")
                        except Exception:
                            neighbor_dist = None
                        downstream.append((neighbor_dist, float(data.get("capacity", 1.0)), str(data.get("link_id", "")), candidate))
                    downstream.sort(key=lambda item: (
                        0 if item[0] is not None and item[0] > source_dist else 1,
                        -(item[0] if item[0] is not None else -1.0),
                        -item[1],
                        item[2],
                    ))
                    chosen = downstream[0][3]
            if chosen is candidates[0]:
                chosen = sorted(
                    candidates,
                    key=lambda item: (
                        -float(item[2].get("capacity", 1.0)),
                        str(item[2].get("link_id", "")),
                    ),
                )[0]
            link_id = chosen[2].get("link_id")
            recommendation.update(
                {
                    "pipe_id": str(link_id) if link_id is not None else None,
                    "action": "close_pipe",
                    "confidence": 0.8,
                    "reason": f"Mapped junction {node_id} to incident pipe {link_id} using the active topology.",
                }
            )
            return recommendation

    if node_id is not None and source_node is not None:
        recommendation.update(
            {
                "pipe_id": str(node_id),
                "action": "close_pipe",
                "confidence": 0.6,
                "reason": f"Fallback mapping used the reported junction {node_id} as the isolation target.",
            }
        )

    return recommendation


def parse_report_to_pipe_id(report: str, topology: Optional[nx.Graph] = None) -> Optional[str]:
    return recommend_isolation_action(report, topology).get("pipe_id")  # type: ignore[return-value]
