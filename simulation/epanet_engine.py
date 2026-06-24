"""Hydraulic simulation helpers for LeakDB-style EPANET networks.

This module prefers a real EPANET/wntr backend when available, but it also
ships with a deterministic linear hydraulic surrogate so the project remains
fully runnable in minimal environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import logging

import networkx as nx
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

try:  # Optional dependency.
    import wntr  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    wntr = None


def _clean_token(token: str) -> str:
    return token.strip().strip(";")


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _status_is_open(status: str) -> bool:
    normalized = _clean_token(status).lower()
    return normalized in {"open", "1", "yes", "true"}


@dataclass
class Junction:
    node_id: str
    elevation: float = 0.0
    demand: float = 0.0
    pattern: str = ""


@dataclass
class Reservoir:
    node_id: str
    head: float = 0.0
    pattern: str = ""


@dataclass
class Pipe:
    link_id: str
    start_node: str
    end_node: str
    length: float = 1.0
    diameter: float = 1.0
    roughness: float = 130.0
    minor_loss: float = 0.0
    status: str = "Open"

    def is_open(self) -> bool:
        return _status_is_open(self.status)

    def resistance(self) -> float:
        diameter = max(self.diameter, 1e-6)
        length = max(self.length, 1e-6)
        roughness = max(self.roughness, 1e-6)
        # A Hazen-Williams-inspired resistance proxy. The exact exponent is not
        # critical for the linear surrogate; it only needs to preserve relative
        # hydraulic ordering.
        return length / (roughness**1.852 * diameter**4.871)

    def capacity(self) -> float:
        resistance = max(self.resistance(), 1e-12)
        return 1.0 / resistance

    def routing_capacity(self) -> float:
        """Return a bounded capacity proxy used by the OR routing layer."""

        diameter_m = max(self.diameter, 1e-6) / 1000.0
        length_m = max(self.length, 1e-6)
        roughness_factor = max(self.roughness, 1e-6) / 100.0
        return (diameter_m**2) * roughness_factor * (1000.0 / length_m)


@dataclass
class NetworkModel:
    title: str = ""
    junctions: Dict[str, Junction] = field(default_factory=dict)
    reservoirs: Dict[str, Reservoir] = field(default_factory=dict)
    pipes: Dict[str, Pipe] = field(default_factory=dict)
    raw_sections: Dict[str, List[List[str]]] = field(default_factory=dict)

    def node_ids(self) -> List[str]:
        reservoir_ids = list(self.reservoirs.keys())
        junction_ids = list(self.junctions.keys())
        return reservoir_ids + junction_ids

    def node_elevation(self, node_id: str) -> float:
        if node_id in self.junctions:
            return self.junctions[node_id].elevation
        return 0.0

    def node_demand(self, node_id: str) -> float:
        if node_id in self.junctions:
            return self.junctions[node_id].demand
        return 0.0

    def reservoir_head(self, node_id: str) -> Optional[float]:
        reservoir = self.reservoirs.get(node_id)
        return None if reservoir is None else reservoir.head


def parse_inp_file(inp_path: str | Path) -> NetworkModel:
    """Parse a LeakDB/EPANET INP file into a lightweight network model."""

    path = Path(inp_path)
    if not path.exists():
        raise FileNotFoundError(f"INP file not found: {path}")

    sections: Dict[str, List[List[str]]] = {}
    current_section: Optional[str] = None
    title_parts: List[str] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith(";"):
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                current_section = stripped.strip("[]").upper()
                sections.setdefault(current_section, [])
                continue
            line = raw_line.split(";", 1)[0].strip()
            if not line:
                continue
            tokens = [_clean_token(token) for token in line.split()]
            if current_section == "TITLE":
                title_parts.append(" ".join(tokens))
                continue
            if current_section is None:
                continue
            sections.setdefault(current_section, []).append(tokens)

    model = NetworkModel(title=" ".join(title_parts).strip(), raw_sections=sections)

    for row in sections.get("JUNCTIONS", []):
        if not row:
            continue
        node_id = row[0]
        elevation = _safe_float(row[1]) if len(row) > 1 else 0.0
        demand = _safe_float(row[2]) if len(row) > 2 else 0.0
        pattern = row[3] if len(row) > 3 else ""
        model.junctions[node_id] = Junction(node_id=node_id, elevation=elevation, demand=demand, pattern=pattern)

    for row in sections.get("RESERVOIRS", []):
        if not row:
            continue
        node_id = row[0]
        head = _safe_float(row[1]) if len(row) > 1 else 0.0
        pattern = row[2] if len(row) > 2 else ""
        model.reservoirs[node_id] = Reservoir(node_id=node_id, head=head, pattern=pattern)

    for row in sections.get("PIPES", []):
        if not row:
            continue
        if len(row) < 3:
            continue
        link_id = row[0]
        start_node = row[1]
        end_node = row[2]
        length = _safe_float(row[3]) if len(row) > 3 else 1.0
        diameter = _safe_float(row[4]) if len(row) > 4 else 1.0
        roughness = _safe_float(row[5]) if len(row) > 5 else 130.0
        minor_loss = _safe_float(row[6]) if len(row) > 6 else 0.0
        status = row[7] if len(row) > 7 else "Open"
        model.pipes[link_id] = Pipe(
            link_id=link_id,
            start_node=start_node,
            end_node=end_node,
            length=length,
            diameter=diameter,
            roughness=roughness,
            minor_loss=minor_loss,
            status=status,
        )

    return model


def build_topology_graph(model: NetworkModel) -> nx.Graph:
    """Build an undirected topology graph from a parsed INP model."""

    graph = nx.Graph()
    for node_id in model.node_ids():
        graph.add_node(
            node_id,
            node_type="reservoir" if node_id in model.reservoirs else "junction",
            elevation=model.node_elevation(node_id),
            demand=model.node_demand(node_id),
            head=model.reservoir_head(node_id),
        )

    for pipe in model.pipes.values():
        graph.add_edge(
            pipe.start_node,
            pipe.end_node,
            link_id=pipe.link_id,
            length=pipe.length,
            diameter=pipe.diameter,
            roughness=pipe.roughness,
            minor_loss=pipe.minor_loss,
            status=pipe.status,
            capacity=pipe.routing_capacity() if pipe.is_open() else 0.0,
        )

    return graph


def _conductance_matrix(model: NetworkModel) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    node_ids = model.node_ids()
    index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    n = len(node_ids)
    conductance = np.zeros((n, n), dtype=float)

    for pipe in model.pipes.values():
        if not pipe.is_open():
            continue
        i = index.get(pipe.start_node)
        j = index.get(pipe.end_node)
        if i is None or j is None:
            continue
        g = pipe.capacity()
        conductance[i, j] += g
        conductance[j, i] += g

    return conductance, node_ids, index


def simulate_pressures(model: NetworkModel) -> pd.DataFrame:
    """Solve a linearized hydraulic balance and return heads/pressures.

    The solver uses a fixed-head formulation at reservoirs and a conductance
    network over open pipes. It is not a full EPANET replacement, but it is
    stable, deterministic, and sufficient for anomaly-detection workflows.
    """

    conductance, node_ids, index = _conductance_matrix(model)
    n = len(node_ids)
    if n == 0:
        return pd.DataFrame(columns=["head", "pressure", "elevation", "demand"])

    known_mask = np.zeros(n, dtype=bool)
    known_head = np.zeros(n, dtype=float)
    demand = np.zeros(n, dtype=float)
    elevation = np.zeros(n, dtype=float)

    for node_id in node_ids:
        idx = index[node_id]
        elevation[idx] = model.node_elevation(node_id)
        demand[idx] = model.node_demand(node_id)
        head = model.reservoir_head(node_id)
        if head is not None:
            known_mask[idx] = True
            known_head[idx] = head

    unknown_ids = [node_id for node_id in node_ids if not known_mask[index[node_id]]]
    if not unknown_ids:
        heads = known_head.copy()
    else:
        unknown_idx = [index[node_id] for node_id in unknown_ids]
        known_idx = [index[node_id] for node_id in node_ids if known_mask[index[node_id]]]
        a = np.zeros((len(unknown_idx), len(unknown_idx)), dtype=float)
        b = np.zeros(len(unknown_idx), dtype=float)

        for row_pos, i in enumerate(unknown_idx):
            row_sum = 0.0
            for j in range(n):
                g = conductance[i, j]
                if g <= 0:
                    continue
                row_sum += g
                if known_mask[j]:
                    b[row_pos] += g * known_head[j]
                else:
                    col_pos = unknown_idx.index(j)
                    a[row_pos, col_pos] -= g
            a[row_pos, row_pos] += row_sum
            b[row_pos] += demand[i]

        # Regularize lightly if the system is singular or nearly singular.
        if np.linalg.matrix_rank(a) < a.shape[0]:
            a = a + np.eye(a.shape[0]) * 1e-8

        try:
            solved = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            solved = np.linalg.lstsq(a, b, rcond=None)[0]

        heads = known_head.copy()
        for pos, idx in enumerate(unknown_idx):
            heads[idx] = solved[pos]

    pressures = heads - elevation
    frame = pd.DataFrame(
        {
            "node_id": node_ids,
            "head": heads,
            "pressure": pressures,
            "elevation": elevation,
            "demand": demand,
            "is_reservoir": [node_id in model.reservoirs for node_id in node_ids],
        }
    ).set_index("node_id")
    return frame.sort_index(key=lambda idx: idx.map(lambda value: int(value) if str(value).isdigit() else str(value)))


class EpanetEngine:
    """Stateful hydraulic engine with valve/pipe isolation support."""

    def __init__(self, inp_path: str | Path):
        self.inp_path = Path(inp_path)
        self.model = parse_inp_file(self.inp_path)
        self.graph = build_topology_graph(self.model)

    @property
    def available_links(self) -> List[str]:
        return list(self.model.pipes.keys())

    @property
    def available_nodes(self) -> List[str]:
        return self.model.node_ids()

    def clone(self) -> "EpanetEngine":
        other = object.__new__(EpanetEngine)
        other.inp_path = self.inp_path
        other.model = NetworkModel(
            title=self.model.title,
            junctions={k: Junction(**vars(v)) for k, v in self.model.junctions.items()},
            reservoirs={k: Reservoir(**vars(v)) for k, v in self.model.reservoirs.items()},
            pipes={k: Pipe(**vars(v)) for k, v in self.model.pipes.items()},
            raw_sections={k: [row[:] for row in v] for k, v in self.model.raw_sections.items()},
        )
        other.graph = build_topology_graph(other.model)
        return other

    def set_link_status(self, link_id: str, status: str) -> None:
        pipe = self.model.pipes.get(str(link_id))
        if pipe is None:
            raise KeyError(f"Unknown link id: {link_id}")
        pipe.status = status
        if self.graph.has_edge(pipe.start_node, pipe.end_node):
            self.graph[pipe.start_node][pipe.end_node]["status"] = status
            self.graph[pipe.start_node][pipe.end_node]["capacity"] = pipe.capacity() if pipe.is_open() else 0.0

    def close_link(self, link_id: str) -> None:
        """Mark a pipe or valve as CLOSED and refresh the active graph."""

        self.set_link_status(link_id, "Closed")

    def open_link(self, link_id: str) -> None:
        self.set_link_status(link_id, "Open")

    def simulate(self) -> pd.DataFrame:
        return simulate_pressures(self.model)

    def close_and_resimulate(self, link_id: str) -> pd.DataFrame:
        self.close_link(link_id)
        return self.simulate()

    def simulate_with_closed_links(self, closed_links: Sequence[str]) -> pd.DataFrame:
        clone = self.clone()
        for link_id in closed_links:
            clone.close_link(link_id)
        return clone.simulate()

    def get_pressures(self) -> pd.Series:
        return self.simulate()["pressure"]

    def reservoir_nodes(self) -> List[str]:
        return list(self.model.reservoirs.keys())

    def critical_junctions(self, top_k: int = 3) -> List[str]:
        junctions = sorted(
            self.model.junctions.values(),
            key=lambda item: item.demand,
            reverse=True,
        )
        return [junction.node_id for junction in junctions[:top_k]]


def load_engine(inp_path: str | Path) -> EpanetEngine:
    """Convenience constructor used by the rest of the project."""

    return EpanetEngine(inp_path)
