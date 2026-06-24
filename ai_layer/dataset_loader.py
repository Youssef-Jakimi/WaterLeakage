"""LeakDB dataset loading utilities.

The loader combines topology from the INP file with the temporal pressure and
leakage series stored in the benchmark folders. It returns a PyG Temporal-style
static graph temporal signal when the dependency is available, otherwise a
compatible lightweight fallback container.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import logging
import re

import networkx as nx
import numpy as np
import pandas as pd

from simulation.epanet_engine import NetworkModel, build_topology_graph, parse_inp_file

LOGGER = logging.getLogger(__name__)

try:  # Optional dependency.
    from torch_geometric_temporal.signal import StaticGraphTemporalSignal as PyGStaticGraphTemporalSignal  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PyGStaticGraphTemporalSignal = None


def _natural_node_key(value: str) -> Tuple[int, str]:
    value = str(value)
    if value.isdigit():
        return int(value), value
    match = re.search(r"\d+", value)
    if match:
        return int(match.group()), value
    return (10**9, value)


def _read_csv_time_series(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if frame.shape[1] < 2:
        raise ValueError(f"Unexpected time-series format in {csv_path}")
    timestamp_col = frame.columns[0]
    value_col = frame.columns[1]
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col])
    frame = frame.rename(columns={timestamp_col: "Timestamp", value_col: "Value"})
    return frame[["Timestamp", "Value"]]


def _maybe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _maybe_int(value: object, default: int = -1) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except Exception:
        return default


def _safe_to_datetime(value: object) -> Optional[pd.Timestamp]:
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


@dataclass
class StaticGraphTemporalSignal:
    """Fallback container with a PyG Temporal-like surface area."""

    edge_index: np.ndarray
    edge_weight: np.ndarray
    features: List[np.ndarray]
    targets: List[np.ndarray]

    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        for feature, target in zip(self.features, self.targets):
            yield feature, self.edge_index, self.edge_weight, target

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.features[index], self.edge_index, self.edge_weight, self.targets[index]


@dataclass
class ScenarioBundle:
    """Container for one LeakDB scenario and its parsed metadata."""

    scenario_name: str
    signal: object
    metadata: Dict[str, object]


class LeakDBDatasetLoader:
    """Loader for LeakDB benchmark folders."""

    def __init__(self, raw_root: str | Path):
        self.raw_root = Path(raw_root)
        if not self.raw_root.exists():
            raise FileNotFoundError(f"Raw data root not found: {self.raw_root}")

    def find_benchmark_root(self, benchmark_name: str = "Hanoi_CMH") -> Path:
        candidates: List[Path] = []
        direct_candidates = [
            self.raw_root / "LeakDB" / benchmark_name,
            self.raw_root / benchmark_name,
            self.raw_root / "LeakDB" / f"{benchmark_name}_1000scenarios" / benchmark_name,
            self.raw_root / "LeakDB" / "LeakDB" / f"{benchmark_name}_1000scenarios" / benchmark_name,
        ]
        for candidate in direct_candidates:
            if candidate.exists() and candidate.is_dir():
                candidates.append(candidate)

        unique_candidates: List[Path] = []
        seen = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_candidates.append(candidate)

        scored: List[Tuple[int, Path]] = []
        for candidate in unique_candidates:
            scenario_count = len([path for path in candidate.iterdir() if path.is_dir() and path.name.lower().startswith("scenario-")])
            if scenario_count > 0:
                scored.append((scenario_count, candidate))

        if scored:
            scored.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
            return scored[0][1]

        raise FileNotFoundError(f"Unable to locate LeakDB benchmark directory under {self.raw_root}")

    def find_inp_path(self, benchmark_name: str = "Hanoi_CMH", scenario_name: Optional[str] = None) -> Path:
        root = self.find_benchmark_root(benchmark_name)
        if scenario_name:
            scenario_path = root / scenario_name
            if scenario_path.exists():
                inp_candidates = list(scenario_path.glob("*.inp"))
                if inp_candidates:
                    return inp_candidates[0]
        inp_candidates = list(root.glob("*_nominal.inp"))
        if inp_candidates:
            return inp_candidates[0]
        inp_candidates = list(root.rglob("*.inp"))
        if inp_candidates:
            return inp_candidates[0]
        raise FileNotFoundError(f"No INP file found under {root}")

    def list_scenarios(self, benchmark_name: str = "Hanoi_CMH") -> List[str]:
        """Return the available scenario folder names for a benchmark."""

        root = self.find_benchmark_root(benchmark_name)
        scenarios = [path.name for path in root.iterdir() if path.is_dir() and path.name.lower().startswith("scenario-")]
        return sorted(scenarios, key=_natural_node_key)

    def _load_pressure_matrix(self, scenario_dir: Path) -> pd.DataFrame:
        pressure_dir = scenario_dir / "Pressures"
        if not pressure_dir.exists():
            raise FileNotFoundError(f"Missing pressure directory: {pressure_dir}")

        frames = []
        for csv_path in sorted(pressure_dir.glob("Node_*.csv"), key=lambda item: _natural_node_key(item.stem)):
            node_id = csv_path.stem.split("_", 1)[1]
            series = _read_csv_time_series(csv_path)
            series = series.rename(columns={"Value": node_id})
            frames.append(series.set_index("Timestamp"))

        if not frames:
            raise ValueError(f"No node pressure files found in {pressure_dir}")

        merged = pd.concat(frames, axis=1).sort_index()
        merged = merged[~merged.index.duplicated(keep="first")]
        merged.columns = [str(column) for column in merged.columns]
        return merged

    def _load_flow_matrix(self, scenario_dir: Path) -> pd.DataFrame:
        flow_dir = scenario_dir / "Flows"
        if not flow_dir.exists():
            return pd.DataFrame()

        frames = []
        for csv_path in sorted(flow_dir.glob("Link_*.csv"), key=lambda item: _natural_node_key(item.stem)):
            link_id = csv_path.stem.split("_", 1)[1]
            series = _read_csv_time_series(csv_path)
            series = series.rename(columns={"Value": link_id})
            frames.append(series.set_index("Timestamp"))

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames, axis=1).sort_index()
        merged = merged[~merged.index.duplicated(keep="first")]
        merged.columns = [str(column) for column in merged.columns]
        return merged

    def _load_labels(self, scenario_dir: Path) -> pd.Series:
        label_file = scenario_dir / "Labels.csv"
        if not label_file.exists():
            return pd.Series(dtype=float)
        frame = pd.read_csv(label_file)
        if frame.shape[1] < 2:
            raise ValueError(f"Unexpected label format in {label_file}")
        frame.columns = ["Timestamp", "Label"]
        frame["Timestamp"] = pd.to_datetime(frame["Timestamp"])
        return frame.set_index("Timestamp")["Label"].astype(float)

    def _load_leak_metadata(self, scenario_dir: Path) -> Dict[str, object]:
        leak_dir = scenario_dir / "Leaks"
        meta: Dict[str, object] = {}
        if not leak_dir.exists():
            return meta
        info_files = sorted(leak_dir.glob("*_info.csv"))
        if not info_files:
            return meta
        info = pd.read_csv(info_files[0])
        if info.shape[1] >= 2:
            info.columns = ["Description", "Value"]
            entries = {str(row["Description"]).strip(): str(row["Value"]).strip() for _, row in info.iterrows()}
            meta["info"] = entries
            meta["leak_node"] = _maybe_int(entries.get("Leak Node", -1))
            meta["leak_start"] = _safe_to_datetime(entries.get("Leak Start"))
            meta["leak_end"] = _safe_to_datetime(entries.get("Leak End"))
        demand_files = sorted(leak_dir.glob("*_demand.csv"))
        if demand_files:
            demand = _read_csv_time_series(demand_files[0])
            meta["leak_demand"] = demand
        return meta

    def load_topology(self, inp_path: str | Path) -> Tuple[NetworkModel, nx.Graph]:
        model = parse_inp_file(inp_path)
        graph = build_topology_graph(model)
        return model, graph

    def build_temporal_signal(
        self,
        benchmark_name: str = "Hanoi_CMH",
        scenario_name: str = "Scenario-1",
        feature_mode: str = "pressure",
        target_mode: str = "pressure_next",
        include_raw_frames: bool = False,
    ) -> Tuple[object, Dict[str, object]]:
        return self.load_scenario(
            benchmark_name=benchmark_name,
            scenario_name=scenario_name,
            feature_mode=feature_mode,
            target_mode=target_mode,
            include_raw_frames=include_raw_frames,
        )

    def load_scenario(
        self,
        benchmark_name: str = "Hanoi_CMH",
        scenario_name: str = "Scenario-1",
        feature_mode: str = "pressure",
        target_mode: str = "pressure_next",
        include_raw_frames: bool = False,
    ) -> Tuple[object, Dict[str, object]]:
        """Build a PyG-Temporal-compatible signal and accompanying metadata."""

        benchmark_root = self.find_benchmark_root(benchmark_name)
        inp_path = self.find_inp_path(benchmark_name, scenario_name)
        scenario_dir = benchmark_root / scenario_name
        if not scenario_dir.exists():
            raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

        model, graph = self.load_topology(inp_path)
        pressure_df = self._load_pressure_matrix(scenario_dir)
        flow_df = self._load_flow_matrix(scenario_dir) if include_raw_frames else pd.DataFrame()
        label_series = self._load_labels(scenario_dir)
        leak_meta = self._load_leak_metadata(scenario_dir)

        node_ids = [node_id for node_id in model.node_ids() if node_id in pressure_df.columns or node_id in model.reservoirs]
        node_ids = sorted(node_ids, key=_natural_node_key)
        if not node_ids:
            node_ids = sorted(pressure_df.columns.tolist(), key=_natural_node_key)

        pressure_df = pressure_df.reindex(columns=node_ids)
        pressure_df = pressure_df.interpolate(limit_direction="both").ffill().bfill()

        leak_node = leak_meta.get("leak_node")
        leak_indicator = pd.DataFrame(0.0, index=pressure_df.index, columns=node_ids)
        if isinstance(leak_node, int) and leak_node > 0:
            leak_col = str(leak_node)
            if leak_col in leak_indicator.columns:
                start = leak_meta.get("leak_start")
                end = leak_meta.get("leak_end")
                if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
                    leak_indicator.loc[(leak_indicator.index >= start) & (leak_indicator.index <= end), leak_col] = 1.0
                elif label_series.empty:
                    leak_indicator[leak_col] = 1.0

        features: List[np.ndarray] = []
        targets: List[np.ndarray] = []

        pressure_values = pressure_df.to_numpy(dtype=np.float32, copy=True)
        leak_values = leak_indicator.to_numpy(dtype=np.float32, copy=True) if not leak_indicator.empty else np.zeros_like(pressure_values)
        labels = label_series.reindex(pressure_df.index).fillna(0.0).to_numpy(dtype=np.float32, copy=True) if not label_series.empty else np.zeros(len(pressure_df), dtype=np.float32)

        if target_mode.lower() == "pressure_next":
            target_values = np.vstack([pressure_values[1:], pressure_values[-1:]])
        elif target_mode.lower() == "pressure":
            target_values = pressure_values.copy()
        elif target_mode.lower() == "label":
            target_values = np.repeat(labels.reshape(-1, 1), pressure_values.shape[1], axis=1)
        else:
            target_values = pressure_values.copy()

        for t in range(len(pressure_df)):
            if feature_mode.lower() == "pressure_only":
                node_features = pressure_values[t][:, None]
            elif feature_mode.lower() == "pressure":
                node_features = np.stack([pressure_values[t], leak_values[t]], axis=-1)
            else:
                node_features = np.stack([pressure_values[t], leak_values[t]], axis=-1)
            features.append(node_features.astype(np.float32))
            targets.append(target_values[t].astype(np.float32))

        raw_pressure_frame = pressure_df.copy() if include_raw_frames else None
        raw_flow_frame = flow_df.copy() if include_raw_frames else None
        raw_labels = label_series.copy() if include_raw_frames else None

        # Release large intermediates as soon as they have been converted.
        del pressure_df, leak_indicator, pressure_values, leak_values, labels, target_values

        edge_index, edge_weight = self._graph_to_edge_index(graph, node_ids)

        signal_cls = PyGStaticGraphTemporalSignal or StaticGraphTemporalSignal
        signal = signal_cls(edge_index, edge_weight, features, targets)

        metadata: Dict[str, object] = {
            "benchmark_root": benchmark_root,
            "scenario_dir": scenario_dir,
            "scenario_name": scenario_name,
            "inp_path": inp_path,
            "model": model,
            "graph": graph,
            "node_ids": node_ids,
            "leak_metadata": leak_meta,
        }
        if include_raw_frames:
            metadata["pressure_frame"] = raw_pressure_frame
            metadata["flow_frame"] = raw_flow_frame
            metadata["labels"] = raw_labels
        return signal, metadata

    def load_scenarios(
        self,
        benchmark_name: str = "Hanoi_CMH",
        max_scenarios: int = 20,
        scenario_names: Optional[Sequence[str]] = None,
        feature_mode: str = "pressure",
        target_mode: str = "pressure_next",
        include_raw_frames: bool = False,
    ) -> List[ScenarioBundle]:
        """Load multiple scenarios for batch training/evaluation."""

        available = list(scenario_names) if scenario_names is not None else self.list_scenarios(benchmark_name)
        selected = available[:max_scenarios]
        bundles: List[ScenarioBundle] = []
        for scenario_name in selected:
            signal, metadata = self.load_scenario(
                benchmark_name=benchmark_name,
                scenario_name=scenario_name,
                feature_mode=feature_mode,
                target_mode=target_mode,
                include_raw_frames=include_raw_frames,
            )
            bundles.append(ScenarioBundle(scenario_name=scenario_name, signal=signal, metadata=metadata))
        return bundles

    def _graph_to_edge_index(self, graph: nx.Graph, node_ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        edges: List[Tuple[int, int]] = []
        weights: List[float] = []
        for source, target, data in graph.edges(data=True):
            if source not in node_to_idx or target not in node_to_idx:
                continue
            i = node_to_idx[source]
            j = node_to_idx[target]
            capacity = float(data.get("capacity", 1.0))
            if capacity <= 0:
                continue
            edges.append((i, j))
            edges.append((j, i))
            weights.extend([capacity, capacity])
        if not edges:
            return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
        edge_index = np.array(edges, dtype=np.int64).T
        edge_weight = np.array(weights, dtype=np.float32)
        return edge_index, edge_weight


def load_leakdb_signal(
    raw_root: str | Path,
    benchmark_name: str = "Hanoi_CMH",
    scenario_name: str = "Scenario-1",
) -> Tuple[object, Dict[str, object]]:
    loader = LeakDBDatasetLoader(raw_root)
    return loader.build_temporal_signal(benchmark_name=benchmark_name, scenario_name=scenario_name)
