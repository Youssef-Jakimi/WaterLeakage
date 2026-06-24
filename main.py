"""End-to-end orchestrator for the smart water leakage management demo.

This version supports batch training across multiple LeakDB scenarios and
produces a global report with convergence and score aggregation plots.
"""

from __future__ import annotations

import gc
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from jinja2 import Template
from torch import nn

from agents.llm_valve_agent import recommend_isolation_action
from ai_layer.dataset_loader import LeakDBDatasetLoader, ScenarioBundle
from ai_layer.stgcn_model import STGCN, anomaly_scores, iter_sequence_batches
from core_ro.ford_fulkerson import maximize_redirection
from core_ro.prim_mst import mst_summary
from simulation.epanet_engine import EpanetEngine


LOGGER = logging.getLogger("water_leakage")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _resolve_raw_root(data_root: str | Path | None = None) -> Path:
    if data_root is not None:
        explicit = Path(data_root)
        if explicit.exists():
            return explicit
        raise FileNotFoundError(f"Could not find the data directory: {explicit}")

    candidates = [
        Path(__file__).resolve().parent / "data" / "raw" / "LeakDB",
        Path(__file__).resolve().parent / "data" / "raw",
        Path(__file__).resolve().parent / "data",
    ]
    best_candidate: Path | None = None
    best_count = -1
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            loader = LeakDBDatasetLoader(candidate)
            scenario_count = len(loader.list_scenarios("Hanoi_CMH"))
        except Exception:
            scenario_count = -1
        if scenario_count > best_count:
            best_count = scenario_count
            best_candidate = candidate
    if best_candidate is not None:
        return best_candidate
    raise FileNotFoundError("Could not find the data/raw directory.")


def _resolve_output_root() -> Path:
    base = Path(__file__).resolve().parent / "reports" / "output"
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / datetime.now().strftime("batch_run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_device(device_name: str = "auto") -> torch.device:
    normalized = device_name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this runtime.")
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    return torch.device(device_name)


def _stack_windows(
    signal_features: Sequence[np.ndarray],
    signal_targets: Sequence[np.ndarray],
    window_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batches = list(iter_sequence_batches(signal_features, signal_targets, window_size=window_size, batch_size=len(signal_features)))
    if not batches:
        raise ValueError("Not enough snapshots to build the requested window")
    return batches[0]


def _prepare_scenario_batch(
    bundle: ScenarioBundle,
    window_size: int,
    window_batch_size: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    signal = bundle.signal
    batches = [
        (x.float(), y.float())
        for x, y in iter_sequence_batches(signal.features, signal.targets, window_size=window_size, batch_size=window_batch_size)
    ]
    return batches


def _cache_scenario_batches(
    scenario_bundles: Sequence[ScenarioBundle],
    window_size: int,
    window_batch_size: int,
    device: torch.device,
) -> List[Dict[str, object]]:
    """Materialize all scenario sliding windows once before training starts."""

    cached_batches: List[Dict[str, object]] = []
    for bundle in scenario_bundles:
        scenario_batches = [
            (x.float().to(device), y.float().to(device))
            for x, y in iter_sequence_batches(
                bundle.signal.features,
                bundle.signal.targets,
                window_size=window_size,
                batch_size=window_batch_size,
            )
        ]
        cached_batches.append(
            {
                "scenario_name": bundle.scenario_name,
                "scenario_dir": str(bundle.metadata.get("scenario_dir", "")),
                "batches": scenario_batches,
            }
        )
    return cached_batches


def _train_streaming_model(
    model: STGCN,
    cached_scenario_batches: Sequence[Dict[str, object]],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    epochs: int = 30,
) -> List[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    history: List[float] = []
    scenario_count = max(1, len(cached_scenario_batches))

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch_losses: List[float] = []

        for scenario_cache in cached_scenario_batches:
            scenario_batches = scenario_cache["batches"]
            scenario_loss_total = 0.0
            scenario_batch_count = 0
            for x_batch, y_batch in scenario_batches:
                prediction = model(x_batch, edge_index=edge_index, edge_weight=edge_weight)
                loss = criterion(prediction, y_batch)
                (loss / scenario_count).backward()
                scenario_loss_total += float(loss.item())
                scenario_batch_count += 1
                del x_batch, y_batch, prediction, loss
            epoch_losses.append(scenario_loss_total / max(1, scenario_batch_count))

        optimizer.step()
        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        history.append(epoch_loss)
        LOGGER.info("Batch epoch %d/%d | loss=%.6f", epoch + 1, epochs, epoch_loss)

    return history


@torch.no_grad()
def _evaluate_streaming_model(
    model: STGCN,
    cached_scenario_batches: Sequence[Dict[str, object]],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> Tuple[List[Dict[str, object]], List[np.ndarray]]:
    model.eval()
    scenario_metrics: List[Dict[str, object]] = []
    score_curves: List[np.ndarray] = []

    for scenario_cache in cached_scenario_batches:
        scenario_batches = scenario_cache["batches"]
        batch_scores: List[np.ndarray] = []
        mean_curve_parts: List[np.ndarray] = []
        for x_batch, y_batch in scenario_batches:
            prediction = model(x_batch, edge_index=edge_index, edge_weight=edge_weight)
            scores = anomaly_scores(prediction, y_batch)
            batch_scores.append(scores.cpu().numpy())
            mean_curve_parts.append(scores.mean(dim=1).cpu().numpy())
            del x_batch, y_batch, prediction, scores
        scores_array = np.concatenate(batch_scores, axis=0) if batch_scores else np.empty((0, 0), dtype=np.float32)
        mean_curve = np.concatenate(mean_curve_parts, axis=0) if mean_curve_parts else np.empty((0,), dtype=np.float32)
        scenario_metrics.append(
            {
                "scenario_name": scenario_cache["scenario_name"],
                "scenario_dir": scenario_cache["scenario_dir"],
                "mean_score": float(scores_array.mean()) if scores_array.size else 0.0,
                "max_score": float(scores_array.max()) if scores_array.size else 0.0,
                "num_windows": int(scores_array.shape[0]) if scores_array.ndim > 0 else 0,
                "num_nodes": int(scores_array.shape[1]) if scores_array.ndim > 1 else 0,
            }
        )
        score_curves.append(mean_curve)
        del scenario_batches, batch_scores, mean_curve_parts, scores_array, mean_curve
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return scenario_metrics, score_curves


def _choose_source_and_critical_nodes(model: object) -> Tuple[str, List[str]]:
    reservoirs = list(getattr(model, "reservoirs", {}).keys())
    source = reservoirs[0] if reservoirs else "1"
    junctions = list(getattr(model, "junctions", {}).values())
    critical = sorted(junctions, key=lambda item: item.demand, reverse=True)[:3]
    return source, [junction.node_id for junction in critical]


def _build_static_vs_gnn_comparison(
    static_pressures: pd.DataFrame,
    gnn_pressures: pd.DataFrame,
    static_flow: Mapping[str, object],
    gnn_flow: Mapping[str, object],
    static_mst: Mapping[str, object],
    gnn_mst: Mapping[str, object],
) -> Dict[str, float]:
    """Summarize a static leak condition versus the GNN-guided intervention."""

    static_loss = float(np.clip(20.0 - static_pressures["pressure"], a_min=0.0, a_max=None).sum())
    gnn_loss = float(np.clip(20.0 - gnn_pressures["pressure"], a_min=0.0, a_max=None).sum())
    static_mean = float(static_pressures["pressure"].mean())
    gnn_mean = float(gnn_pressures["pressure"].mean())
    static_flow_value = float(static_flow.get("max_flow", 0.0))
    gnn_flow_value = float(gnn_flow.get("max_flow", 0.0))
    static_mst_value = float(static_mst.get("total_cost", 0.0))
    gnn_mst_value = float(gnn_mst.get("total_cost", 0.0))

    return {
        "static_mean_pressure": static_mean,
        "gnn_mean_pressure": gnn_mean,
        "static_pressure_loss": static_loss,
        "gnn_pressure_loss": gnn_loss,
        "static_max_flow": static_flow_value,
        "gnn_max_flow": gnn_flow_value,
        "static_mst_cost": static_mst_value,
        "gnn_mst_cost": gnn_mst_value,
        "pressure_loss_reduction": static_loss - gnn_loss,
        "pressure_loss_reduction_pct": 0.0 if static_loss <= 0 else ((static_loss - gnn_loss) / static_loss) * 100.0,
        "flow_delta": gnn_flow_value - static_flow_value,
        "mst_delta": gnn_mst_value - static_mst_value,
    }


def _build_training_batches(
    bundles: Sequence[ScenarioBundle],
    window_size: int,
) -> List[Dict[str, object]]:
    batches: List[Dict[str, object]] = []
    for bundle in bundles:
        x, y = _stack_windows(bundle.signal.features, bundle.signal.targets, window_size=window_size)
        batches.append(
            {
                "scenario_name": bundle.scenario_name,
                "x": x.float(),
                "y": y.float(),
                "metadata": bundle.metadata,
                "signal": bundle.signal,
            }
        )
    return batches


def _train_batch_model(
    model: STGCN,
    batches: Sequence[Dict[str, object]],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    epochs: int = 30,
) -> List[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    history: List[float] = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses: List[torch.Tensor] = []
        for batch in batches:
            prediction = model(batch["x"], edge_index=edge_index, edge_weight=edge_weight)
            loss = criterion(prediction, batch["y"])
            losses.append(loss)
        epoch_loss = torch.stack(losses).mean()
        epoch_loss.backward()
        optimizer.step()
        history.append(float(epoch_loss.item()))
        LOGGER.info("Batch epoch %d/%d | loss=%.6f", epoch + 1, epochs, epoch_loss.item())

    return history


@torch.no_grad()
def _evaluate_batches(
    model: STGCN,
    batches: Sequence[Dict[str, object]],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> Tuple[List[Dict[str, object]], List[np.ndarray]]:
    model.eval()
    scenario_metrics: List[Dict[str, object]] = []
    score_curves: List[np.ndarray] = []

    for batch in batches:
        prediction = model(batch["x"], edge_index=edge_index, edge_weight=edge_weight)
        scores = anomaly_scores(prediction, batch["y"])
        mean_curve = scores.mean(dim=1).cpu().numpy()
        mean_score = float(scores.mean().item())
        max_score = float(scores.max().item())
        scenario_metrics.append(
            {
                "scenario_name": batch["scenario_name"],
                "mean_score": mean_score,
                "max_score": max_score,
                "num_windows": int(scores.shape[0]),
                "num_nodes": int(scores.shape[1]) if scores.dim() > 1 else 0,
            }
        )
        score_curves.append(mean_curve)

    return scenario_metrics, score_curves


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return path


def _plot_loss_curve(loss_history: Sequence[float], output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(1, len(loss_history) + 1), loss_history, color="#1f77b4", linewidth=2.0, marker="o")
    ax.set_title("Batch Training Loss Convergence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.25)
    return _save_figure(fig, output_path)


def _plot_scenario_mean_scores(scenario_metrics: Sequence[Dict[str, object]], output_path: Path) -> Path:
    names = [item["scenario_name"] for item in scenario_metrics]
    values = [float(item["mean_score"]) for item in scenario_metrics]
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.7), 5))
    ax.bar(names, values, color="#ff7f0e")
    ax.set_title("Mean Anomaly Score by Scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Mean Anomaly Score")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.25)
    return _save_figure(fig, output_path)


def _plot_average_score_curve(mean_curve: Sequence[float], output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(len(mean_curve)), mean_curve, color="#d62728", linewidth=2.0)
    ax.set_title("Global Average Anomaly Score Curve")
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Mean Absolute Error")
    ax.grid(True, alpha=0.25)
    return _save_figure(fig, output_path)


def _render_global_report(context: Dict[str, object], output_dir: Path) -> Tuple[Path, Path]:
    html_path = output_dir / "global_report.html"
    md_path = output_dir / "global_report.md"
    template = Template(
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>Global LeakDB Batch Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 2rem; color: #111827; }
                .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }
                .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem; background: #fff; }
                img { width: 100%; height: auto; border-radius: 8px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #e5e7eb; padding: 0.5rem 0.75rem; text-align: left; }
                th { background: #f8fafc; }
                .muted { color: #6b7280; }
            </style>
        </head>
        <body>
            <h1>Global LeakDB Batch Report</h1>
            <p class="muted">Generated at {{ generated_at }}</p>
            <div class="card">
                <h2>Batch Summary</h2>
                <table>
                    <tbody>
                    {% for key, value in summary.items() %}
                        <tr><th>{{ key }}</th><td>{{ value }}</td></tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
            <div class="grid">
                <div class="card">
                    <h2>Training Loss</h2>
                    <img src="images/loss_curve.png" alt="Loss curve" />
                </div>
                <div class="card">
                    <h2>Scenario Mean Scores</h2>
                    <img src="images/scenario_mean_scores.png" alt="Scenario mean scores" />
                </div>
                <div class="card">
                    <h2>Average Score Curve</h2>
                    <img src="images/average_score_curve.png" alt="Average anomaly curve" />
                </div>
            </div>
            <div class="card" style="margin-top:1rem;">
                <h2>Scenario Metrics</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Scenario</th>
                            <th>Source</th>
                            <th>Mean Score</th>
                            <th>Max Score</th>
                            <th>Windows</th>
                            <th>Nodes</th>
                        </tr>
                    </thead>
                    <tbody>
                    {% for row in scenario_metrics %}
                        <tr>
                            <td>{{ row.scenario_name }}</td>
                            <td class="muted">{{ row.scenario_dir }}</td>
                            <td>{{ "%.6f"|format(row.mean_score) }}</td>
                            <td>{{ "%.6f"|format(row.max_score) }}</td>
                            <td>{{ row.num_windows }}</td>
                            <td>{{ row.num_nodes }}</td>
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
            <div class="card" style="margin-top:1rem;">
                <h2>Static Leak Simulation vs GNN Detection</h2>
                <table>
                    <tbody>
                        <tr><th>Static baseline pressure loss</th><td>{{ "%.6f"|format(static_vs_gnn.static_pressure_loss) }}</td></tr>
                        <tr><th>GNN pressure loss</th><td>{{ "%.6f"|format(static_vs_gnn.gnn_pressure_loss) }}</td></tr>
                        <tr><th>Pressure loss reduction</th><td>{{ "%.6f"|format(static_vs_gnn.pressure_loss_reduction) }}</td></tr>
                        <tr><th>Pressure loss reduction %</th><td>{{ "%.2f"|format(static_vs_gnn.pressure_loss_reduction_pct) }}</td></tr>
                        <tr><th>Static max flow</th><td>{{ "%.6f"|format(static_vs_gnn.static_max_flow) }}</td></tr>
                        <tr><th>GNN max flow</th><td>{{ "%.6f"|format(static_vs_gnn.gnn_max_flow) }}</td></tr>
                        <tr><th>Static MST cost</th><td>{{ "%.6f"|format(static_vs_gnn.static_mst_cost) }}</td></tr>
                        <tr><th>GNN MST cost</th><td>{{ "%.6f"|format(static_vs_gnn.gnn_mst_cost) }}</td></tr>
                    </tbody>
                </table>
            </div>
            <div class="card" style="margin-top:1rem;">
                <h2>Representative Operational Check</h2>
                <p><strong>Report:</strong> {{ recommendation.report }}</p>
                <p><strong>Action:</strong> {{ recommendation.action }}</p>
                <p><strong>Pipe ID:</strong> {{ recommendation.pipe_id }}</p>
                <p><strong>Reason:</strong> {{ recommendation.reason }}</p>
                <p><strong>Representative Scenario:</strong> {{ representative_scenario }}</p>
                <p><strong>Mean Pressure Before:</strong> {{ "%.4f"|format(representative_before_pressure) }}</p>
                <p><strong>Mean Pressure After:</strong> {{ "%.4f"|format(representative_after_pressure) }}</p>
                <p><strong>Max Flow Before:</strong> {{ "%.6f"|format(representative_max_flow_before) }}</p>
                <p><strong>Max Flow After:</strong> {{ "%.6f"|format(representative_max_flow_after) }}</p>
                <p><strong>Max Flow Delta:</strong> {{ "%.6f"|format(representative_max_flow_delta) }}</p>
                <p><strong>MST Cost Before:</strong> {{ "%.6f"|format(representative_mst_cost_before) }}</p>
                <p><strong>MST Cost After:</strong> {{ "%.6f"|format(representative_mst_cost_after) }}</p>
                <p><strong>MST Cost Delta:</strong> {{ "%.6f"|format(representative_mst_cost_delta) }}</p>
            </div>
        </body>
        </html>
        """
    )
    html_path.write_text(template.render(**context), encoding="utf-8")

    md_lines = [
        "# Global LeakDB Batch Report",
        "",
        f"Generated at: {context['generated_at']}",
        "",
        "## Batch Summary",
    ]
    for key, value in context["summary"].items():
        md_lines.append(f"- {key}: {value}")
    md_lines.extend(
        [
            "",
            "## Scenario Metrics",
        ]
    )
    for row in context["scenario_metrics"]:
        md_lines.append(
            f"- {row['scenario_name']} [{row.get('scenario_dir', '')}]: mean={row['mean_score']:.6f}, max={row['max_score']:.6f}, windows={row['num_windows']}, nodes={row['num_nodes']}"
        )
    md_lines.extend(
        [
            "",
            "## Representative Operational Check",
            f"- Representative scenario: {context['representative_scenario']}",
            f"- Recommendation: {context['recommendation'].get('pipe_id')} ({context['recommendation'].get('reason')})",
            f"- Max flow before: {context['representative_max_flow_before']:.6f}",
            f"- Max flow after: {context['representative_max_flow_after']:.6f}",
            f"- Max flow delta: {context['representative_max_flow_delta']:.6f}",
            f"- MST cost before: {context['representative_mst_cost_before']:.6f}",
            f"- MST cost after: {context['representative_mst_cost_after']:.6f}",
            f"- MST cost delta: {context['representative_mst_cost_delta']:.6f}",
            "",
            "## Static Leak Simulation vs GNN Detection",
            f"- Static baseline pressure loss: {context['static_vs_gnn']['static_pressure_loss']:.6f}",
            f"- GNN pressure loss: {context['static_vs_gnn']['gnn_pressure_loss']:.6f}",
            f"- Pressure loss reduction: {context['static_vs_gnn']['pressure_loss_reduction']:.6f}",
            f"- Pressure loss reduction %: {context['static_vs_gnn']['pressure_loss_reduction_pct']:.2f}",
            f"- Static max flow: {context['static_vs_gnn']['static_max_flow']:.6f}",
            f"- GNN max flow: {context['static_vs_gnn']['gnn_max_flow']:.6f}",
            f"- Static MST cost: {context['static_vs_gnn']['static_mst_cost']:.6f}",
            f"- GNN MST cost: {context['static_vs_gnn']['gnn_mst_cost']:.6f}",
        ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return html_path, md_path


def run_pipeline(
    benchmark_name: str = "Hanoi_CMH",
    max_scenarios: int = 20,
    epochs: int = 30,
    window_batch_size: int = 64,
    data_root: str | Path | None = None,
    device: str = "auto",
) -> Dict[str, object]:
    raw_root = _resolve_raw_root(data_root)
    loader = LeakDBDatasetLoader(raw_root)
    run_device = _resolve_device(device)

    available_scenarios = loader.list_scenarios(benchmark_name)
    scenario_bundles = loader.load_scenarios(
        benchmark_name=benchmark_name,
        max_scenarios=max_scenarios,
        scenario_names=available_scenarios,
        include_raw_frames=False,
    )
    if not scenario_bundles:
        raise FileNotFoundError(f"No scenarios found for benchmark {benchmark_name}")
    selected_scenarios = [bundle.scenario_name for bundle in scenario_bundles]

    if len(selected_scenarios) < max_scenarios:
        LOGGER.info(
            "Batch mode requested %d scenarios, but only %d are available in %s. Using all available scenarios.",
            max_scenarios,
            len(selected_scenarios),
            benchmark_name,
        )

    first_bundle = scenario_bundles[0]
    first_signal = first_bundle.signal
    first_metadata = first_bundle.metadata
    first_model = first_metadata["model"]
    first_graph = first_metadata["graph"]
    first_node_ids = first_metadata["node_ids"]

    window_size = min(12, max(4, len(first_signal.features) // 12))

    stgcn = STGCN(
        num_nodes=len(first_node_ids),
        in_channels=first_signal.features[0].shape[-1],
        hidden_channels=32,
        output_channels=1,
        temporal_kernel_size=3,
        dropout=0.1,
    )
    stgcn = stgcn.to(run_device)
    if run_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model_device = next(stgcn.parameters()).device
    edge_index = torch.tensor(first_signal.edge_index, dtype=torch.long, device=model_device)
    edge_weight = torch.tensor(first_signal.edge_weight, dtype=torch.float32, device=model_device)
    cached_scenario_batches = _cache_scenario_batches(
        scenario_bundles=scenario_bundles,
        window_size=window_size,
        window_batch_size=window_batch_size,
        device=model_device,
    )

    loss_history = _train_streaming_model(
        stgcn,
        cached_scenario_batches=cached_scenario_batches,
        edge_index=edge_index,
        edge_weight=edge_weight,
        epochs=epochs,
    )
    scenario_metrics, score_curves = _evaluate_streaming_model(
        stgcn,
        cached_scenario_batches=cached_scenario_batches,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )

    min_curve_len = min(len(curve) for curve in score_curves)
    stacked_curves = np.stack([curve[:min_curve_len] for curve in score_curves], axis=0)
    global_average_curve = stacked_curves.mean(axis=0)
    overall_mean_score = float(np.mean([item["mean_score"] for item in scenario_metrics]))
    overall_max_score = float(np.max([item["max_score"] for item in scenario_metrics]))
    best_scenario = min(scenario_metrics, key=lambda item: item["mean_score"])
    worst_scenario = max(scenario_metrics, key=lambda item: item["mean_score"])

    output_dir = _resolve_output_root()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    loss_curve_path = _plot_loss_curve(loss_history, images_dir / "loss_curve.png")
    scenario_scores_path = _plot_scenario_mean_scores(scenario_metrics, images_dir / "scenario_mean_scores.png")
    average_curve_path = _plot_average_score_curve(global_average_curve, images_dir / "average_score_curve.png")

    representative_engine = EpanetEngine(first_metadata["inp_path"])
    representative_before = representative_engine.simulate()
    representative_report_text = "Big water leak near junction 5!"
    recommendation = recommend_isolation_action(
        representative_report_text,
        topology=first_graph,
        source_node=representative_engine.reservoir_nodes()[0] if representative_engine.reservoir_nodes() else None,
    )
    isolated_pipe = recommendation.get("pipe_id")
    representative_after = representative_before.copy()
    if isolated_pipe is not None:
        representative_after = representative_engine.close_and_resimulate(str(isolated_pipe))

    source_node, critical_nodes = _choose_source_and_critical_nodes(first_model)
    before_flow = maximize_redirection(first_model, source_node=source_node, critical_nodes=critical_nodes)
    after_flow = maximize_redirection(
        first_model,
        source_node=source_node,
        critical_nodes=critical_nodes,
        closed_links=[str(isolated_pipe)] if isolated_pipe is not None else None,
    )

    mst_before = mst_summary(first_model)
    degraded_engine = representative_engine.clone()
    if isolated_pipe is not None:
        degraded_engine.close_link(str(isolated_pipe))
    mst_after = mst_summary(degraded_engine.model)

    service_threshold = 20.0
    representative_pressure_drop = representative_before["pressure"] - representative_after["pressure"]
    representative_loss_before = float(
        np.clip(service_threshold - representative_before["pressure"], a_min=0.0, a_max=None).sum()
    )
    representative_loss_after = float(
        np.clip(service_threshold - representative_after["pressure"], a_min=0.0, a_max=None).sum()
    )

    static_vs_gnn = _build_static_vs_gnn_comparison(
        static_pressures=representative_before,
        gnn_pressures=representative_after,
        static_flow=before_flow,
        gnn_flow=after_flow,
        static_mst=mst_before,
        gnn_mst=mst_after,
    )

    representative_results = {
        "benchmark": benchmark_name,
        "scenario_name": selected_scenarios[0],
        "scenario_dir": str(first_metadata.get("scenario_dir", "")),
        "inp_path": str(first_metadata["inp_path"]),
        "recommendation": recommendation,
        "isolated_pipe": isolated_pipe,
        "before_mean_pressure": float(representative_before["pressure"].mean()),
        "after_mean_pressure": float(representative_after["pressure"].mean()),
        "mean_pressure_drop": float(representative_pressure_drop.mean()),
        "pressure_loss_before": representative_loss_before,
        "pressure_loss_after": representative_loss_after,
        "max_flow_before": before_flow["max_flow"],
        "max_flow_after": after_flow["max_flow"],
        "max_flow_delta": float(after_flow["max_flow"] - before_flow["max_flow"]),
        "mst_cost_before": mst_before["total_cost"],
        "mst_cost_after": mst_after["total_cost"],
        "mst_cost_delta": float(mst_after["total_cost"] - mst_before["total_cost"]),
        "static_vs_gnn": static_vs_gnn,
    }

    global_summary = {
        "Benchmark": benchmark_name,
        "Requested scenarios": max_scenarios,
        "Scenarios used": len(selected_scenarios),
        "Training epochs": epochs,
        "Window batch size": window_batch_size,
        "Window size": window_size,
        "Overall mean anomaly score": round(overall_mean_score, 6),
        "Overall max anomaly score": round(overall_max_score, 6),
        "Best scenario": f"{best_scenario['scenario_name']} ({best_scenario['mean_score']:.6f})",
        "Worst scenario": f"{worst_scenario['scenario_name']} ({worst_scenario['mean_score']:.6f})",
        "Representative scenario source": str(first_metadata.get("scenario_dir", "")),
        "Representative max flow before": round(representative_results["max_flow_before"], 6),
        "Representative max flow after": round(representative_results["max_flow_after"], 6),
        "Representative max flow delta": round(representative_results["max_flow_delta"], 6),
        "Representative MST cost before": round(representative_results["mst_cost_before"], 6),
        "Representative MST cost after": round(representative_results["mst_cost_after"], 6),
        "Representative MST cost delta": round(representative_results["mst_cost_delta"], 6),
        "Static baseline pressure loss": round(static_vs_gnn["static_pressure_loss"], 6),
        "GNN pressure loss": round(static_vs_gnn["gnn_pressure_loss"], 6),
        "Pressure loss reduction": round(static_vs_gnn["pressure_loss_reduction"], 6),
        "Pressure loss reduction %": round(static_vs_gnn["pressure_loss_reduction_pct"], 2),
    }

    report_context = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": global_summary,
        "scenario_metrics": scenario_metrics,
        "recommendation": recommendation,
        "representative_scenario": selected_scenarios[0],
        "representative_before_pressure": representative_results["before_mean_pressure"],
        "representative_after_pressure": representative_results["after_mean_pressure"],
        "representative_max_flow_before": representative_results["max_flow_before"],
        "representative_max_flow_after": representative_results["max_flow_after"],
        "representative_max_flow_delta": representative_results["max_flow_delta"],
        "representative_mst_cost_before": representative_results["mst_cost_before"],
        "representative_mst_cost_after": representative_results["mst_cost_after"],
        "representative_mst_cost_delta": representative_results["mst_cost_delta"],
        "static_vs_gnn": static_vs_gnn,
    }
    report_html, report_md = _render_global_report(report_context, output_dir)

    results: Dict[str, object] = {
        "batch_mode": True,
        "benchmark": benchmark_name,
        "requested_scenarios": max_scenarios,
        "scenarios_used": len(selected_scenarios),
        "scenario_names": list(selected_scenarios),
        "window_size": window_size,
        "epochs": epochs,
        "loss_history": loss_history,
        "scenario_metrics": scenario_metrics,
        "global_average_curve": global_average_curve.tolist(),
        "overall_mean_score": overall_mean_score,
        "overall_max_score": overall_max_score,
        "best_scenario": best_scenario,
        "worst_scenario": worst_scenario,
        "representative_results": representative_results,
        "static_vs_gnn": static_vs_gnn,
        "report_dir": str(output_dir),
        "report_html": str(report_html),
        "report_md": str(report_md),
        "figures": {
            "loss_curve": str(loss_curve_path),
            "scenario_mean_scores": str(scenario_scores_path),
            "average_score_curve": str(average_curve_path),
        },
    }

    LOGGER.info("Batch scenarios used: %d", len(selected_scenarios))
    LOGGER.info("Overall mean anomaly score: %.6f", overall_mean_score)
    LOGGER.info("Best scenario: %s", results["best_scenario"])
    LOGGER.info("Worst scenario: %s", results["worst_scenario"])
    LOGGER.info("Representative recommendation: %s", recommendation)
    LOGGER.info("Visual report saved to: %s", report_html)
    del first_signal
    gc.collect()
    return results


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Smart Water Leakage Management batch runner")
    parser.add_argument("--benchmark", default="Hanoi_CMH", help="LeakDB benchmark name, e.g. Hanoi_CMH")
    parser.add_argument("--max-scenarios", type=int, default=20, help="Maximum number of scenarios to train on")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for the STGCN model")
    parser.add_argument("--window-batch-size", type=int, default=64, help="Mini-batch size for sliding windows")
    parser.add_argument("--data-root", default=None, help="Path to the dataset root or project data folder")
    parser.add_argument("--device", default="auto", help="Training device: auto, cuda, cpu, or a torch device string")
    args = parser.parse_args()

    results = run_pipeline(
        benchmark_name=args.benchmark,
        max_scenarios=args.max_scenarios,
        epochs=args.epochs,
        window_batch_size=args.window_batch_size,
        data_root=args.data_root,
        device=args.device,
    )
    LOGGER.info("Pipeline finished successfully: %s", results)


if __name__ == "__main__":
    main()
