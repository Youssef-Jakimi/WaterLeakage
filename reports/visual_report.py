"""Generate charts and an HTML report for a pipeline run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import base64
import io
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from jinja2 import Template

LOGGER = logging.getLogger(__name__)


@dataclass
class ReportArtifacts:
    output_dir: Path
    images: Dict[str, Path]
    report_html: Path
    report_md: Path


def _ensure_output_dir(base_dir: str | Path | None = None) -> Path:
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent / "output"
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _figure_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return path


def _network_layout(graph: nx.Graph) -> Dict[str, Tuple[float, float]]:
    if graph.number_of_nodes() == 0:
        return {}
    try:
        return nx.spring_layout(graph, seed=42, weight="weight")
    except Exception:
        return nx.circular_layout(graph)


def _plot_pressure_timeseries(pressure_frame: pd.DataFrame, leak_metadata: Mapping[str, object], output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    pressure_frame.mean(axis=1).plot(ax=ax, color="#1f77b4", linewidth=2.0, label="Mean node pressure")
    ax.set_title("Network Mean Pressure Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Pressure")
    ax.grid(True, alpha=0.25)

    leak_start = leak_metadata.get("leak_start")
    leak_end = leak_metadata.get("leak_end")
    if isinstance(leak_start, pd.Timestamp):
        ax.axvline(leak_start, color="#d62728", linestyle="--", alpha=0.8, label="Leak start")
    if isinstance(leak_end, pd.Timestamp):
        ax.axvline(leak_end, color="#2ca02c", linestyle="--", alpha=0.8, label="Leak end")
    ax.legend(loc="best")
    return _save_figure(fig, output_path)


def _plot_anomaly_scores(score_curve: Sequence[float], output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(score_curve)), score_curve, color="#ff7f0e", linewidth=2.0)
    ax.set_title("STGCN Mean Anomaly Score by Window")
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Mean Absolute Error")
    ax.grid(True, alpha=0.25)
    return _save_figure(fig, output_path)


def _plot_pressure_before_after(before_pressures: pd.DataFrame, after_pressures: pd.DataFrame, output_path: Path) -> Path:
    aligned = before_pressures[["pressure"]].rename(columns={"pressure": "Before"}).join(
        after_pressures[["pressure"]].rename(columns={"pressure": "After"}),
        how="inner",
    )
    aligned = aligned.sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    aligned.plot(kind="bar", ax=ax, width=0.85)
    ax.set_title("Node Pressure Before vs After Isolation")
    ax.set_xlabel("Node")
    ax.set_ylabel("Pressure")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=90)
    return _save_figure(fig, output_path)


def _plot_metric_comparison(results: Mapping[str, object], output_path: Path) -> Path:
    labels = ["Max Flow", "MST Cost", "Pressure Loss"]
    before = [
        float(results.get("max_flow_before", 0.0)),
        float(results.get("mst_cost_before", 0.0)),
        float(results.get("pressure_loss_before", 0.0)),
    ]
    after = [
        float(results.get("max_flow_after", 0.0)),
        float(results.get("mst_cost_after", 0.0)),
        float(results.get("pressure_loss_after", 0.0)),
    ]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, before, width, label="Before", color="#1f77b4")
    ax.bar(x + width / 2, after, width, label="After", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Operational Performance Comparison")
    ax.set_ylabel("Metric Value")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    return _save_figure(fig, output_path)


def _plot_network(graph: nx.Graph, pressures: pd.Series, output_path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = _network_layout(graph)
    if not pos:
        ax.text(0.5, 0.5, "No graph data available", ha="center", va="center")
        ax.axis("off")
        return _save_figure(fig, output_path)

    nodes = list(graph.nodes())
    values = []
    for node in nodes:
        try:
            values.append(float(pressures.loc[node]))
        except Exception:
            values.append(np.nan)
    values_array = np.array(values, dtype=float)
    finite = values_array[np.isfinite(values_array)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.35, width=1.2)
    nodes_artist = nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=values_array,
        cmap=plt.cm.viridis,
        node_size=450,
        vmin=vmin,
        vmax=vmax,
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_color="white")
    fig.colorbar(nodes_artist, ax=ax, label="Pressure")
    ax.set_title(title)
    ax.axis("off")
    return _save_figure(fig, output_path)


def _render_html_report(context: Mapping[str, object], output_path: Path) -> Path:
    template = Template(
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>Smart Water Leakage Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 2rem; color: #1f2937; }
                h1, h2 { color: #0f172a; }
                .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }
                .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); background: #fff; }
                img { width: 100%; height: auto; border-radius: 8px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #e5e7eb; padding: 0.5rem 0.75rem; text-align: left; }
                th { background: #f8fafc; }
                .muted { color: #6b7280; }
                code { background: #f8fafc; padding: 0.1rem 0.3rem; border-radius: 4px; }
            </style>
        </head>
        <body>
            <h1>Smart Water Leakage Management Report</h1>
            <p class="muted">Generated at {{ generated_at }}</p>

            <div class="card">
                <h2>Run Summary</h2>
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
                    <h2>Mean Pressure Over Time</h2>
                    <img src="data:image/png;base64,{{ plots.pressure_timeseries }}" alt="Pressure time series" />
                </div>
                <div class="card">
                    <h2>Anomaly Scores</h2>
                    <img src="data:image/png;base64,{{ plots.anomaly_scores }}" alt="Anomaly scores" />
                </div>
                <div class="card">
                    <h2>Pressure Before vs After</h2>
                    <img src="data:image/png;base64,{{ plots.pressure_before_after }}" alt="Pressure before after" />
                </div>
                <div class="card">
                    <h2>Operational Comparison</h2>
                    <img src="data:image/png;base64,{{ plots.metric_comparison }}" alt="Metric comparison" />
                </div>
                <div class="card">
                    <h2>Network Before Isolation</h2>
                    <img src="data:image/png;base64,{{ plots.network_before }}" alt="Network before" />
                </div>
                <div class="card">
                    <h2>Network After Isolation</h2>
                    <img src="data:image/png;base64,{{ plots.network_after }}" alt="Network after" />
                </div>
            </div>

            <div class="card" style="margin-top:1rem;">
                <h2>Operational Resilience Metrics</h2>
                <table>
                    <tbody>
                        <tr><th>Max flow before</th><td>{{ "%.6f"|format(summary.max_flow_before) }}</td></tr>
                        <tr><th>Max flow after</th><td>{{ "%.6f"|format(summary.max_flow_after) }}</td></tr>
                        <tr><th>Max flow delta</th><td>{{ "%.6f"|format(summary.max_flow_delta) }}</td></tr>
                        <tr><th>MST cost before</th><td>{{ "%.6f"|format(summary.mst_cost_before) }}</td></tr>
                        <tr><th>MST cost after</th><td>{{ "%.6f"|format(summary.mst_cost_after) }}</td></tr>
                        <tr><th>MST cost delta</th><td>{{ "%.6f"|format(summary.mst_cost_delta) }}</td></tr>
                    </tbody>
                </table>
            </div>

            <div class="card" style="margin-top:1rem;">
                <h2>Static Leak Simulation vs GNN Detection</h2>
                <table>
                    <tbody>
                        <tr><th>Static baseline pressure loss</th><td>{{ "%.6f"|format(summary.static_pressure_loss) }}</td></tr>
                        <tr><th>GNN pressure loss</th><td>{{ "%.6f"|format(summary.gnn_pressure_loss) }}</td></tr>
                        <tr><th>Pressure loss reduction</th><td>{{ "%.6f"|format(summary.pressure_loss_reduction) }}</td></tr>
                        <tr><th>Pressure loss reduction %</th><td>{{ "%.2f"|format(summary.pressure_loss_reduction_pct) }}</td></tr>
                    </tbody>
                </table>
            </div>

            <div class="card" style="margin-top:1rem;">
                <h2>Recommendation</h2>
                <p><strong>Report:</strong> {{ recommendation.report }}</p>
                <p><strong>Action:</strong> {{ recommendation.action }}</p>
                <p><strong>Pipe ID:</strong> {{ recommendation.pipe_id }}</p>
                <p><strong>Reason:</strong> {{ recommendation.reason }}</p>
            </div>
        </body>
        </html>
        """
    )
    rendered = template.render(**context)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def _render_markdown_report(context: Mapping[str, object], output_path: Path) -> Path:
    summary = context["summary"]
    recommendation = context["recommendation"]
    lines = [
        "# Smart Water Leakage Management Report",
        "",
        f"Generated at: {context['generated_at']}",
        "",
        "## Summary",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Static Leak Simulation vs GNN Detection",
            f"- Static baseline pressure loss: {summary['static_pressure_loss']}",
            f"- GNN pressure loss: {summary['gnn_pressure_loss']}",
            f"- Pressure loss reduction: {summary['pressure_loss_reduction']}",
            f"- Pressure loss reduction %: {summary['pressure_loss_reduction_pct']}",
            "",
            "## Recommendation",
            f"- Report: {recommendation.get('report')}",
            f"- Action: {recommendation.get('action')}",
            f"- Pipe ID: {recommendation.get('pipe_id')}",
            f"- Reason: {recommendation.get('reason')}",
            "",
            "## Figures",
        ]
    )
    for name, path in context["image_paths"].items():
        lines.append(f"- {name}: {path.name}")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def generate_report(
    results: Mapping[str, object],
    pressure_frame: pd.DataFrame,
    graph: nx.Graph,
    before_pressures: pd.DataFrame,
    after_pressures: pd.DataFrame,
    signal_scores: Sequence[float],
    leak_metadata: Mapping[str, object],
    output_dir: str | Path | None = None,
) -> ReportArtifacts:
    """Create plots and an HTML/Markdown report for a pipeline execution."""

    run_dir = _ensure_output_dir(output_dir)
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    images: Dict[str, Path] = {}
    images["pressure_timeseries"] = _plot_pressure_timeseries(
        pressure_frame,
        leak_metadata,
        images_dir / "pressure_timeseries.png",
    )
    images["anomaly_scores"] = _plot_anomaly_scores(signal_scores, images_dir / "anomaly_scores.png")
    images["pressure_before_after"] = _plot_pressure_before_after(
        before_pressures,
        after_pressures,
        images_dir / "pressure_before_after.png",
    )
    images["metric_comparison"] = _plot_metric_comparison(results, images_dir / "metric_comparison.png")
    images["network_before"] = _plot_network(
        graph,
        before_pressures["pressure"],
        images_dir / "network_before.png",
        "Network Pressure Before Isolation",
    )
    images["network_after"] = _plot_network(
        graph,
        after_pressures["pressure"],
        images_dir / "network_after.png",
        "Network Pressure After Isolation",
    )

    context = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "Benchmark": results.get("benchmark"),
            "Scenario source": results.get("scenario_dir"),
            "Input file": results.get("inp_path"),
            "Isolated pipe": results.get("isolated_pipe"),
            "Mean pressure before": round(float(results.get("before_mean_pressure", 0.0)), 4),
            "Mean pressure after": round(float(results.get("after_mean_pressure", 0.0)), 4),
            "Max flow before": round(float(results.get("max_flow_before", 0.0)), 4),
            "Max flow after": round(float(results.get("max_flow_after", 0.0)), 4),
            "Max flow delta": round(float(results.get("max_flow_delta", results.get("max_flow_after", 0.0) - results.get("max_flow_before", 0.0))), 4),
            "MST cost before": round(float(results.get("mst_cost_before", 0.0)), 6),
            "MST cost after": round(float(results.get("mst_cost_after", 0.0)), 6),
            "MST cost delta": round(float(results.get("mst_cost_delta", results.get("mst_cost_after", 0.0) - results.get("mst_cost_before", 0.0))), 6),
            "static_pressure_loss": round(float(results.get("static_vs_gnn", {}).get("static_pressure_loss", 0.0)), 6) if isinstance(results.get("static_vs_gnn"), dict) else 0.0,
            "gnn_pressure_loss": round(float(results.get("static_vs_gnn", {}).get("gnn_pressure_loss", 0.0)), 6) if isinstance(results.get("static_vs_gnn"), dict) else 0.0,
            "pressure_loss_reduction": round(float(results.get("static_vs_gnn", {}).get("pressure_loss_reduction", 0.0)), 6) if isinstance(results.get("static_vs_gnn"), dict) else 0.0,
            "pressure_loss_reduction_pct": round(float(results.get("static_vs_gnn", {}).get("pressure_loss_reduction_pct", 0.0)), 2) if isinstance(results.get("static_vs_gnn"), dict) else 0.0,
        },
        "recommendation": results.get("recommendation", {}),
        "plots": {name: _figure_to_base64(plt.imread(path)) if False else None for name, path in images.items()},
        "image_paths": images,
    }

    # Render the base64 plots from the actual files to avoid keeping large figures alive.
    encoded_plots: Dict[str, str] = {}
    for name, path in images.items():
        encoded_plots[name] = base64.b64encode(path.read_bytes()).decode("ascii")
    context["plots"] = encoded_plots

    html_path = run_dir / "report.html"
    md_path = run_dir / "report.md"
    _render_html_report(context, html_path)
    _render_markdown_report(context, md_path)

    return ReportArtifacts(output_dir=run_dir, images=images, report_html=html_path, report_md=md_path)
