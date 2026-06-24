from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn

from ai_layer.dataset_loader import LeakDBDatasetLoader, ScenarioBundle
from ai_layer.stgcn_model import STGCN, anomaly_scores


LOGGER = logging.getLogger("leakdb_colab")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _resolve_raw_root(data_root: str | Path | None) -> Path:
    if data_root is not None:
        path = Path(data_root).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"Could not find data root: {path}")
    candidates = [
        Path.cwd() / "data" / "raw" / "LeakDB",
        Path.cwd() / "data" / "raw",
        Path.cwd() / "data",
        Path(__file__).resolve().parent / "data" / "raw" / "LeakDB",
        Path(__file__).resolve().parent / "data" / "raw",
        Path(__file__).resolve().parent / "data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate LeakDB. Pass --data-root explicitly.")


def _results_dir(path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _device(name: str) -> torch.device:
    name = name.lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return path


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2), encoding="utf-8")


def _array_stats(arr: np.ndarray) -> Dict[str, float]:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def _build_window_labels(labels: Optional[pd.Series], num_snapshots: int, window_size: int) -> np.ndarray:
    if labels is None or labels.empty:
        return np.zeros(max(0, num_snapshots - window_size + 1), dtype=np.int64)
    values = labels.fillna(0.0).to_numpy(dtype=np.float32, copy=True)
    return np.asarray(
        [int(np.max(values[i : i + window_size]) > 0.0) for i in range(num_snapshots - window_size + 1)],
        dtype=np.int64,
    )


@dataclass
class ScenarioData:
    scenario_name: str
    features: List[np.ndarray]
    targets: List[np.ndarray]
    window_labels: np.ndarray
    metadata: Dict[str, object]

    @property
    def num_windows(self) -> int:
        return int(self.window_labels.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.features[0].shape[0]) if self.features else 0


@dataclass
class EarlyStopping:
    patience: int = 15
    restore_best_weights: bool = True
    best_val_loss: float = float("inf")
    best_epoch: int = -1
    counter: int = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    def step(self, epoch: int, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_val_loss - 1e-8:
            self.best_val_loss = float(val_loss)
            self.best_epoch = int(epoch)
            self.counter = 0
            if self.restore_best_weights:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience


def _load_bundles(loader: LeakDBDatasetLoader, benchmark: str, max_scenarios: int) -> List[ScenarioBundle]:
    selected = loader.list_scenarios(benchmark)[:max_scenarios]
    if not selected:
        raise FileNotFoundError(f"No scenarios found for benchmark {benchmark}")
    return loader.load_scenarios(
        benchmark_name=benchmark,
        scenario_names=selected,
        max_scenarios=max_scenarios,
        feature_mode="pressure_only",
        target_mode="pressure_next",
        include_raw_frames=True,
    )


def _to_scenario_data(bundle: ScenarioBundle, window_size: int) -> ScenarioData:
    labels = bundle.metadata.get("labels")
    label_series = labels if isinstance(labels, pd.Series) else None
    return ScenarioData(
        scenario_name=bundle.scenario_name,
        features=[np.asarray(x, dtype=np.float32) for x in bundle.signal.features],
        targets=[np.asarray(y, dtype=np.float32) for y in bundle.signal.targets],
        window_labels=_build_window_labels(label_series, len(bundle.signal.features), window_size),
        metadata=dict(bundle.metadata),
    )


def _split_normal_windows(window_labels: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    normal = np.flatnonzero(window_labels == 0)
    total = int(window_labels.shape[0])
    if normal.size >= 4:
        cut = max(1, min(normal.size - 1, int(round(normal.size * train_ratio))))
        return normal[:cut], normal[cut:]
    cut = max(1, min(total - 1, int(round(total * train_ratio))))
    return np.arange(0, cut, dtype=np.int64), np.arange(cut, total, dtype=np.int64)


def _snapshot_indices(window_indices: Sequence[int], window_size: int) -> List[int]:
    out = set()
    for start in window_indices:
        for offset in range(window_size):
            out.add(int(start) + offset)
    return sorted(out)


def _fit_scaler(scenarios: Sequence[ScenarioData], train_indices: Sequence[np.ndarray], window_size: int) -> StandardScaler:
    mats: List[np.ndarray] = []
    for scenario, indices in zip(scenarios, train_indices):
        for snapshot_index in _snapshot_indices(indices.tolist(), window_size):
            mats.append(scenario.features[snapshot_index].reshape(-1, scenario.features[snapshot_index].shape[-1]))
    train_matrix = np.concatenate(mats, axis=0)
    return StandardScaler().fit(train_matrix)


def _apply_scaler(scenarios: Sequence[ScenarioData], scaler: StandardScaler) -> None:
    for scenario in scenarios:
        scenario.features = [
            scaler.transform(snapshot.reshape(-1, snapshot.shape[-1])).reshape(snapshot.shape).astype(np.float32)
            for snapshot in scenario.features
        ]


def _window_batch(scenario: ScenarioData, indices: Sequence[int], window_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = [np.stack(scenario.features[i : i + window_size], axis=0) for i in indices]
    ys = [scenario.targets[i + window_size - 1] for i in indices]
    return torch.tensor(np.stack(xs, axis=0), dtype=torch.float32), torch.tensor(np.stack(ys, axis=0), dtype=torch.float32)


def _chunks(indices: Sequence[int], batch_size: int) -> Iterable[Sequence[int]]:
    for i in range(0, len(indices), batch_size):
        yield indices[i : i + batch_size]


def _graph_to_edge_index(graph: object, node_ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    node_to_idx = {str(node): idx for idx, node in enumerate(node_ids)}
    edges: List[Tuple[int, int]] = []
    weights: List[float] = []
    for u, v, data in graph.edges(data=True):  # type: ignore[attr-defined]
        u_key, v_key = str(u), str(v)
        if u_key not in node_to_idx or v_key not in node_to_idx:
            continue
        w = float(data.get("capacity", 1.0))
        if w <= 0:
            continue
        i, j = node_to_idx[u_key], node_to_idx[v_key]
        edges.append((i, j))
        edges.append((j, i))
        weights.extend([w, w])
    if not edges:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    return np.asarray(edges, dtype=np.int64).T, np.asarray(weights, dtype=np.float32)


def _create_model(num_nodes: int, in_channels: int, latent_dim: int) -> STGCN:
    return STGCN(num_nodes=num_nodes, in_channels=in_channels, hidden_channels=latent_dim, output_channels=1, temporal_kernel_size=3, dropout=0.1)


def _train_model(
    model: STGCN,
    scenarios: Sequence[ScenarioData],
    train_indices: Sequence[np.ndarray],
    val_indices: Sequence[np.ndarray],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
    window_size: int,
    epochs: int,
    batch_size: int,
    patience: int = 15,
) -> Dict[str, object]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    stopper = EarlyStopping(patience=patience)
    train_history: List[float] = []
    val_history: List[float] = []

    for epoch in range(epochs):
        model.train()
        batch_losses: List[float] = []
        for scenario, idxs in zip(scenarios, train_indices):
            for batch in _chunks(idxs.tolist(), batch_size):
                x, y = _window_batch(scenario, batch, window_size)
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(x, edge_index=edge_index, edge_weight=edge_weight)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        val_loss = _eval_loss(model, scenarios, val_indices, edge_index, edge_weight, device, window_size, batch_size)
        train_history.append(train_loss)
        val_history.append(val_loss)
        LOGGER.info("Epoch %d/%d | train_loss=%.6f | val_loss=%.6f", epoch + 1, epochs, train_loss, val_loss)
        if stopper.step(epoch, val_loss, model):
            LOGGER.info("Early stopping at epoch %d", epoch + 1)
            break

    if stopper.restore_best_weights and stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    return {
        "train_loss_history": train_history,
        "val_loss_history": val_history,
        "best_epoch": stopper.best_epoch + 1 if stopper.best_epoch >= 0 else len(val_history),
        "best_val_loss": stopper.best_val_loss,
        "epochs_ran": len(val_history),
    }


@torch.no_grad()
def _eval_loss(
    model: STGCN,
    scenarios: Sequence[ScenarioData],
    indices_per_scenario: Sequence[np.ndarray],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
    window_size: int,
    batch_size: int,
) -> float:
    model.eval()
    criterion = nn.MSELoss()
    losses: List[float] = []
    for scenario, idxs in zip(scenarios, indices_per_scenario):
        if idxs.size == 0:
            continue
        for batch in _chunks(idxs.tolist(), batch_size):
            x, y = _window_batch(scenario, batch, window_size)
            x, y = x.to(device), y.to(device)
            pred = model(x, edge_index=edge_index, edge_weight=edge_weight)
            losses.append(float(criterion(pred, y).item()))
    return float(np.mean(losses)) if losses else float("nan")


def _plot_training(train_loss: Sequence[float], val_loss: Sequence[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(np.arange(1, len(train_loss) + 1), train_loss, label="Train Loss", color="#1f77b4", linewidth=2.2)
    ax.plot(np.arange(1, len(val_loss) + 1), val_loss, label="Validation Loss", color="#d62728", linewidth=2.2)
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_figure(fig, path)


def _plot_hist_box(normal_scores: np.ndarray, leak_scores: np.ndarray, hist_path: Path, box_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = min(40, max(12, int(np.sqrt(max(len(normal_scores), len(leak_scores))))))
    ax.hist(normal_scores, bins=bins, alpha=0.65, label="Normal", color="#1f77b4", density=True)
    ax.hist(leak_scores, bins=bins, alpha=0.65, label="Leak", color="#d62728", density=True)
    ax.set_title("Normal vs Leak Anomaly Score Distribution")
    ax.set_xlabel("Mean Absolute Error per Window")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.2)
    ax.legend()
    _save_figure(fig, hist_path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot([normal_scores, leak_scores], labels=["Normal", "Leak"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#1f77b4")
    bp["boxes"][1].set_facecolor("#d62728")
    ax.set_title("Normal vs Leak Anomaly Score Boxplot")
    ax.set_ylabel("Mean Absolute Error per Window")
    ax.grid(True, axis="y", alpha=0.2)
    _save_figure(fig, box_path)


def _plot_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Normal", "Leak"])
    ax.set_yticks([0, 1], labels=["Normal", "Leak"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_figure(fig, path)


def _plot_reconstruction(actual: np.ndarray, predicted: np.ndarray, node_ids: Sequence[str], path: Path, title: str) -> None:
    if actual.size == 0 or predicted.size == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No reconstruction windows available", ha="center", va="center")
        ax.axis("off")
        _save_figure(fig, path)
        return
    node_order = np.argsort(np.var(actual, axis=0))[::-1][: min(4, actual.shape[1])]
    fig, axes = plt.subplots(len(node_order), 1, figsize=(11, max(3.0, 2.6 * len(node_order))), sharex=True)
    if len(node_order) == 1:
        axes = [axes]
    x = np.arange(actual.shape[0])
    for ax, idx in zip(axes, node_order):
        ax.plot(x, actual[:, idx], label="Actual", color="#1f77b4", linewidth=2.0)
        ax.plot(x, predicted[:, idx], label="Predicted", color="#d62728", linewidth=2.0, linestyle="--")
        ax.set_ylabel(f"Node {node_ids[idx]}")
        ax.grid(True, alpha=0.2)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Window Index")
    axes[0].legend(loc="upper right")
    _save_figure(fig, path)


def _evaluate_windows(
    model: STGCN,
    scenarios: Sequence[ScenarioData],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
    window_size: int,
    batch_size: int,
) -> Dict[str, object]:
    model.eval()
    all_scores: List[float] = []
    all_labels: List[int] = []
    scenario_rows: List[Dict[str, object]] = []
    normal_actuals: List[np.ndarray] = []
    normal_preds: List[np.ndarray] = []
    leak_actuals: List[np.ndarray] = []
    leak_preds: List[np.ndarray] = []
    node_ids = list(scenarios[0].metadata["node_ids"])

    for scenario in scenarios:
        scenario_scores: List[float] = []
        for batch in _chunks(list(range(scenario.num_windows)), batch_size):
            x, y = _window_batch(scenario, batch, window_size)
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x, edge_index=edge_index, edge_weight=edge_weight)
                scores = anomaly_scores(pred, y).mean(dim=1)
            scores_np = scores.detach().cpu().numpy()
            labels_np = scenario.window_labels[np.asarray(batch, dtype=np.int64)]
            all_scores.extend(scores_np.tolist())
            all_labels.extend(labels_np.tolist())
            scenario_scores.extend(scores_np.tolist())
            for i, label in enumerate(labels_np.tolist()):
                if label == 0 and len(normal_actuals) < 60:
                    normal_actuals.append(y[i].detach().cpu().numpy())
                    normal_preds.append(pred[i].detach().cpu().numpy())
                if label == 1 and len(leak_actuals) < 60:
                    leak_actuals.append(y[i].detach().cpu().numpy())
                    leak_preds.append(pred[i].detach().cpu().numpy())
        scenario_rows.append(
            {
                "scenario_name": scenario.scenario_name,
                "mean_score": float(np.mean(scenario_scores)) if scenario_scores else float("nan"),
                "max_score": float(np.max(scenario_scores)) if scenario_scores else float("nan"),
                "windows": int(len(scenario_scores)),
                "normal_windows": int(np.sum(scenario.window_labels == 0)),
                "leak_windows": int(np.sum(scenario.window_labels == 1)),
            }
        )

    all_scores_arr = np.asarray(all_scores, dtype=np.float32)
    all_labels_arr = np.asarray(all_labels, dtype=np.int64)
    normal_scores = all_scores_arr[all_labels_arr == 0]
    leak_scores = all_scores_arr[all_labels_arr == 1]
    mean_normal = float(np.mean(normal_scores)) if normal_scores.size else float("nan")
    mean_leak = float(np.mean(leak_scores)) if leak_scores.size else float("nan")
    std_normal = float(np.std(normal_scores)) if normal_scores.size else float("nan")
    std_leak = float(np.std(leak_scores)) if leak_scores.size else float("nan")

    precision, recall, thresholds = precision_recall_curve(all_labels_arr, all_scores_arr)
    f1_values = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)
    if thresholds.size:
        best_idx = int(np.nanargmax(f1_values[:-1]))
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_values[:-1][best_idx])
    else:
        best_threshold = float(np.median(all_scores_arr))
        best_f1 = float(f1_score(all_labels_arr, (all_scores_arr >= best_threshold).astype(int), zero_division=0))

    predicted = (all_scores_arr >= best_threshold).astype(int)
    metrics = {
        "mean_normal": mean_normal,
        "mean_leak": mean_leak,
        "std_normal": std_normal,
        "std_leak": std_leak,
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "precision": float(precision_score(all_labels_arr, predicted, zero_division=0)),
        "recall": float(recall_score(all_labels_arr, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(all_labels_arr, all_scores_arr)) if np.unique(all_labels_arr).size > 1 else float("nan"),
        "pr_auc": float(average_precision_score(all_labels_arr, all_scores_arr)) if np.unique(all_labels_arr).size > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(all_labels_arr, predicted, labels=[0, 1]),
    }
    return {
        "scores": all_scores_arr,
        "labels": all_labels_arr,
        "metrics": metrics,
        "scenario_rows": scenario_rows,
        "normal_actuals": np.asarray(normal_actuals, dtype=np.float32),
        "normal_preds": np.asarray(normal_preds, dtype=np.float32),
        "leak_actuals": np.asarray(leak_actuals, dtype=np.float32),
        "leak_preds": np.asarray(leak_preds, dtype=np.float32),
        "node_ids": node_ids,
    }


def _build_report_text(training: Mapping[str, object], evaluation: Mapping[str, object], latent_rows: Sequence[Mapping[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# Final LeakDB Evaluation Report")
    lines.append("")
    lines.append("## Training Summary")
    lines.append(f"- Benchmark: {training.get('benchmark_name')}")
    lines.append(f"- Scenarios used: {training.get('max_scenarios')}")
    lines.append(f"- Window size: {training.get('window_size')}")
    lines.append(f"- Epochs ran: {training.get('epochs_ran')}")
    lines.append(f"- Best epoch: {training.get('best_epoch')}")
    lines.append(f"- Best validation loss: {training.get('best_val_loss'):.6f}")
    lines.append(f"- Training loss start: {training.get('train_loss_history', [float('nan')])[0]:.6f}")
    lines.append(f"- Training loss end: {training.get('train_loss_history', [float('nan')])[-1]:.6f}")
    lines.append("")
    lines.append("## Validation Summary")
    lines.append(f"- Validation loss plot: `validation_training_loss.png`")
    lines.append(f"- Early stopping patience: 15")
    lines.append(f"- Normalization before: mean={training['normalization_before']['mean']:.6f}, std={training['normalization_before']['std']:.6f}, min={training['normalization_before']['min']:.6f}, max={training['normalization_before']['max']:.6f}")
    lines.append(f"- Normalization after: mean={training['normalization_after']['mean']:.6f}, std={training['normalization_after']['std']:.6f}, min={training['normalization_after']['min']:.6f}, max={training['normalization_after']['max']:.6f}")
    lines.append("")
    lines.append("## Detection Metrics")
    lines.append(f"- Best threshold: {evaluation['best_threshold']:.6f}")
    lines.append(f"- Best F1: {evaluation['best_f1']:.6f}")
    lines.append(f"- Precision: {evaluation['precision']:.6f}")
    lines.append(f"- Recall: {evaluation['recall']:.6f}")
    lines.append(f"- ROC-AUC: {evaluation['roc_auc']:.6f}")
    lines.append(f"- PR-AUC: {evaluation['pr_auc']:.6f}")
    lines.append(f"- Mean normal score: {evaluation['mean_normal']:.6f}")
    lines.append(f"- Mean leak score: {evaluation['mean_leak']:.6f}")
    lines.append(f"- Std normal score: {evaluation['std_normal']:.6f}")
    lines.append(f"- Std leak score: {evaluation['std_leak']:.6f}")
    lines.append("")
    lines.append("## Latent Dimension Comparison")
    if latent_rows:
        lines.append("| latent_dim | train_loss | val_loss | precision | recall | f1 | roc_auc | pr_auc |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in latent_rows:
            lines.append(
                f"| {int(row['latent_dim'])} | {row['train_loss']:.6f} | {row['val_loss']:.6f} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['roc_auc']:.6f} | {row['pr_auc']:.6f} |"
            )
    else:
        lines.append("- Latent experiment not run.")
    lines.append("")
    lines.append("## Recommendation")
    if latent_rows:
        best = sorted(latent_rows, key=lambda r: (-float(r["f1"]), -float(r["roc_auc"]), float(r["val_loss"])))[0]
        lines.append(
            f"- Best model: latent_dim={best['latent_dim']} with F1={best['f1']:.6f}, ROC-AUC={best['roc_auc']:.6f}, val_loss={best['val_loss']:.6f}."
        )
    else:
        lines.append("- Best model recommendation unavailable because latent comparison was not run.")
    lines.append("- If normal and leak means are still close, the model is not cleanly separating leaks yet.")
    lines.append("")
    lines.append("## Saved Files")
    lines.append("- validation_training_loss.png")
    lines.append("- histogram_normal_vs_leak.png")
    lines.append("- boxplot_normal_vs_leak.png")
    lines.append("- confusion_matrix.png")
    lines.append("- reconstruction_normal.png")
    lines.append("- reconstruction_leak.png")
    lines.append("- latent_comparison.csv")
    lines.append("- latent_comparison.png")
    return "\n".join(lines)


def _prepare_dataset(
    benchmark: str,
    max_scenarios: int,
    window_size: int,
    data_root: str | Path | None,
) -> Tuple[List[ScenarioData], List[np.ndarray], List[np.ndarray], StandardScaler, Dict[str, object]]:
    loader = LeakDBDatasetLoader(_resolve_raw_root(data_root))
    bundles = _load_bundles(loader, benchmark, max_scenarios)
    scenarios = [_to_scenario_data(bundle, window_size) for bundle in bundles]
    train_indices: List[np.ndarray] = []
    val_indices: List[np.ndarray] = []
    for scenario in scenarios:
        tr, va = _split_normal_windows(scenario.window_labels)
        train_indices.append(tr)
        val_indices.append(va)
    train_matrix = np.concatenate(
        [
            scenarios[i].features[snapshot_index].reshape(-1, scenarios[i].features[snapshot_index].shape[-1])
            for i, idxs in enumerate(train_indices)
            for snapshot_index in _snapshot_indices(idxs.tolist(), window_size)
        ],
        axis=0,
    )
    before = _array_stats(train_matrix)
    scaler = StandardScaler().fit(train_matrix)
    _apply_scaler(scenarios, scaler)
    after_matrix = np.concatenate(
        [
            scenarios[i].features[snapshot_index].reshape(-1, scenarios[i].features[snapshot_index].shape[-1])
            for i, idxs in enumerate(train_indices)
            for snapshot_index in _snapshot_indices(idxs.tolist(), window_size)
        ],
        axis=0,
    )
    after = _array_stats(after_matrix)
    summary = {
        "benchmark_name": benchmark,
        "max_scenarios": max_scenarios,
        "window_size": window_size,
        "scenario_names": [s.scenario_name for s in scenarios],
        "num_nodes": scenarios[0].num_nodes,
        "feature_channels": int(scenarios[0].features[0].shape[-1]),
        "train_windows": int(sum(len(x) for x in train_indices)),
        "val_windows": int(sum(len(x) for x in val_indices)),
        "normalization_before": before,
        "normalization_after": after,
    }
    return scenarios, train_indices, val_indices, scaler, summary


def _load_checkpoint(path: Path, device: torch.device) -> Tuple[STGCN, Dict[str, object]]:
    checkpoint = torch.load(path, map_location=device)
    cfg = dict(checkpoint["model_config"])
    model = _create_model(int(cfg["num_nodes"]), int(cfg["in_channels"]), int(cfg["latent_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


def _compute_metrics(all_labels: np.ndarray, scores: np.ndarray) -> Dict[str, object]:
    precision, recall, thresholds = precision_recall_curve(all_labels, scores)
    f1s = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)
    if thresholds.size:
        best = int(np.nanargmax(f1s[:-1]))
        threshold = float(thresholds[best])
        best_f1 = float(f1s[:-1][best])
    else:
        threshold = float(np.median(scores))
        best_f1 = float(f1_score(all_labels, (scores >= threshold).astype(int), zero_division=0))
    pred = (scores >= threshold).astype(int)
    return {
        "best_threshold": threshold,
        "best_f1": best_f1,
        "precision": float(precision_score(all_labels, pred, zero_division=0)),
        "recall": float(recall_score(all_labels, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(all_labels, scores)) if np.unique(all_labels).size > 1 else float("nan"),
        "pr_auc": float(average_precision_score(all_labels, scores)) if np.unique(all_labels).size > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(all_labels, pred, labels=[0, 1]),
        "predicted": pred,
    }


def _evaluate_model(
    model: STGCN,
    scenarios: Sequence[ScenarioData],
    edge_index_t: torch.Tensor,
    edge_weight_t: torch.Tensor,
    device: torch.device,
    window_size: int,
    batch_size: int,
) -> Dict[str, object]:
    all_scores: List[float] = []
    all_labels: List[int] = []
    scenario_rows: List[Dict[str, object]] = []
    normal_actuals: List[np.ndarray] = []
    normal_preds: List[np.ndarray] = []
    leak_actuals: List[np.ndarray] = []
    leak_preds: List[np.ndarray] = []
    node_ids = list(scenarios[0].metadata["node_ids"])
    for scenario in scenarios:
        scenario_scores: List[float] = []
        for batch in _chunks(list(range(scenario.num_windows)), batch_size):
            x, y = _window_batch(scenario, batch, window_size)
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x, edge_index=edge_index_t, edge_weight=edge_weight_t)
                window_scores = anomaly_scores(pred, y).mean(dim=1)
            scores_np = window_scores.detach().cpu().numpy()
            labels_np = scenario.window_labels[np.asarray(batch, dtype=np.int64)]
            all_scores.extend(scores_np.tolist())
            all_labels.extend(labels_np.tolist())
            scenario_scores.extend(scores_np.tolist())
            for i, label in enumerate(labels_np.tolist()):
                if label == 0 and len(normal_actuals) < 60:
                    normal_actuals.append(y[i].detach().cpu().numpy())
                    normal_preds.append(pred[i].detach().cpu().numpy())
                if label == 1 and len(leak_actuals) < 60:
                    leak_actuals.append(y[i].detach().cpu().numpy())
                    leak_preds.append(pred[i].detach().cpu().numpy())
        scenario_rows.append(
            {
                "scenario_name": scenario.scenario_name,
                "mean_score": float(np.mean(scenario_scores)) if scenario_scores else float("nan"),
                "max_score": float(np.max(scenario_scores)) if scenario_scores else float("nan"),
                "windows": int(len(scenario_scores)),
                "normal_windows": int(np.sum(scenario.window_labels == 0)),
                "leak_windows": int(np.sum(scenario.window_labels == 1)),
            }
        )
    scores = np.asarray(all_scores, dtype=np.float32)
    labels = np.asarray(all_labels, dtype=np.int64)
    metrics = _compute_metrics(labels, scores)
    metrics.update(
        {
            "mean_normal": float(np.mean(scores[labels == 0])) if np.any(labels == 0) else float("nan"),
            "mean_leak": float(np.mean(scores[labels == 1])) if np.any(labels == 1) else float("nan"),
            "std_normal": float(np.std(scores[labels == 0])) if np.any(labels == 0) else float("nan"),
            "std_leak": float(np.std(scores[labels == 1])) if np.any(labels == 1) else float("nan"),
        }
    )
    return {
        "scores": scores,
        "labels": labels,
        "metrics": metrics,
        "scenario_rows": scenario_rows,
        "normal_actuals": np.asarray(normal_actuals, dtype=np.float32),
        "normal_preds": np.asarray(normal_preds, dtype=np.float32),
        "leak_actuals": np.asarray(leak_actuals, dtype=np.float32),
        "leak_preds": np.asarray(leak_preds, dtype=np.float32),
        "node_ids": node_ids,
    }


def _plot_latent(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    dims = [int(r["latent_dim"]) for r in rows]
    axes[0].bar(dims, [float(r["train_loss"]) for r in rows], color="#1f77b4")
    axes[0].set_title("Train Loss")
    axes[1].bar(dims, [float(r["val_loss"]) for r in rows], color="#2ca02c")
    axes[1].set_title("Validation Loss")
    axes[2].bar(dims, [float(r["f1"]) for r in rows], color="#d62728")
    axes[2].set_title("F1")
    for ax in axes:
        ax.set_xlabel("Latent Dim")
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_xticks(dims)
    fig.tight_layout()
    _save_figure(fig, path)


def _run_latent_experiment(
    benchmark: str,
    max_scenarios: int,
    window_size: int,
    batch_size: int,
    latent_dims: Sequence[int],
    data_root: str | Path | None,
    results_dir: Path,
    device_name: str,
    epochs: int,
) -> List[Dict[str, object]]:
    device = _device(device_name)
    scenarios, train_indices, val_indices, scaler, summary = _prepare_dataset(benchmark, max_scenarios, window_size, data_root)
    first = scenarios[0]
    edge_index, edge_weight = _graph_to_edge_index(first.metadata["graph"], first.metadata["node_ids"])
    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)
    edge_weight_t = torch.tensor(edge_weight, dtype=torch.float32, device=device)
    rows: List[Dict[str, object]] = []
    for latent_dim in latent_dims:
        model = _create_model(first.num_nodes, int(first.features[0].shape[-1]), int(latent_dim)).to(device)
        training = _train_model(model, scenarios, train_indices, val_indices, edge_index_t, edge_weight_t, device, window_size, epochs, batch_size)
        evaluation = _evaluate_model(model, scenarios, edge_index_t, edge_weight_t, device, window_size, batch_size)
        metrics = evaluation["metrics"]
        rows.append(
            {
                "latent_dim": int(latent_dim),
                "train_loss": float(training["train_loss_history"][-1]) if training["train_loss_history"] else float("nan"),
                "val_loss": float(training["best_val_loss"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["best_f1"]),
                "roc_auc": float(metrics["roc_auc"]),
                "pr_auc": float(metrics["pr_auc"]),
            }
        )
    pd.DataFrame(rows).to_csv(results_dir / "latent_comparison.csv", index=False)
    _plot_latent(rows, results_dir / "latent_comparison.png")
    _save_json(results_dir / "latent_experiment_summary.json", {"latent_models": rows, "best_model": sorted(rows, key=lambda r: (-float(r["f1"]), -float(r["roc_auc"]), float(r["val_loss"])))[0]})
    return rows


def run_evaluate(
    benchmark: str,
    max_scenarios: int,
    window_size: int,
    batch_size: int,
    data_root: str | Path | None,
    results_dir: Path,
    device_name: str,
    run_latent: bool,
    epochs: int,
) -> Dict[str, object]:
    device = _device(device_name)
    scenarios, _, _, scaler, summary = _prepare_dataset(benchmark, max_scenarios, window_size, data_root)
    with open(results_dir / "scaler.pkl", "rb") as f:
        saved_scaler: StandardScaler = pickle.load(f)
    _apply_scaler(scenarios, saved_scaler)
    model, cfg = _load_checkpoint(results_dir / "best_model.pt", device)
    first = scenarios[0]
    edge_index, edge_weight = _graph_to_edge_index(first.metadata["graph"], first.metadata["node_ids"])
    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)
    edge_weight_t = torch.tensor(edge_weight, dtype=torch.float32, device=device)
    evaluation = _evaluate_model(model, scenarios, edge_index_t, edge_weight_t, device, window_size, batch_size)
    metrics = evaluation["metrics"]
    normal_scores = evaluation["scores"][evaluation["labels"] == 0]
    leak_scores = evaluation["scores"][evaluation["labels"] == 1]
    _plot_hist_box(normal_scores, leak_scores, results_dir / "histogram_normal_vs_leak.png", results_dir / "boxplot_normal_vs_leak.png")
    _plot_confusion_matrix(metrics["confusion_matrix"], results_dir / "confusion_matrix.png")
    _plot_reconstruction(evaluation["normal_actuals"], evaluation["normal_preds"], evaluation["node_ids"], results_dir / "reconstruction_normal.png", "Representative Normal Reconstruction")
    _plot_reconstruction(evaluation["leak_actuals"], evaluation["leak_preds"], evaluation["node_ids"], results_dir / "reconstruction_leak.png", "Representative Leak Reconstruction")
    scenario_df = pd.DataFrame(evaluation["scenario_rows"])
    scenario_df.to_csv(results_dir / "scenario_metrics.csv", index=False)
    _save_json(results_dir / "evaluation_metrics.json", {**metrics, "scenario_metrics": evaluation["scenario_rows"], "n_windows": int(evaluation["labels"].size), "n_normal_windows": int(np.sum(evaluation["labels"] == 0)), "n_leak_windows": int(np.sum(evaluation["labels"] == 1))})
    if run_latent:
        latent_rows = _run_latent_experiment(benchmark, max_scenarios, window_size, batch_size, [16, 32, 64], data_root, results_dir, device_name, epochs)
    else:
        latent_rows = []
    training_summary = json.loads((results_dir / "training_summary.json").read_text(encoding="utf-8"))
    training_summary["normalization_before"] = training_summary.get("normalization_before") or summary["normalization_before"]
    training_summary["normalization_after"] = training_summary.get("normalization_after") or summary["normalization_after"]
    _save_json(results_dir / "evaluation_summary.json", {**metrics, "latent_experiment_ran": run_latent, "model_config": cfg})
    report_text = _build_report_text(training_summary, metrics, latent_rows)
    (results_dir / "final_evaluation_report.md").write_text(report_text, encoding="utf-8")
    return metrics


def run_report(results_dir: Path) -> Path:
    training = json.loads((results_dir / "training_summary.json").read_text(encoding="utf-8"))
    evaluation = json.loads((results_dir / "evaluation_metrics.json").read_text(encoding="utf-8"))
    latent_rows: List[Dict[str, object]] = []
    latent_path = results_dir / "latent_comparison.csv"
    if latent_path.exists():
        latent_rows = pd.read_csv(latent_path).to_dict(orient="records")
    report_text = _build_report_text(training, evaluation, latent_rows)
    report_path = results_dir / "final_evaluation_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    LOGGER.info("Wrote report to %s", report_path)
    return report_path


def run_train(
    benchmark: str,
    max_scenarios: int,
    window_size: int,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    data_root: str | Path | None,
    results_dir: Path,
    device_name: str,
) -> Dict[str, object]:
    device = _device(device_name)
    scenarios, train_indices, val_indices, scaler, summary = _prepare_dataset(benchmark, max_scenarios, window_size, data_root)
    first = scenarios[0]
    edge_index, edge_weight = _graph_to_edge_index(first.metadata["graph"], first.metadata["node_ids"])
    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)
    edge_weight_t = torch.tensor(edge_weight, dtype=torch.float32, device=device)
    model = _create_model(first.num_nodes, int(first.features[0].shape[-1]), latent_dim).to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    training = _train_model(model, scenarios, train_indices, val_indices, edge_index_t, edge_weight_t, device, window_size, epochs, batch_size)
    _plot_training(training["train_loss_history"], training["val_loss_history"], results_dir / "validation_training_loss.png")
    torch.save({"model_state_dict": model.state_dict(), "model_config": {"num_nodes": first.num_nodes, "in_channels": int(first.features[0].shape[-1]), "latent_dim": latent_dim}}, results_dir / "best_model.pt")
    with open(results_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    payload = {**summary, **training, "latent_dim": latent_dim}
    _save_json(results_dir / "training_summary.json", payload)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LeakDB Hanoi_CMH Colab pipeline")
    parser.add_argument("--mode", choices=["train", "evaluate", "report", "all"], default="all")
    parser.add_argument("--benchmark", default="Hanoi_CMH")
    parser.add_argument("--max-scenarios", type=int, default=30)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-latent", action="store_true")
    return parser


def main() -> None:
    _configure_logging()
    args = _build_parser().parse_args()
    results_dir = _results_dir(args.results_dir)
    if args.mode in {"train", "all"}:
        run_train(args.benchmark, args.max_scenarios, args.window_size, args.epochs, args.batch_size, args.latent_dim, args.data_root, results_dir, args.device)
    if args.mode in {"evaluate", "all"}:
        run_evaluate(args.benchmark, args.max_scenarios, args.window_size, args.batch_size, args.data_root, results_dir, args.device, run_latent=True if args.mode == "all" else args.run_latent, epochs=args.epochs)
    if args.mode in {"report", "all"}:
        run_report(results_dir)


if __name__ == "__main__":
    main()
