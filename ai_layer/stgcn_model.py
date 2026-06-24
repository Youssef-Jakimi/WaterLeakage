"""Spatio-temporal graph convolution network for pressure anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import math

import numpy as np
import torch
from torch import Tensor, nn

try:  # Optional dependency.
    from torch_geometric_temporal.nn.recurrent import A3TGCN  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    A3TGCN = None


def _to_dense_adjacency(edge_index: Tensor, edge_weight: Optional[Tensor], num_nodes: int, device: torch.device) -> Tensor:
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    if edge_index.numel() == 0:
        adjacency.fill_diagonal_(1.0)
        return adjacency
    src = edge_index[0].long()
    dst = edge_index[1].long()
    if edge_weight is None:
        values = torch.ones(src.shape[0], dtype=torch.float32, device=device)
    else:
        values = edge_weight.to(device=device, dtype=torch.float32)
    adjacency[src, dst] = values
    adjacency.fill_diagonal_(1.0)
    degree = adjacency.sum(dim=1).clamp_min(1e-6)
    inv_sqrt_degree = degree.pow(-0.5)
    normalized = inv_sqrt_degree.unsqueeze(1) * adjacency * inv_sqrt_degree.unsqueeze(0)
    return normalized


class SpatialGraphConvolution(nn.Module):
    """Simple graph convolution implemented with a normalized dense adjacency."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        # x: [batch, nodes, features]
        adjacency = adjacency.contiguous()
        x = x.contiguous()
        # Vectorized batch propagation: [nodes, nodes] x [batch, nodes, features] -> [batch, nodes, features]
        propagated = torch.einsum("ij,bjf->bif", adjacency, x)
        out = self.linear(propagated)
        out = self.norm(out)
        return torch.relu(self.dropout(out))


class TemporalConvBlock(nn.Module):
    """Temporal convolution over the sequence dimension for each node."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, time, nodes, features]
        batch, time_steps, nodes, features = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch * nodes, features, time_steps)
        x = self.conv(x)
        x = self.norm(x)
        x = torch.relu(self.dropout(x))
        time_steps = x.shape[-1]
        x = x.reshape(batch, nodes, -1, time_steps).permute(0, 3, 1, 2)
        return x


class STGCN(nn.Module):
    """Compact STGCN-style predictor for node pressures.

    The model accepts input shaped as:
      - [time, nodes, features]
      - [batch, time, nodes, features]

    and returns node-wise pressure predictions.
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int = 2,
        hidden_channels: int = 32,
        output_channels: int = 1,
        temporal_kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.temporal_kernel_size = temporal_kernel_size

        self.temporal_in = TemporalConvBlock(in_channels, hidden_channels, temporal_kernel_size, dropout)
        self.spatial = SpatialGraphConvolution(hidden_channels, hidden_channels, dropout)
        self.temporal_out = TemporalConvBlock(hidden_channels, hidden_channels, temporal_kernel_size, dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, output_channels),
        )
        self._cached_adjacency: Optional[Tensor] = None
        self._cached_adjacency_signature: Optional[Tuple[int, int, Tuple[int, ...], Tuple[int, ...], str, Optional[int]]] = None

    @staticmethod
    def _edge_signature(edge_index: Tensor, edge_weight: Optional[Tensor], device: torch.device) -> Tuple[int, int, Tuple[int, ...], Tuple[int, ...], str, Optional[int]]:
        edge_weight_ptr = -1 if edge_weight is None else int(edge_weight.data_ptr())
        edge_weight_shape = tuple(edge_weight.shape) if edge_weight is not None else ()
        return (
            int(edge_index.data_ptr()),
            edge_weight_ptr,
            tuple(edge_index.shape),
            edge_weight_shape,
            device.type,
            device.index,
        )

    def _get_cached_adjacency(self, edge_index: Tensor, edge_weight: Optional[Tensor], num_nodes: int, device: torch.device) -> Tensor:
        signature = self._edge_signature(edge_index, edge_weight, device)
        if self._cached_adjacency is not None and self._cached_adjacency_signature == signature:
            return self._cached_adjacency
        adjacency = _to_dense_adjacency(edge_index, edge_weight, num_nodes, device)
        self._cached_adjacency = adjacency
        self._cached_adjacency_signature = signature
        return adjacency

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError("Expected input with shape [batch, time, nodes, features] or [time, nodes, features]")

        batch, time_steps, nodes, features = x.shape
        if nodes != self.num_nodes:
            raise ValueError(f"Input node count {nodes} does not match configured num_nodes={self.num_nodes}")

        adjacency = self._get_cached_adjacency(edge_index, edge_weight, nodes, x.device)

        x = self.temporal_in(x)
        # Mix spatial structure independently at each time step.
        spatial_outputs = []
        for t in range(x.shape[1]):
            spatial_outputs.append(self.spatial(x[:, t], adjacency))
        x = torch.stack(spatial_outputs, dim=1)
        x = self.temporal_out(x)
        x = x[:, -1]  # Final time slice for node-wise pressure prediction.
        out = self.readout(x).squeeze(-1)
        return out


def build_sequence_batch(
    feature_sequence: Sequence[np.ndarray],
    target_sequence: Sequence[np.ndarray],
    window_size: int,
) -> Tuple[Tensor, Tensor]:
    """Convert snapshot lists into sliding-window tensors for training."""

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if len(feature_sequence) != len(target_sequence):
        raise ValueError("feature and target sequences must have equal length")
    if len(feature_sequence) < window_size:
        raise ValueError("Not enough snapshots to build the requested window")

    windows = []
    labels = []
    for start in range(0, len(feature_sequence) - window_size + 1):
        x_window = np.stack(feature_sequence[start : start + window_size], axis=0)
        y_target = target_sequence[start + window_size - 1]
        windows.append(x_window)
        labels.append(y_target)

    x_tensor = torch.tensor(np.stack(windows, axis=0), dtype=torch.float32)
    y_tensor = torch.tensor(np.stack(labels, axis=0), dtype=torch.float32)
    return x_tensor, y_tensor


def iter_sequence_batches(
    feature_sequence: Sequence[np.ndarray],
    target_sequence: Sequence[np.ndarray],
    window_size: int,
    batch_size: int,
) -> Sequence[Tuple[Tensor, Tensor]]:
    """Yield sliding-window mini-batches without materializing the full scenario tensor."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if len(feature_sequence) != len(target_sequence):
        raise ValueError("feature and target sequences must have equal length")
    if len(feature_sequence) < window_size:
        raise ValueError("Not enough snapshots to build the requested window")

    total_windows = len(feature_sequence) - window_size + 1
    for start in range(0, total_windows, batch_size):
        end = min(start + batch_size, total_windows)
        windows = []
        labels = []
        for window_start in range(start, end):
            x_window = np.stack(feature_sequence[window_start : window_start + window_size], axis=0)
            y_target = target_sequence[window_start + window_size - 1]
            windows.append(x_window)
            labels.append(y_target)
        x_tensor = torch.tensor(np.stack(windows, axis=0), dtype=torch.float32)
        y_tensor = torch.tensor(np.stack(labels, axis=0), dtype=torch.float32)
        yield x_tensor, y_tensor


@torch.no_grad()
def anomaly_scores(predictions: Tensor, targets: Tensor) -> Tensor:
    """Return per-node absolute error scores."""

    return torch.abs(predictions - targets)
