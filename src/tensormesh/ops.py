from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

type FeatureMap = Mapping[str, Tensor]


def cell_areas(xy_vertex: Tensor, cell_indices: Tensor) -> Tensor:
    """Calculate the areas of triangular cells in a mesh.

    Uses the 2D cross product.

    Args:
        xy_vertex: `(n_vertices, 2)` vertex coordinates.
        cell_indices: `(n_cells, 3)` indices mapping each cell to its
            vertices.

    Returns:
        Tensor of shape `(n_cells,)` with the area of each cell.
    """
    verts = xy_vertex[cell_indices]
    v0_v1 = verts[:, 1] - verts[:, 0]
    v0_v2 = verts[:, 2] - verts[:, 0]
    cross = v0_v1[..., 0] * v0_v2[..., 1] - v0_v1[..., 1] * v0_v2[..., 0]
    return 0.5 * torch.abs(cross)


def edges(cell_indices: Tensor) -> Tensor:
    """Unique undirected edges from triangle cell indices.

    Args:
        cell_indices: `(n_cells, 3)` integer tensor of vertex indices.

    Returns:
        `(n_edges, 2)` integer tensor of sorted, unique edge pairs.
    """
    v1 = cell_indices[:, 0]
    v2 = cell_indices[:, 1]
    v3 = cell_indices[:, 2]

    # Build all three edge pairs per triangle
    pairs = torch.stack(
        [
            torch.stack([v1, v2], dim=1),
            torch.stack([v1, v3], dim=1),
            torch.stack([v2, v3], dim=1),
        ],
        dim=0,
    ).reshape(-1, 2)

    # Sort endpoints so (a,b) == (b,a)
    pairs = torch.sort(pairs, dim=1).values

    # torch.unique sorted=True by default → rows already lexicographically ordered
    return torch.unique(pairs, dim=0)


def interpolate_at_cells(vertex_values: Tensor, cell_indices: Tensor) -> Tensor:
    """Averages a vertex field at cells.

    Args:
        vertex_values: `(n_vertices, ...)` tensor.
        cell_indices: `(n_cells, 3)` integer tensor of vertex indices.

    Returns:
        `(n_cells, ...)` tensor of interpolated values.
    """
    v0 = vertex_values[cell_indices[:, 0]]
    v1 = vertex_values[cell_indices[:, 1]]
    v2 = vertex_values[cell_indices[:, 2]]
    return (v0 + v1 + v2) / 3.0


def stack_features(features: FeatureMap, names: list[str] | None = None) -> Tensor:
    """Stack feature tensors along the last feature axis.

    Each selected tensor must have shape `(n, ...)`. Any tensor with a trailing
    dimension of size `1` is squeezed first. After that normalization, all
    selected tensors must have the same trailing shape `...`.

    Args:
        features: Mapping from feature name to tensor of shape `(n, ...)`.
        names: Optional ordered subset of feature names. If omitted, all
            features are used in mapping iteration order.

    Returns:
        A tensor of shape `(n, ..., num_features)`.
    """
    if names is None:
        names = list(features.keys())

    if not names:
        msg = "At least one feature must be selected."
        raise ValueError(msg)

    tensors = [features[name] for name in names]

    if any(t.ndim == 0 for t in tensors):
        msg = "Feature tensors must have shape (n, ...), not scalars."
        raise ValueError(msg)

    batch_size = tensors[0].shape[0]
    if any(t.shape[0] != batch_size for t in tensors):
        msg = "All feature tensors must have the same batch size."
        raise ValueError(msg)

    def _squeeze_trailing_unit_dim(t: Tensor) -> Tensor:
        if t.ndim >= 2 and t.shape[-1] == 1:
            return t.squeeze(-1)
        return t

    normalized = [_squeeze_trailing_unit_dim(t) for t in tensors]
    trailing_shapes = [t.shape[1:] for t in normalized]

    target_shape = trailing_shapes[0]
    if any(shape != target_shape for shape in trailing_shapes):
        shapes_str = ", ".join(
            f"{name}: {tuple(t.shape)}" for name, t in zip(names, tensors, strict=False)
        )
        msg = (
            "All feature tensors must have matching trailing shapes after "
            f"squeezing trailing dimension 1. Got: {shapes_str}"
        )
        raise ValueError(msg)

    return torch.stack(normalized, dim=-1)


def any_feature(features: FeatureMap, names: list[str] | None = None) -> Tensor:
    """Compute the elementwise boolean OR over selected feature tensors.

    Each selected tensor must have shape `(n, ...)`. Any tensor with a trailing
    dimension of size `1` is squeezed first. After that normalization, all
    selected tensors must have dtype `torch.bool` and matching trailing shape.

    Args:
        features: Mapping from feature name to boolean tensor of shape `(n, ...)`.
        names: Optional ordered subset of feature names. If omitted, all
            features are used in mapping iteration order.

    Returns:
        A boolean tensor of shape `(n, ...)` whose entries are the elementwise
        OR over the selected feature tensors.
    """
    if names is None:
        names = list(features.keys())

    if any(features[name].dtype != torch.bool for name in names):
        msg = "All feature tensors must be boolean."
        raise ValueError(msg)

    return stack_features(features, names).any(dim=-1)
