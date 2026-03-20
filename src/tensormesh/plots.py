from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import plotly.colors as pcolors
import plotly.graph_objects as go
from plotly import figure_factory as ff

from tensormesh import ops as _ops

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray
    from torch import Tensor

    from tensormesh import Mesh


def _viridis_hex(values: NDArray[np.floating], vmin: float, vmax: float) -> list[str]:
    """Map *values* to hex colours via the Viridis colorscale.

    Values outside `[vmin, vmax]` are clamped; non-finite values map to
    `#cccccc`.
    """
    span = vmax - vmin
    ts = np.where(np.isfinite(values), np.clip((values - vmin) / span, 0.0, 1.0), 0.0)
    rgb_list = pcolors.sample_colorscale("Viridis", ts.tolist(), colortype="tuple")
    return [
        f"#{round(float(r) * 255):02x}"
        f"{round(float(g) * 255):02x}"
        f"{round(float(b) * 255):02x}"
        if np.isfinite(v)
        else "#cccccc"
        for v, (r, g, b, *_) in zip(values, rgb_list, strict=False)
    ]


def _nan_gap_trace(
    xy: NDArray[np.floating],
    cells_closed: NDArray[np.integer],
    *,
    fillcolor: str,
    edge_color: str,
    edge_width: float,
    name: str = "",
    showlegend: bool = False,
    legendgroup: str | None = None,
) -> go.Scatter:
    """Build a single go.Scatter with NaN-separated closed polygons."""
    coords = xy[cells_closed]  # (n_cells, 4, 2)
    nan_block = np.full((len(coords), 1, 2), np.nan)
    segs = np.concatenate([coords, nan_block], axis=1).reshape(-1, 2)
    return go.Scatter(
        x=segs[:, 0],
        y=segs[:, 1],
        fill="toself",
        mode="lines",
        line={"color": edge_color, "width": edge_width},
        fillcolor=fillcolor,
        hoverinfo="skip",
        name=name,
        showlegend=showlegend,
        legendgroup=legendgroup,
    )


def make_pretty(fig: go.Figure) -> None:
    """Make a Plotly figure nicer with equal-aspect axes and no grid."""
    fig.update_layout(
        xaxis={
            "scaleanchor": "y",
            "showgrid": False,
            "zeroline": False,
            "visible": True,
        },
        yaxis={"showgrid": False, "zeroline": False, "visible": True},
        showlegend=True,
        width=800,
        height=800,
    )
    fig.update_traces(marker_size=4)


def default_color_map(labels: Sequence[str]) -> dict[str, str]:
    """Return a dictionary with colours associated to each label.

    Args:
        labels: List of unique labels to associate colours to.
    """
    # Matplotlib tab20 palette
    tab20 = [
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#ffbb78",
        "#2ca02c",
        "#98df8a",
        "#d62728",
        "#ff9896",
        "#9467bd",
        "#c5b0d5",
        "#8c564b",
        "#c49c94",
        "#e377c2",
        "#f7b6d2",
        "#7f7f7f",
        "#c7c7c7",
        "#bcbd22",
        "#dbdb8d",
        "#17becf",
        "#9edae5",
    ]
    return {label: tab20[i % len(tab20)] for i, label in enumerate(labels)}


# ---------------------------------------------------------------------------
# Cell / vertex / vector-field / 3D plotting
# ---------------------------------------------------------------------------


def plot_mesh(
    mesh: Mesh,
    *,
    edge_color: str = "black",
    fill_color: str = "lightgrey",
    edge_width: float = 0.5,
) -> go.Figure:
    """Plot the mesh geometry as filled triangles.

    Args:
        mesh: Mesh object containing vertex coordinates and triangle connectivity.
        edge_color: Colour of triangle edges.
        fill_color: Colour of triangle interiors.
        edge_width: Width of triangle edges.

    Returns:
        A Plotly Figure object representing the mesh.
    """
    xy = mesh.xy.detach().cpu().numpy()
    cell_indices = mesh.cell_indices.detach().cpu().numpy()
    cells_closed = np.concatenate([cell_indices, cell_indices[:, [0]]], axis=1)
    fig = go.Figure()
    fig.add_trace(
        _nan_gap_trace(
            xy,
            cells_closed,
            fillcolor=fill_color,
            edge_color=edge_color,
            edge_width=edge_width,
            showlegend=False,
        )
    )
    make_pretty(fig)
    return fig


def plot_cell_features(
    mesh: Mesh,
    features: str | Tensor,
    *,
    edge_color: str = "black",
    edge_width: float = 0.5,
    n_buckets: int = 25,
) -> go.Figure:
    """Plot mesh with cells colored according to a feature.

    Args:
        mesh: Mesh object containing vertex coordinates and triangle connectivity.
        features: either a key in the cell_features dictionary or a tensor of shape
            `(n_cells, )` or `(n_cells, 1)`.
        edge_color: Colour of cell edges.
        edge_width: Width of cell edges.
        n_buckets: Number of colour buckets for numeric colouring (default 25).

    Returns:
        A Plotly Figure object representing the mesh with colored cells.
    """
    xy = mesh.xy.detach().cpu().numpy()
    cell_indices = mesh.cell_indices.detach().cpu().numpy()
    cells_closed = np.concatenate([cell_indices, cell_indices[:, [0]]], axis=1)

    feature_name = ""  # Default colourbar title; overridden if features is a string.
    if isinstance(features, str):
        feature_name = features
        features = mesh.cell_features[features]

    if features.shape[0] != cell_indices.shape[0]:
        msg = "Expected features with shape (num_cells,) or (num_cells, 1), "
        msg += f"got {features.shape}."
        raise ValueError(msg)

    if features.ndim != 1 and features.shape[1] != 1:
        msg = "Expected features with shape (num_cells,) or (num_cells, 1), "
        msg += f"got {features.shape}."
        raise ValueError(msg)

    values = features.detach().cpu().numpy().squeeze()  # (n_cells,)

    traces: list[go.Scatter] = []

    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.min(finite_vals))
        vmax = float(np.max(finite_vals))
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5

    span = vmax - vmin
    finite_mask = np.isfinite(values)
    safe_vals = np.where(finite_mask, values, vmin)
    ts = np.clip((safe_vals - vmin) / span, 0.0, 1.0)
    bucket_idx = np.minimum(np.floor(ts * n_buckets).astype(int), n_buckets - 1)

    bucket_edges = np.linspace(vmin, vmax, n_buckets + 1)
    bucket_mids = (bucket_edges[:-1] + bucket_edges[1:]) / 2.0
    bucket_colors = _viridis_hex(bucket_mids, vmin, vmax)

    for b in range(n_buckets):
        mask = finite_mask & (bucket_idx == b)
        if not np.any(mask):
            continue
        traces.append(
            _nan_gap_trace(
                xy,
                cells_closed[mask],
                fillcolor=bucket_colors[b],
                edge_color=edge_color,
                edge_width=edge_width,
                showlegend=False,
            )
        )

    if np.any(~finite_mask):
        traces.append(
            _nan_gap_trace(
                xy,
                cells_closed[~finite_mask],
                fillcolor="#cccccc",
                edge_color=edge_color,
                edge_width=edge_width,
                showlegend=False,
            )
        )

    traces.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "colorscale": "Viridis",
                "cmin": vmin,
                "cmax": vmax,
                "colorbar": {"title": feature_name},
                "color": [vmin, vmax],
            },
            showlegend=False,
        )
    )

    fig = go.Figure(traces)
    make_pretty(fig)
    return fig


def plot_boolean_cell_features(
    mesh: Mesh,
    columns: list[str],
    *,
    colormap: Mapping[str, str] | None = None,
    edge_color: str = "black",
    edge_width: float = 0.5,
) -> go.Figure:
    """Plot mesh cells coloured by one-hot boolean flags.

    Args:
        mesh: Mesh object containing the cell features.
        columns: names of the boolean cell_features to plot. Each feature is expected to
            have shape `(n_cells,)` or `(n_cells, 1)` and contain only boolean values.
        colormap: column_name -> colour mapping.  Missing names fall back to
            `default_material_color_map` then `default_color_map`.
        edge_color: Colour of cell edges.
        edge_width: Width of cell edges.

    Returns:
        A Plotly Figure object representing the mesh with colored cells.
    """
    xy = mesh.xy.detach().cpu().numpy()
    cell_indices = mesh.cell_indices.detach().cpu().numpy()
    features = [mesh.cell_features[c].detach().cpu().numpy() for c in columns]
    flags = np.hstack([f.reshape(-1, 1) for f in features])  # (n_cells, n_columns)
    unique_display = list(dict.fromkeys(columns))

    # Build colormap keyed by display names
    base_cmap = default_color_map(labels=columns)
    missing = [d for d in unique_display if d not in base_cmap]
    merged: dict[str, str] = {**base_cmap, **default_color_map(missing)}
    if colormap is not None:
        merged.update(colormap)

    cells_closed = np.concatenate([cell_indices, cell_indices[:, [0]]], axis=1)
    display_arr = np.array(columns)
    traces: list[go.Scatter] = []
    for disp in unique_display:
        col_mask = display_arr == disp  # (n_cols,) — which flag columns carry this name
        cell_mask = np.any(flags[:, col_mask], axis=1)  # (n_cells,) — which cells match
        traces.append(
            _nan_gap_trace(
                xy,
                cells_closed[cell_mask],
                fillcolor=merged.get(disp, "#cccccc"),
                edge_color=edge_color,
                edge_width=edge_width,
                name=disp,
                showlegend=True,
                legendgroup=disp,
            )
        )

    fig = go.Figure(traces)
    make_pretty(fig)
    return fig


def plot_vertex_features(
    mesh: Mesh, features: str | Tensor, *, show_cells: bool = True
) -> go.Figure:
    """Plot mesh vertices as a scatter plot.

    Args:
        mesh: Mesh object containing vertex coordinates and triangle connectivity.
        features: either a key in the vertex_features dictionary or a tensor of shape
            `(n_vertices, )` or `(n_vertices, 1)`.
        show_cells: If True, show the mesh cells in the background.

    Returns:
        A Plotly Figure object representing the mesh with vertex features.
    """
    xy = mesh.xy.detach().cpu().numpy()
    feature_name = ""  # Default colourbar title; overridden if features is a string.
    if isinstance(features, str):
        feature_name = features
        vv = mesh.vertex_features[features].detach().cpu().numpy()
    else:
        vv = features.detach().cpu().numpy()
    vv = vv.astype(float).squeeze()  # (n_vertices,)
    finite_vv = vv[np.isfinite(vv)]
    if finite_vv.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.min(finite_vv))
        vmax = float(np.max(finite_vv))
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5

    scatter = go.Scattergl(
        x=xy[:, 0],
        y=xy[:, 1],
        mode="markers",
        marker={
            "size": 5,
            "color": vv,
            "colorscale": "Viridis",
            "colorbar": {"title": ""},
            "cmin": vmin,
            "cmax": vmax,
        },
        customdata=np.arange(xy.shape[0]),
        hovertemplate=(
            "x: %{x:.2f}<br>y: %{y:.2f}<br>vertex index: %{customdata:.0f}"
            "<extra></extra>"
        ),
        showlegend=False,
    )

    if show_cells:
        fig = plot_mesh(mesh)
        fig.add_trace(scatter)
        fig.update_layout(title=feature_name)
        return fig

    fig = go.Figure([scatter], layout={"title": feature_name})
    make_pretty(fig)
    return fig


def _trace_single_vector_field(
    positions: NDArray[np.floating],
    vectors: NDArray[np.floating],
    *,
    scale: float = 1.0,
    label: str | None = None,
    color: str | None = None,
) -> go.Figure:
    """Plot a 2D vector field as quiver arrows.

    Args:
        positions: `(n, 2)` array with the `(x, y)` node coordinates.
        vectors: `(n, 2)` array with the field components.
        scale: Scaling factor for the arrows.
        label: label to be displayed in the legend.
        color: Colour of the arrows.
    """
    if positions.shape[0] != vectors.shape[0]:
        msg = (
            f"positions and vectors must have the same number of rows,"
            f" got {positions.shape[0]} and {vectors.shape[0]}"
        )
        raise ValueError(msg)
    if positions.shape[1] != 2:
        msg = f"positions must have 2 columns, got {positions.shape[1]}"
        raise ValueError(msg)
    if vectors.shape[1] != 2:
        msg = f"vectors must have 2 columns, got {vectors.shape[1]}"
        raise ValueError(msg)

    _color = color if color is not None else default_color_map(["default"])["default"]
    return ff.create_quiver(
        positions[:, 0],
        positions[:, 1],
        vectors[:, 0],
        vectors[:, 1],
        scale=scale,
        line_color=_color,
        name=label or "",
    )


def plot_vector_field(
    mesh: Mesh,
    face_type: Literal["vertex", "cell"],
    features: str | Tensor,
    *,
    mask_columns: Sequence[str] | None = None,
    scale: float = 1.0,
    colormap: Mapping[str, str] | None = None,
    show_cells: bool = False,
) -> go.Figure:
    """Plot a 2D vector field over the mesh as quiver arrows.

    Exactly one of *vertex_columns* or *cell_columns* must be supplied.
    When *masks* is `None` all arrows share a single default colour.
    When *masks* is provided the arrows are split into one group per mask,
    each receiving its own colour and legend entry.

    Args:
        mesh: Mesh object containing vertex coordinates and triangle connectivity.
        face_type: whether the feature field is defined at vertices or cells.
        features: either a key in the vertex_features or cell_features dictionary
            (depending on *face_type*) or a tensor of shape `(n_vertices, 2)` or
            `(n_cells, 2)`.
        mask_columns: Optional list of boolean feature column names to use as masks for
            colouring. Each column is expected to have shape `(n_vertices, 1)` or
            `(n_cells, 1)` and contain boolean values. If provided, arrows are coloured
            according to which mask(s) they belong to; if None, all arrows share the
            same colour.
        scale: Scaling factor for the arrows.
        colormap: Optional mapping from mask column name to colour. Missing names fall
            back to `default_color_map`.
        show_cells: If True, show the mesh cells in the background.

    Returns:
        A Plotly figure.
    """
    if face_type == "vertex":
        positions = mesh.xy.detach().cpu().numpy()
        if isinstance(features, str):
            vectors = mesh.vertex_features[features].detach().cpu().numpy()
        else:
            vectors = features.detach().cpu().numpy()

    else:
        positions = _ops.interpolate_at_cells(mesh.xy, mesh.cell_indices)
        positions = positions.detach().cpu().numpy()
        if isinstance(features, str):
            vectors = mesh.cell_features[features].detach().cpu().numpy()
        else:
            vectors = features.detach().cpu().numpy()

    if mask_columns is None:
        color = colormap.get("default") if colormap else None
        fig = _trace_single_vector_field(positions, vectors, scale=scale, color=color)
        make_pretty(fig)

    else:
        cmap = colormap if colormap is not None else default_color_map(mask_columns)
        fig = go.Figure()
        for col in mask_columns:
            if face_type == "vertex":
                mask = mesh.vertex_features[col].detach().cpu().numpy().squeeze()
            else:
                mask = mesh.cell_features[col].detach().cpu().numpy().squeeze()
            sub = _trace_single_vector_field(
                positions[mask],
                vectors[mask],
                scale=scale,
                label=col,
                color=cmap.get(col),
            )
            fig.add_traces(sub.data)
        make_pretty(fig)
        fig.update_layout(showlegend=True)

    if show_cells:
        fig_outline = plot_mesh(mesh)
        fig_outline.add_traces(fig.data)
        return fig_outline
    return fig


def plot_wireframe(mesh: Mesh, features: str | Tensor) -> go.Figure:
    """Plot a 3D wireframe of a triangle mesh.

    Args:
        mesh: Mesh object containing vertex coordinates and triangle connectivity.
        features: either a key in the vertex_features dictionary or a tensor of shape
            `(n_vertices, )` or `(n_vertices, 1)` to be used as the z-coordinate of the
            vertices.
    """
    # Build unique edges via sorted pairs, deduplicating with numpy
    xy = mesh.xy.detach().cpu().numpy()
    cell_indices = mesh.cell_indices.detach().cpu().numpy()
    if isinstance(features, str):
        z = mesh.vertex_features[features].detach().cpu().numpy().squeeze()
    else:
        z = features.detach().cpu().numpy().squeeze()
    if z.shape != (xy.shape[0],):
        msg = f"Expected features shape (num_vertices,) got {z.shape}."
        raise ValueError(msg)

    raw = np.concatenate(
        [cell_indices[:, [0, 1]], cell_indices[:, [1, 2]], cell_indices[:, [2, 0]]],
        axis=0,
    )
    edges = np.unique(np.sort(raw, axis=1), axis=0)

    x1, y1, z1 = xy[edges[:, 0], 0], xy[edges[:, 0], 1], z[edges[:, 0]]
    x2, y2, z2 = xy[edges[:, 1], 0], xy[edges[:, 1], 1], z[edges[:, 1]]
    nan = np.full_like(x1, np.nan, dtype=float)

    xs = np.column_stack([x1, x2, nan]).ravel()
    ys = np.column_stack([y1, y2, nan]).ravel()
    zs = np.column_stack([z1, z2, nan]).ravel()

    fig = go.Figure([go.Scatter3d(x=xs, y=ys, z=zs, mode="lines")])
    fig.update_layout(width=800, height=800, margin={"l": 0, "r": 0, "t": 0, "b": 0})
    return fig
