from typing import Any

import plotly.graph_objects as go
import torch

from tensormesh import Mesh
from tensormesh.plots import (
    default_color_map,
    make_pretty,
    plot_boolean_cell_features,
    plot_cell_features,
    plot_mesh,
    plot_vector_field,
    plot_vertex_features,
    plot_wireframe,
)


def _simple_mesh() -> Mesh:
    """4-vertex, 2-cell mesh suitable for all plot-function tests."""
    xy = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64
    )
    cell_indices = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    return Mesh(
        xy=xy,
        cell_indices=cell_indices,
        vertex_features={
            "scalar_v": torch.tensor([0.0, 1.0, 2.0, 3.0]),
            "vec2_v": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]),
            "bool_v": torch.tensor([True, False, True, False]),
        },
        cell_features={
            "scalar_c": torch.tensor([1.0, 2.0]),
            "vec2_c": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "flag_a": torch.tensor([[True], [False]]),
            "flag_b": torch.tensor([[False], [True]]),
        },
    )


def _fig_dict(fig: go.Figure) -> dict[str, Any]:
    return fig.to_dict()


def _traces(fig: go.Figure) -> list[dict[str, Any]]:
    """Return figure traces as plain dicts for Pyright-safe access."""
    return _fig_dict(fig)["data"]


class TestDefaultColorMap:
    def test_returns_correct_keys(self) -> None:
        labels = ["a", "b", "c"]
        cmap = default_color_map(labels)
        assert set(cmap.keys()) == set(labels)

    def test_values_are_hex_strings(self) -> None:
        cmap = default_color_map(["x"])
        assert cmap["x"].startswith("#")
        assert len(cmap["x"]) == 7

    def test_cycles_past_20(self) -> None:
        labels = [str(i) for i in range(21)]
        cmap = default_color_map(labels)
        assert len(cmap) == 21


class TestMakePretty:
    def test_sets_equal_aspect(self) -> None:
        fig = go.Figure()
        make_pretty(fig)
        layout = _fig_dict(fig)["layout"]
        xaxis = layout["xaxis"]
        assert xaxis["scaleanchor"] == "y"

    def test_hides_grid(self) -> None:
        fig = go.Figure()
        make_pretty(fig)
        layout = _fig_dict(fig)["layout"]
        xaxis = layout["xaxis"]
        yaxis = layout["yaxis"]
        assert xaxis["showgrid"] is False
        assert yaxis["showgrid"] is False


class TestPlotMesh:
    def test_returns_figure(self) -> None:
        m = _simple_mesh()
        fig = plot_mesh(m)
        assert isinstance(fig, go.Figure)

    def test_has_trace(self) -> None:
        m = _simple_mesh()
        fig = plot_mesh(m)
        assert len(_traces(fig)) >= 1


class TestPlotCellFeatures:
    def test_string_key(self) -> None:
        m = _simple_mesh()
        fig = plot_cell_features(m, "scalar_c")
        assert isinstance(fig, go.Figure)

    def test_string_key_sets_colorbar(self) -> None:
        m = _simple_mesh()
        fig = plot_cell_features(m, "scalar_c")
        # Last trace is the dummy colorbar scatter; verify its colorbar title.
        last = _traces(fig)[-1]
        marker = last["marker"]
        colorbar = marker["colorbar"]
        title = colorbar["title"]
        assert title["text"] == "scalar_c"

    def test_tensor_input(self) -> None:
        m = _simple_mesh()
        fig = plot_cell_features(m, torch.tensor([0.5, 1.5]))
        assert isinstance(fig, go.Figure)

    def test_shape_2d_accepted(self) -> None:
        m = _simple_mesh()
        fig = plot_cell_features(m, torch.tensor([[0.5], [1.5]]))
        assert isinstance(fig, go.Figure)


class TestPlotBooleanCellFeatures:
    def test_one_trace_per_column(self) -> None:
        m = _simple_mesh()
        fig = plot_boolean_cell_features(m, ["flag_a", "flag_b"])
        named = [t for t in _traces(fig) if t.get("name")]
        assert len(named) == 2

    def test_trace_names_match_columns(self) -> None:
        m = _simple_mesh()
        fig = plot_boolean_cell_features(m, ["flag_a", "flag_b"])
        names = {t.get("name") for t in _traces(fig)}
        assert "flag_a" in names
        assert "flag_b" in names


class TestPlotVertexFeatures:
    def test_returns_figure(self) -> None:
        m = _simple_mesh()
        assert isinstance(plot_vertex_features(m, "scalar_v"), go.Figure)

    def test_string_key(self) -> None:
        m = _simple_mesh()
        fig = plot_vertex_features(m, "scalar_v")
        assert isinstance(fig, go.Figure)

    def test_tensor_input(self) -> None:
        m = _simple_mesh()
        fig = plot_vertex_features(m, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert isinstance(fig, go.Figure)

    def test_show_cells_adds_background(self) -> None:
        m = _simple_mesh()
        fig_with = plot_vertex_features(m, "scalar_v", show_cells=True)
        fig_without = plot_vertex_features(m, "scalar_v", show_cells=False)
        assert len(_traces(fig_with)) > len(_traces(fig_without))


class TestPlotVectorField:
    def test_vertex_mode(self) -> None:
        m = _simple_mesh()
        fig = plot_vector_field(m, "vertex", "vec2_v")
        assert isinstance(fig, go.Figure)

    def test_cell_mode(self) -> None:
        m = _simple_mesh()
        fig = plot_vector_field(m, "cell", "vec2_c")
        assert isinstance(fig, go.Figure)

    def test_with_masks_splits_traces(self) -> None:
        m = _simple_mesh()
        # Two mask columns → three named traces, one per group + hoovering info.
        fig = plot_vector_field(
            m, "vertex", "vec2_v", mask_columns=["bool_v", "bool_v"]
        )
        assert isinstance(fig, go.Figure)
        traces = _traces(fig)
        assert len(traces) == 3


class TestPlotWireframe:
    def test_returns_figure(self) -> None:
        m = _simple_mesh()
        fig = plot_wireframe(m, "scalar_v")
        assert isinstance(fig, go.Figure)

    def test_has_scatter3d(self) -> None:
        m = _simple_mesh()
        fig = plot_wireframe(m, "scalar_v")
        assert any(t.get("type") == "scatter3d" for t in _traces(fig))
