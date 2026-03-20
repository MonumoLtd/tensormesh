import pytest
import torch

from tensormesh.ops import (
    any_feature,
    cell_areas,
    edges,
    interpolate_at_cells,
    stack_features,
)


class TestCellAreas:
    def test_unit_right_triangle(self) -> None:
        xy = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        ci = torch.tensor([[0, 1, 2]], dtype=torch.long)
        areas = cell_areas(xy, ci)
        assert areas.shape == (1,)
        assert torch.isclose(areas[0], torch.tensor(0.5, dtype=torch.float64))

    def test_two_triangles(self) -> None:
        xy = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64
        )
        ci = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
        areas = cell_areas(xy, ci)
        assert areas.shape == (2,)
        assert torch.allclose(areas, torch.tensor([0.5, 0.5], dtype=torch.float64))

    def test_clockwise_winding_same_area(self) -> None:
        # Reversing vertex order flips the cross-product sign; abs() absorbs it.
        xy = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        ccw = torch.tensor([[0, 1, 2]], dtype=torch.long)
        cw = torch.tensor([[0, 2, 1]], dtype=torch.long)
        assert torch.isclose(cell_areas(xy, ccw)[0], cell_areas(xy, cw)[0])

    def test_known_non_unit_area(self) -> None:
        # Right triangle with base=4, height=6 -> area=12.
        xy = torch.tensor([[0.0, 0.0], [4.0, 0.0], [0.0, 6.0]], dtype=torch.float64)
        ci = torch.tensor([[0, 1, 2]], dtype=torch.long)
        assert torch.isclose(
            cell_areas(xy, ci)[0], torch.tensor(12.0, dtype=torch.float64)
        )


class TestEdges:
    def test_single_triangle(self) -> None:
        ci = torch.tensor([[0, 1, 2]], dtype=torch.long)
        e = edges(ci)
        assert e.shape == (3, 2)
        # Edges should be sorted
        assert (e[:, 0] <= e[:, 1]).all()

    def test_two_triangles_shared_edge(self) -> None:
        ci = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
        e = edges(ci)
        # 2 triangles sharing edge (1,2) -> 5 unique edges
        assert e.shape == (5, 2)

    def test_all_edges_sorted(self) -> None:
        ci = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
        e = edges(ci)
        assert (e[:, 0] <= e[:, 1]).all()

    def test_exact_edges_single_triangle(self) -> None:
        ci = torch.tensor([[0, 1, 2]], dtype=torch.long)
        e = edges(ci)
        expected = {(0, 1), (0, 2), (1, 2)}
        actual = {(int(row[0]), int(row[1])) for row in e}
        assert actual == expected

    def test_unique_no_duplicates(self) -> None:
        # Shared edge (1,2) appears in both triangles but must appear only once.
        ci = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
        e = edges(ci)
        unique_e = torch.unique(e, dim=0)
        assert e.shape == unique_e.shape


class TestInterpolateVertexToCells:
    def test_1d_values(self) -> None:
        vals = torch.tensor([0.0, 3.0, 6.0], dtype=torch.float64)
        ci = torch.tensor([[0, 1, 2]], dtype=torch.long)
        result = interpolate_at_cells(vals, ci)
        assert result.shape == (1,)
        assert torch.isclose(result[0], torch.tensor(3.0, dtype=torch.float64))

    def test_2d_values(self) -> None:
        vals = torch.tensor([[0.0, 1.0], [3.0, 4.0], [6.0, 7.0]], dtype=torch.float64)
        ci = torch.tensor([[0, 1, 2]], dtype=torch.long)
        result = interpolate_at_cells(vals, ci)
        assert result.shape == (1, 2)
        assert torch.allclose(result[0], torch.tensor([3.0, 4.0], dtype=torch.float64))

    def test_multi_cell(self) -> None:
        # Vertex values: 0, 3, 6, 9 — two cells use distinct, non-overlapping vertices.
        vals = torch.tensor([0.0, 3.0, 6.0, 9.0], dtype=torch.float64)
        ci = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        result = interpolate_at_cells(vals, ci)
        assert result.shape == (2,)
        assert torch.isclose(result[0], torch.tensor(3.0, dtype=torch.float64))
        assert torch.isclose(result[1], torch.tensor(6.0, dtype=torch.float64))


class TestStackFeatures:
    def test_scalar_like_features_stack_on_last_axis(self) -> None:
        features = {
            "a": torch.tensor([1.0, 2.0], dtype=torch.float64),
            "b": torch.tensor([[3.0], [4.0]], dtype=torch.float64),
        }
        result = stack_features(features)
        expected = torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.float64)
        assert result.shape == (2, 2)
        assert torch.allclose(result, expected)

    def test_vector_features_stack_on_last_axis(self) -> None:
        features = {
            "x": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
            "y": torch.tensor(
                [[[10.0], [20.0]], [[30.0], [40.0]]], dtype=torch.float64
            ),
        }
        result = stack_features(features)
        expected = torch.tensor(
            [[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]],
            dtype=torch.float64,
        )
        assert result.shape == (2, 2, 2)
        assert torch.allclose(result, expected)

    def test_respects_requested_name_order(self) -> None:
        features = {
            "first": torch.tensor([1.0, 2.0]),
            "second": torch.tensor([3.0, 4.0]),
        }
        result = stack_features(features, ["second", "first"])
        expected = torch.tensor([[3.0, 1.0], [4.0, 2.0]])
        assert torch.equal(result, expected)

    def test_raises_for_empty_names(self) -> None:
        with pytest.raises(ValueError, match="At least one feature"):
            stack_features({}, [])

    def test_raises_for_scalar_tensor(self) -> None:
        features = {"a": torch.tensor(1.0), "b": torch.tensor([1.0])}
        with pytest.raises(ValueError, match="not scalars"):
            stack_features(features)

    def test_raises_for_mismatched_batch_size(self) -> None:
        features = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0, 5.0])}
        with pytest.raises(ValueError, match="same batch size"):
            stack_features(features)

    def test_raises_for_incompatible_shapes_after_squeeze(self) -> None:
        features = {"a": torch.randn(2, 3), "b": torch.randn(2, 4, 1)}
        with pytest.raises(ValueError, match="matching trailing shapes"):
            stack_features(features)


class TestAnyFeature:
    def test_scalar_like_boolean_features(self) -> None:
        features = {
            "a": torch.tensor([True, False, False]),
            "b": torch.tensor([[False], [True], [False]]),
        }
        result = any_feature(features)
        expected = torch.tensor([True, True, False])
        assert result.dtype == torch.bool
        assert result.shape == (3,)
        assert torch.equal(result, expected)

    def test_higher_rank_boolean_features(self) -> None:
        features = {
            "a": torch.tensor(
                [[[True], [False]], [[False], [False]]], dtype=torch.bool
            ),
            "b": torch.tensor([[False, True], [True, False]], dtype=torch.bool),
        }
        result = any_feature(features)
        expected = torch.tensor([[True, True], [True, False]], dtype=torch.bool)
        assert result.shape == (2, 2)
        assert torch.equal(result, expected)

    def test_respects_subset_selection(self) -> None:
        features = {
            "a": torch.tensor([False, False, False]),
            "b": torch.tensor([False, True, False]),
            "c": torch.tensor([True, True, True]),
        }
        result = any_feature(features, ["a", "b"])
        expected = torch.tensor([False, True, False])
        assert torch.equal(result, expected)

    def test_raises_for_non_boolean_tensor(self) -> None:
        features = {"a": torch.tensor([True, False]), "b": torch.tensor([1.0, 0.0])}
        with pytest.raises(ValueError, match="must be boolean"):
            any_feature(features)

    def test_propagates_stack_features_validation(self) -> None:
        features = {
            "a": torch.tensor([True, False]),
            "b": torch.tensor([[True, False], [False, True]], dtype=torch.bool),
        }
        with pytest.raises(ValueError, match="matching trailing shapes"):
            any_feature(features)
