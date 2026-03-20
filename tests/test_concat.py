import unittest

import torch
from frozendict import frozendict

from tensormesh import Mesh
from tensormesh.mesh import concat


def _make_triangle(
    offset: float = 0.0,
    dtype: torch.dtype = torch.float64,
    vertex_cols: tuple[str, ...] = ("f1",),
    cell_cols: tuple[str, ...] = ("c1",),
    global_cols: tuple[str, ...] = (),
) -> Mesh:
    """Single-triangle mesh at a horizontal offset."""
    return Mesh(
        xy=torch.tensor(
            [[0.0 + offset, 0.0], [1.0 + offset, 0.0], [0.5 + offset, 1.0]], dtype=dtype
        ),
        cell_indices=torch.tensor([[0, 1, 2]], dtype=torch.long),
        vertex_features=frozendict(
            {k: torch.ones(3, dtype=dtype) for k in vertex_cols}
        ),
        cell_features=frozendict({k: torch.ones(1, dtype=dtype) for k in cell_cols}),
        global_features=frozendict(
            {k: torch.ones(1, dtype=dtype) for k in global_cols}
        ),
    )


class TestConcat(unittest.TestCase):
    def test_same_schema_vertex_count(self) -> None:
        m1 = _make_triangle(offset=0.0)
        m2 = _make_triangle(offset=2.0)
        merged = concat([m1, m2])
        self.assertEqual(merged.num_vertices, 6)

    def test_same_schema_cell_count(self) -> None:
        m1 = _make_triangle(offset=0.0)
        m2 = _make_triangle(offset=2.0)
        merged = concat([m1, m2])
        self.assertEqual(merged.num_cells, 2)

    def test_index_offset(self) -> None:
        """Second mesh cell indices must be shifted by the vertex count of the first."""
        m1 = _make_triangle(offset=0.0)
        m2 = _make_triangle(offset=2.0)
        merged = concat([m1, m2])
        # m1 has 3 vertices; m2's first cell index (originally 0) should become 3
        self.assertEqual(merged.cell_indices[1, 0].item(), 3)
        self.assertEqual(merged.cell_indices[1, 1].item(), 4)
        self.assertEqual(merged.cell_indices[1, 2].item(), 5)

    def test_preserves_dtype(self) -> None:
        m1 = _make_triangle(dtype=torch.float32)
        m2 = _make_triangle(offset=2.0, dtype=torch.float32)
        merged = concat([m1, m2])
        self.assertEqual(merged.xy.dtype, torch.float32)

    def test_vertex_key_mismatch_raises(self) -> None:
        m1 = _make_triangle(vertex_cols=("f1",))
        m2 = _make_triangle(offset=2.0, vertex_cols=("f2",))
        with self.assertRaisesRegex(ValueError, "same vertex feature keys"):
            concat([m1, m2])

    def test_cell_key_mismatch_raises(self) -> None:
        m1 = _make_triangle(cell_cols=("c1",))
        m2 = _make_triangle(offset=2.0, cell_cols=("c2",))
        with self.assertRaisesRegex(ValueError, "same cell feature keys"):
            concat([m1, m2])

    def test_global_key_mismatch_raises(self) -> None:
        m1 = _make_triangle(global_cols=("g1",))
        m2 = _make_triangle(offset=2.0, global_cols=("g2",))
        with self.assertRaisesRegex(ValueError, "same global feature keys"):
            concat([m1, m2])

    def test_empty_sequence_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            concat([])

    def test_single_mesh(self) -> None:
        """Concat of one mesh is equivalent to the original (indices unchanged)."""
        m = _make_triangle()
        merged = concat([m])
        self.assertEqual(merged.num_vertices, m.num_vertices)
        self.assertEqual(merged.num_cells, m.num_cells)
        self.assertTrue(torch.equal(merged.cell_indices, m.cell_indices))
        self.assertTrue(torch.equal(merged.xy, m.xy))

    def test_three_meshes(self) -> None:
        m1 = _make_triangle(offset=0.0)
        m2 = _make_triangle(offset=2.0)
        m3 = _make_triangle(offset=4.0)
        merged = concat([m1, m2, m3])
        self.assertEqual(merged.num_vertices, 9)
        self.assertEqual(merged.num_cells, 3)
        # Third mesh cell indices should start at 6
        self.assertEqual(merged.cell_indices[2, 0].item(), 6)

    def test_with_global_features(self) -> None:
        m1 = _make_triangle(global_cols=("g1",))
        m2 = _make_triangle(offset=2.0, global_cols=("g1",))
        merged = concat([m1, m2])
        self.assertIn("g1", merged.global_features)
        # Both global tensors of shape (1,) get concatenated → (2,)
        self.assertEqual(merged.global_features["g1"].shape[0], 2)

    def test_feature_values_concatenated(self) -> None:
        m1 = _make_triangle()
        m2 = _make_triangle(offset=2.0)
        merged = concat([m1, m2])
        self.assertEqual(merged.vertex_features["f1"].shape[0], 6)
        self.assertEqual(merged.cell_features["c1"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
