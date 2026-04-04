from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensormesh import Mesh
from tensormesh.batch import MeshBatch

if TYPE_CHECKING:
    from pathlib import Path


def _make_triangle(offset: float = 0.0, dtype: torch.dtype = torch.float64) -> Mesh:
    """Single-triangle mesh at a horizontal offset."""
    return Mesh(
        xy=torch.tensor(
            [[0.0 + offset, 0.0], [1.0 + offset, 0.0], [0.5 + offset, 1.0]], dtype=dtype
        ),
        cell_indices=torch.tensor([[0, 1, 2]], dtype=torch.long),
        vertex_features={"vf": torch.ones(3, dtype=dtype)},
        cell_features={"cf": torch.ones(1, dtype=dtype)},
        global_features={"gf": torch.tensor([42.0], dtype=dtype)},
    )


def _make_quad(dtype: torch.dtype = torch.float64) -> Mesh:
    """2-triangle, 4-vertex mesh."""
    return Mesh(
        xy=torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=dtype),
        cell_indices=torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long),
        vertex_features={"vf": torch.arange(4, dtype=dtype)},
        cell_features={"cf": torch.ones(2, dtype=dtype)},
        global_features={"gf": torch.tensor([7.0], dtype=dtype)},
    )


class TestFromMeshes:
    def test_single_mesh(self) -> None:
        m = _make_triangle()
        batch = MeshBatch.from_meshes([m])
        assert len(batch) == 1
        assert batch.num_meshes == 1

    def test_two_meshes(self) -> None:
        batch = MeshBatch.from_meshes([_make_triangle(), _make_quad()])
        assert len(batch) == 2
        assert batch.meshes.xy.shape == (7, 2)
        assert batch.meshes.cell_indices.shape == (3, 3)

    def test_empty_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="empty"):
            MeshBatch.from_meshes([])

    def test_pointer_values(self) -> None:
        m1 = _make_triangle()
        m2 = _make_quad()
        batch = MeshBatch.from_meshes([m1, m2])
        assert batch.vertex_ptr.tolist() == [0, 3, 7]
        assert batch.cell_ptr.tolist() == [0, 1, 3]


class TestGetitem:
    def test_round_trip_single(self) -> None:
        m = _make_triangle()
        batch = MeshBatch.from_meshes([m])
        recovered = batch[0]
        assert torch.equal(recovered.xy, m.xy)
        assert torch.equal(recovered.cell_indices, m.cell_indices)
        assert torch.equal(recovered.vertex_features["vf"], m.vertex_features["vf"])
        # Global features lose the leading (batch) dimension: (1,) → scalar
        assert recovered.global_features["gf"].item() == 42.0

    def test_round_trip_two_meshes(self) -> None:
        m1 = _make_triangle()
        m2 = _make_quad()
        batch = MeshBatch.from_meshes([m1, m2])

        r1 = batch[0]
        assert torch.equal(r1.xy, m1.xy)
        assert torch.equal(r1.cell_indices, m1.cell_indices)

        r2 = batch[1]
        assert torch.equal(r2.xy, m2.xy)
        assert torch.equal(r2.cell_indices, m2.cell_indices)
        assert torch.equal(r2.vertex_features["vf"], m2.vertex_features["vf"])

    def test_negative_index(self) -> None:
        batch = MeshBatch.from_meshes([_make_triangle(), _make_quad()])
        last = batch[-1]
        assert last.num_vertices == 4

    def test_out_of_range_raises(self) -> None:
        import pytest

        batch = MeshBatch.from_meshes([_make_triangle()])
        with pytest.raises(IndexError):
            batch[1]


class TestTo:
    def test_float_dtype(self) -> None:
        batch = MeshBatch.from_meshes([_make_triangle(dtype=torch.float64)])
        batch32 = batch.to(float_dtype=torch.float32)
        assert batch32.meshes.xy.dtype == torch.float32
        assert batch32.meshes.vertex_features["vf"].dtype == torch.float32
        # pointers stay int64
        assert batch32.vertex_ptr.dtype == torch.int64


class TestSaveLoad:
    def test_round_trip(self, tmp_path: Path) -> None:
        batch = MeshBatch.from_meshes([_make_triangle(), _make_quad()])
        path = tmp_path / "batch.pt"
        batch.save(path)
        loaded = MeshBatch.load(path)
        assert len(loaded) == 2
        assert torch.equal(loaded.meshes.xy, batch.meshes.xy)
        assert torch.equal(loaded.vertex_ptr, batch.vertex_ptr)
