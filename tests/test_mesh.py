import torch

from tensormesh import Mesh


def _make_mesh(dtype: torch.dtype = torch.float64) -> Mesh:
    """2-triangle, 4-vertex mesh with one feature per category."""
    xy = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=dtype)
    cell_indices = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    return Mesh(
        xy=xy,
        cell_indices=cell_indices,
        vertex_features={"vf": torch.arange(4, dtype=dtype)},
        cell_features={"cf": torch.ones(2, dtype=dtype)},
        global_features={"gf": torch.tensor(42.0, dtype=dtype)},
    )


class TestMeshProperties:
    def test_num_vertices(self) -> None:
        m = _make_mesh()
        assert m.num_vertices == 4

    def test_num_cells(self) -> None:
        m = _make_mesh()
        assert m.num_cells == 2

    def test_device(self) -> None:
        m = _make_mesh()
        assert m.device == torch.device("cpu")


class TestMeshTo:
    def test_float_dtype_cast(self) -> None:
        m = _make_mesh(dtype=torch.float64)
        m32 = m.to(float_dtype=torch.float32)
        assert m32.xy.dtype == torch.float32
        assert m32.vertex_features["vf"].dtype == torch.float32
        assert m32.cell_features["cf"].dtype == torch.float32
        assert m32.global_features["gf"].dtype == torch.float32

    def test_long_preserved(self) -> None:
        m = _make_mesh()
        m32 = m.to(float_dtype=torch.float32)
        assert m32.cell_indices.dtype == torch.long

    def test_bool_preserved(self) -> None:
        m = _make_mesh()
        m_bool = m.with_features(cell_features={"mask": torch.tensor([True, False])})
        m32 = m_bool.to(float_dtype=torch.float32)
        assert m32.cell_features["mask"].dtype == torch.bool

    def test_device_none(self) -> None:
        m = _make_mesh()
        m2 = m.to(device=None, float_dtype=torch.float32)
        assert m2.device == torch.device("cpu")


class TestWithFeatures:
    def test_adds_vertex_feature(self) -> None:
        m = _make_mesh()
        new_vf = torch.zeros(4)
        m2 = m.with_features(vertex_features={"new": new_vf})
        assert "new" in m2.vertex_features
        assert m2.vertex_features["new"].shape == (4,)

    def test_adds_cell_feature(self) -> None:
        m = _make_mesh()
        m2 = m.with_features(cell_features={"new": torch.zeros(2)})
        assert "new" in m2.cell_features

    def test_adds_global_feature(self) -> None:
        m = _make_mesh()
        m2 = m.with_features(global_features={"extra": torch.tensor(1.0)})
        assert "extra" in m2.global_features

    def test_overwrites_existing_key(self) -> None:
        m = _make_mesh()
        replacement = torch.full((4,), 99.0)
        m2 = m.with_features(vertex_features={"vf": replacement})
        assert torch.all(m2.vertex_features["vf"] == 99.0)

    def test_original_unchanged(self) -> None:
        m = _make_mesh()
        m.with_features(vertex_features={"new": torch.zeros(4)})
        assert "new" not in m.vertex_features

    def test_geometry_preserved(self) -> None:
        m = _make_mesh()
        m2 = m.with_features(vertex_features={"x": torch.zeros(4)})
        assert m2.xy is m.xy
        assert m2.cell_indices is m.cell_indices


class TestDeleteFeatures:
    def test_removes_vertex_feature(self) -> None:
        m = _make_mesh()
        m2 = m.delete_features(vertex_features=["vf"])
        assert "vf" not in m2.vertex_features

    def test_removes_cell_feature(self) -> None:
        m = _make_mesh()
        m2 = m.delete_features(cell_features=["cf"])
        assert "cf" not in m2.cell_features

    def test_removes_global_feature(self) -> None:
        m = _make_mesh()
        m2 = m.delete_features(global_features=["gf"])
        assert "gf" not in m2.global_features

    def test_missing_key_silently_ignored(self) -> None:
        m = _make_mesh()
        m2 = m.delete_features(vertex_features=["does_not_exist"])
        assert m2.vertex_features.keys() == m.vertex_features.keys()

    def test_original_unchanged(self) -> None:
        m = _make_mesh()
        m.delete_features(vertex_features=["vf"])
        assert "vf" in m.vertex_features


class TestSelectFeatures:
    def test_keeps_requested_vertex_features(self) -> None:
        m = _make_mesh()
        m2 = m.with_features(
            vertex_features={"vf2": torch.arange(4, dtype=torch.float64)}
        )
        m3 = m2.select_features(vertex_features=["vf"])
        assert "vf" in m3.vertex_features
        assert "vf2" not in m3.vertex_features

    def test_none_keeps_all(self) -> None:
        m = _make_mesh()
        m2 = m.select_features(vertex_features=None)
        assert m2.vertex_features.keys() == m.vertex_features.keys()

    def test_empty_keeps_none(self) -> None:
        m = _make_mesh()
        m2 = m.select_features(vertex_features=[])
        assert len(m2.vertex_features) == 0

    def test_missing_name_raises(self) -> None:
        import pytest

        m = _make_mesh()
        with pytest.raises(ValueError, match="not_here"):
            m.select_features(vertex_features=["not_here"])

    def test_cell_and_global(self) -> None:
        m = _make_mesh()
        m2 = m.select_features(cell_features=["cf"], global_features=[])
        assert "cf" in m2.cell_features
        assert len(m2.global_features) == 0

    def test_geometry_preserved(self) -> None:
        m = _make_mesh()
        m2 = m.select_features(vertex_features=["vf"])
        assert m2.xy is m.xy
        assert m2.cell_indices is m.cell_indices


class TestClone:
    def test_values_equal(self) -> None:
        m = _make_mesh()
        c = m.clone()
        assert torch.equal(c.xy, m.xy)
        assert torch.equal(c.cell_indices, m.cell_indices)
        assert torch.equal(c.vertex_features["vf"], m.vertex_features["vf"])

    def test_tensors_independent(self) -> None:
        m = _make_mesh()
        c = m.clone()
        # Cloned xy must be a different storage
        assert c.xy.data_ptr() != m.xy.data_ptr()
