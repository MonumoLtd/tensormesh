from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch

from tensormesh import Mesh
from tensormesh.batch import MeshBatch

if TYPE_CHECKING:
    from pathlib import Path
from tensormesh.datasets import (
    FeatureSchema,
    MeshDataset,
    MeshShardedDataset,
    ShardShuffleSampler,
)


def _make_triangle(offset: float = 0.0) -> Mesh:
    return Mesh(
        xy=torch.tensor(
            [[0.0 + offset, 0.0], [1.0 + offset, 0.0], [0.5 + offset, 1.0]],
            dtype=torch.float64,
        ),
        cell_indices=torch.tensor([[0, 1, 2]], dtype=torch.long),
        vertex_features={
            "temp": torch.ones(3, dtype=torch.float64),
            "pressure": torch.zeros(3, dtype=torch.float64),
        },
        cell_features={"area": torch.ones(1, dtype=torch.float64)},
        global_features={"re": torch.tensor([1000.0], dtype=torch.float64)},
    )


# -- FeatureSchema ---------------------------------------------------------


class TestFeatureSchema:
    def test_json_round_trip(self, tmp_path: Path) -> None:
        schema = FeatureSchema(
            vertex_feature_names=("temp",),
            cell_feature_names=("area",),
            global_feature_names=("re",),
        )
        path = tmp_path / "schema.json"
        schema.to_json(path)
        loaded = FeatureSchema.from_json(path)
        assert loaded == schema

    def test_from_json_converts_lists_to_tuples(self, tmp_path: Path) -> None:
        path = tmp_path / "schema.json"
        path.write_text(
            json.dumps(
                {
                    "vertex_feature_names": ["a", "b"],
                    "cell_feature_names": [],
                    "global_feature_names": ["g"],
                }
            )
        )
        schema = FeatureSchema.from_json(path)
        assert isinstance(schema.vertex_feature_names, tuple)
        assert schema.vertex_feature_names == ("a", "b")


# -- MeshDatum -------------------------------------------------------------


class TestMeshDatum:
    def test_to_preserves_data(self) -> None:
        from tensormesh.datasets import MeshDatum

        m = _make_triangle()
        datum = MeshDatum(x=m, y=m)
        datum32 = datum.to(float_dtype=torch.float32)
        assert datum32.x.xy.dtype == torch.float32
        assert datum32.y.xy.dtype == torch.float32


# -- MeshDataset -----------------------------------------------------------


def _x_schema() -> FeatureSchema:
    return FeatureSchema(
        vertex_feature_names=("temp",),
        cell_feature_names=("area",),
        global_feature_names=("re",),
    )


def _y_schema() -> FeatureSchema:
    return FeatureSchema(
        vertex_feature_names=("pressure",),
        cell_feature_names=(),
        global_feature_names=(),
    )


class TestMeshDataset:
    def test_len(self) -> None:
        batch = MeshBatch.from_meshes([_make_triangle(), _make_triangle(1.0)])
        ds = MeshDataset(batch, _x_schema(), _y_schema())
        assert len(ds) == 2

    def test_getitem_returns_datum(self) -> None:
        batch = MeshBatch.from_meshes([_make_triangle()])
        ds = MeshDataset(batch, _x_schema(), _y_schema())
        datum = ds[0]
        # x should have 'temp' but not 'pressure'
        assert "temp" in datum.x.vertex_features
        assert "pressure" not in datum.x.vertex_features
        # y should have 'pressure' but not 'temp'
        assert "pressure" in datum.y.vertex_features
        assert "temp" not in datum.y.vertex_features

    def test_schema_mismatch_raises(self) -> None:
        batch = MeshBatch.from_meshes([_make_triangle()])
        bad_schema = FeatureSchema(vertex_feature_names=("nonexistent",))
        with pytest.raises(ValueError, match="nonexistent"):
            MeshDataset(batch, bad_schema, _y_schema())

    def test_from_file(self, tmp_path: Path) -> None:
        batch = MeshBatch.from_meshes([_make_triangle(), _make_triangle(1.0)])
        path = tmp_path / "mesh.pt"
        batch.save(path)
        ds = MeshDataset.from_file(path, _x_schema(), _y_schema())
        assert len(ds) == 2


# -- MeshShardedDataset ----------------------------------------------------


def _save_shard(root: Path, name: str, meshes: list[Mesh]) -> Path:
    """Write a shard .pt file and return its path."""
    path = root / f"{name}.pt"
    MeshBatch.from_meshes(meshes).save(path)
    return path


class TestMeshShardedDataset:
    def test_two_equal_shards(self, tmp_path: Path) -> None:
        p0 = _save_shard(tmp_path, "s0", [_make_triangle(0.0), _make_triangle(1.0)])
        p1 = _save_shard(tmp_path, "s1", [_make_triangle(2.0), _make_triangle(3.0)])

        ds = MeshShardedDataset([p0, p1], _x_schema(), _y_schema())
        assert len(ds) == 4
        assert ds.num_shards == 2

        # Access item in second shard triggers swap
        datum = ds[2]
        assert "temp" in datum.x.vertex_features

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            MeshShardedDataset([], _x_schema(), _y_schema())

    def test_shard_swap(self, tmp_path: Path) -> None:
        p0 = _save_shard(tmp_path, "s0", [_make_triangle(0.0)])
        p1 = _save_shard(tmp_path, "s1", [_make_triangle(5.0)])

        ds = MeshShardedDataset([p0, p1], _x_schema(), _y_schema())
        d0 = ds[0]
        d1 = ds[1]
        # The two shards have different xy offsets
        assert not torch.equal(d0.x.xy, d1.x.xy)


# -- ShardShuffleSampler ---------------------------------------------------


class TestShardShuffleSampler:
    def test_len(self) -> None:
        s = ShardShuffleSampler(length=100, shard_size=30)
        assert len(s) == 100

    def test_all_indices_yielded(self) -> None:
        s = ShardShuffleSampler(length=100, shard_size=30)
        assert set(s) == set(range(100))

    def test_no_cross_shard_indices(self) -> None:
        s = ShardShuffleSampler(length=100, shard_size=25)
        indices = list(s)
        # Each consecutive chunk of shard_size indices should belong to one shard
        for chunk_start in range(0, len(indices), 25):
            chunk = indices[chunk_start : chunk_start + 25]
            shards = {idx // 25 for idx in chunk}
            assert len(shards) == 1

    def test_deterministic_same_seed(self) -> None:
        a = list(ShardShuffleSampler(length=50, shard_size=10, seed=42))
        b = list(ShardShuffleSampler(length=50, shard_size=10, seed=42))
        assert a == b

    def test_different_epochs_differ(self) -> None:
        s = ShardShuffleSampler(length=50, shard_size=10)
        epoch0 = list(s)
        s.set_epoch(1)
        epoch1 = list(s)
        assert epoch0 != epoch1

    def test_no_shuffle(self) -> None:
        s = ShardShuffleSampler(
            length=20, shard_size=5, shuffle_shards=False, shuffle_within_shard=False
        )
        assert list(s) == list(range(20))

    def test_last_shard_smaller(self) -> None:
        # 7 items, shard_size 3 → shards of [3, 3, 1]
        s = ShardShuffleSampler(
            length=7, shard_size=3, shuffle_shards=False, shuffle_within_shard=False
        )
        assert list(s) == list(range(7))
        assert s.num_shards == 3

    def test_from_dataset(self, tmp_path: Path) -> None:
        p0 = _save_shard(tmp_path, "s0", [_make_triangle(0.0), _make_triangle(1.0)])
        p1 = _save_shard(tmp_path, "s1", [_make_triangle(2.0), _make_triangle(3.0)])
        ds = MeshShardedDataset([p0, p1], _x_schema(), _y_schema())

        s = ShardShuffleSampler.from_dataset(ds)
        assert s.length == 4
        assert s.shard_size == 2
        assert s.num_shards == 2

    def test_invalid_length_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            ShardShuffleSampler(length=0, shard_size=10)

    def test_invalid_shard_size_raises(self) -> None:
        with pytest.raises(ValueError, match="shard_size"):
            ShardShuffleSampler(length=10, shard_size=0)
