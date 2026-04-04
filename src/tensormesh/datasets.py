"""ML dataset wrappers for batched mesh data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset, Sampler

from tensormesh.batch import MeshBatch

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from tensormesh.mesh import Mesh


@dataclass(frozen=True)
class FeatureSchema:
    """Names of vertex, cell, and global features for one side of a datum."""

    vertex_feature_names: tuple[str, ...] = ()
    cell_feature_names: tuple[str, ...] = ()
    global_feature_names: tuple[str, ...] = ()

    def to_json(self, path: Path) -> None:
        with path.open("w") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def from_json(cls, path: Path) -> FeatureSchema:
        with path.open("r") as f:
            data = json.load(f)
        # JSON round-trips lists; convert to tuples
        return cls(
            vertex_feature_names=tuple(data["vertex_feature_names"]),
            cell_feature_names=tuple(data["cell_feature_names"]),
            global_feature_names=tuple(data["global_feature_names"]),
        )


@dataclass(frozen=True)
class MeshDatum:
    """An (x, y) pair of Mesh objects for supervised learning."""

    x: Mesh
    y: Mesh

    def to(
        self,
        device: torch.device | str | None = None,
        float_dtype: torch.dtype | None = None,
    ) -> MeshDatum:
        """Move both meshes to the specified device and/or dtype."""
        return MeshDatum(
            x=self.x.to(device=device, float_dtype=float_dtype),
            y=self.y.to(device=device, float_dtype=float_dtype),
        )


def _validate_schema(batch: MeshBatch, schema: FeatureSchema) -> None:
    """Check that every name in *schema* exists in *batch*."""
    missing_v = set(schema.vertex_feature_names) - set(batch.meshes.vertex_features)
    missing_c = set(schema.cell_feature_names) - set(batch.meshes.cell_features)
    missing_g = set(schema.global_feature_names) - set(batch.meshes.global_features)

    missing_parts: list[str] = []
    if missing_v:
        missing_parts.append(f"vertex: {sorted(missing_v)}")
    if missing_c:
        missing_parts.append(f"cell: {sorted(missing_c)}")
    if missing_g:
        missing_parts.append(f"global: {sorted(missing_g)}")

    if missing_parts:
        msg = "Schema references missing features — " + "; ".join(missing_parts)
        raise ValueError(msg)


class MeshDataset(Dataset[MeshDatum]):
    """PyTorch dataset backed by a `tensormesh.batch.MeshBatch`.

    Each item is an `(x, y)` `MeshDatum` pair sliced according to
    two feature schemas.
    """

    def __init__(
        self, batch: MeshBatch, x_schema: FeatureSchema, y_schema: FeatureSchema
    ) -> None:
        _validate_schema(batch, x_schema)
        _validate_schema(batch, y_schema)
        self._batch = batch
        self._x_schema = x_schema
        self._y_schema = y_schema

    @classmethod
    def from_file(
        cls,
        meshbatch_path: Path,
        x_schema: FeatureSchema,
        y_schema: FeatureSchema,
        *,
        mmap: bool = True,
    ) -> MeshDataset:
        """Load a dataset from a .pt file containing a MeshBatch."""
        batch = MeshBatch.load(meshbatch_path, mmap=mmap)
        return cls(batch=batch, x_schema=x_schema, y_schema=y_schema)

    def __len__(self) -> int:
        return len(self._batch)

    def __getitem__(self, idx: int) -> MeshDatum:
        mesh = self._batch[idx]
        return MeshDatum(
            x=mesh.select_features(
                vertex_features=self._x_schema.vertex_feature_names,
                cell_features=self._x_schema.cell_feature_names,
                global_features=self._x_schema.global_feature_names,
            ),
            y=mesh.select_features(
                vertex_features=self._y_schema.vertex_feature_names,
                cell_features=self._y_schema.cell_feature_names,
                global_features=self._y_schema.global_feature_names,
            ),
        )


class MeshShardedDataset(Dataset[MeshDatum]):
    """Sharded variant of `MeshDataset`.

    Keeps one shard memory-mapped at a time and swaps shards transparently
    when the requested index falls outside the current shard.
    """

    def __init__(
        self,
        shard_paths: Sequence[Path],
        x_schema: FeatureSchema,
        y_schema: FeatureSchema,
        *,
        assert_equal_shard_sizes: bool = True,
    ) -> None:
        if not shard_paths:
            msg = "shard_paths must not be empty"
            raise ValueError(msg)

        self._shard_paths = list(shard_paths)
        self._x_schema = x_schema
        self._y_schema = y_schema
        self._assert_equal_shard_sizes = assert_equal_shard_sizes

        # Load the first shard
        self._current_shard_idx = 0
        self._current_dataset = MeshDataset.from_file(
            self._shard_paths[0], x_schema, y_schema
        )
        self._shard_size = len(self._current_dataset)

        # Compute total length
        if assert_equal_shard_sizes:
            self._length = self._shard_size * len(self._shard_paths)
        else:
            self._length = sum(
                len(MeshBatch.load(p, mmap=True)) for p in self._shard_paths
            )

    @property
    def num_shards(self) -> int:
        return len(self._shard_paths)

    @property
    def shard_size(self) -> int:
        """Number of items in each shard (based on the first shard)."""
        return self._shard_size

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> MeshDatum:
        shard_idx = idx // self._shard_size
        local_idx = idx - shard_idx * self._shard_size

        if shard_idx != self._current_shard_idx:
            self._swap_shard(shard_idx)

        return self._current_dataset[local_idx]

    def _swap_shard(self, shard_idx: int) -> None:
        """Replace the current shard with shard *shard_idx*."""
        if not 0 <= shard_idx < len(self._shard_paths):
            n = len(self._shard_paths) - 1
            msg = f"Shard index {shard_idx} out of range (0..{n})"
            raise IndexError(msg)

        self._current_dataset = MeshDataset.from_file(
            self._shard_paths[shard_idx], self._x_schema, self._y_schema
        )
        self._current_shard_idx = shard_idx

        if (
            self._assert_equal_shard_sizes
            and len(self._current_dataset) != self._shard_size
        ):
            msg = (
                f"Shard {shard_idx} has {len(self._current_dataset)} items, "
                f"expected {self._shard_size}"
            )
            raise ValueError(msg)


class ShardShuffleSampler(Sampler[int]):
    """Sampler that shuffles within and across shards without crossing shard boundaries.

    Designed for `MeshShardedDataset`: each iteration visits every
    index exactly once, but indices from different shards are never
    interleaved, avoiding expensive shard swaps.

    Call `set_epoch` before each epoch for a different permutation.
    """

    def __init__(
        self,
        length: int,
        shard_size: int,
        seed: int = 0,
        *,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
    ) -> None:
        if length <= 0:
            msg = "length must be positive"
            raise ValueError(msg)
        if shard_size <= 0:
            msg = "shard_size must be positive"
            raise ValueError(msg)

        self.length = length
        self.shard_size = shard_size
        self.num_shards = -(length // -shard_size)  # ceiling division
        self.seed = seed
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.epoch = 0

    @classmethod
    def from_dataset(
        cls,
        dataset: MeshShardedDataset,
        seed: int = 0,
        *,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
    ) -> ShardShuffleSampler:
        """Construct a sampler from a `MeshShardedDataset`."""
        return cls(
            length=len(dataset),
            shard_size=dataset.shard_size,
            seed=seed,
            shuffle_shards=shuffle_shards,
            shuffle_within_shard=shuffle_within_shard,
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle_shards:
            shard_order = torch.randperm(self.num_shards, generator=g)
        else:
            shard_order = torch.arange(self.num_shards)

        for shard_idx in shard_order:
            shard_start = int(shard_idx) * self.shard_size
            shard_len = min(self.shard_size, self.length - shard_start)

            if self.shuffle_within_shard:
                local_perm = torch.randperm(shard_len, generator=g)
            else:
                local_perm = torch.arange(shard_len)

            for local_idx in local_perm:
                yield shard_start + int(local_idx)
