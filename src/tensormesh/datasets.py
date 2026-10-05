"""ML dataset wrappers for batched mesh data."""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import asdict, dataclass
from itertools import accumulate
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
    when the requested index falls outside the current shard. Shards may
    hold different numbers of meshes.
    """

    def __init__(
        self,
        shard_paths: Sequence[Path],
        x_schema: FeatureSchema,
        y_schema: FeatureSchema,
        *,
        shard_sizes: Sequence[int] | None = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            shard_paths: paths to `MeshBatch` files, in dataset order.
            x_schema: features making up the input side of each datum.
            y_schema: features making up the target side of each datum.
            shard_sizes: number of meshes in each shard. If `None`, each shard
                is memory-mapped once to read its length. Each size is checked
                again when its shard is loaded.
        """
        if not shard_paths:
            msg = "shard_paths must not be empty"
            raise ValueError(msg)

        self._shard_paths = list(shard_paths)
        self._x_schema = x_schema
        self._y_schema = y_schema

        if shard_sizes is None:
            shard_sizes = [len(MeshBatch.load(p, mmap=True)) for p in self._shard_paths]
        self._shard_sizes = _validate_shard_sizes(shard_sizes)
        if len(self._shard_sizes) != len(self._shard_paths):
            msg = (
                f"Got {len(self._shard_sizes)} shard sizes "
                f"for {len(self._shard_paths)} shard paths"
            )
            raise ValueError(msg)
        self._offsets = _cumulative_offsets(self._shard_sizes)

        # Load the first shard, which also validates the schemas
        self._current_shard_idx = 0
        self._current_dataset = self._load_shard(0)

    @property
    def num_shards(self) -> int:
        return len(self._shard_paths)

    @property
    def shard_sizes(self) -> tuple[int, ...]:
        """Number of items in each shard."""
        return self._shard_sizes

    @property
    def shard_offsets(self) -> tuple[int, ...]:
        """(num_shards + 1,) cumulative shard sizes, starting at 0."""
        return self._offsets

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, idx: int) -> MeshDatum:
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            msg = f"index {idx} out of range for dataset of {len(self)} items"
            raise IndexError(msg)

        shard_idx = bisect_right(self._offsets, idx) - 1
        local_idx = idx - self._offsets[shard_idx]

        if shard_idx != self._current_shard_idx:
            self._current_dataset = self._load_shard(shard_idx)
            self._current_shard_idx = shard_idx

        return self._current_dataset[local_idx]

    def _load_shard(self, shard_idx: int) -> MeshDataset:
        """Load shard *shard_idx* and check it has the expected size."""
        dataset = MeshDataset.from_file(
            self._shard_paths[shard_idx], self._x_schema, self._y_schema
        )
        if len(dataset) != self._shard_sizes[shard_idx]:
            msg = (
                f"Shard {shard_idx} ({self._shard_paths[shard_idx]}) has "
                f"{len(dataset)} items, expected {self._shard_sizes[shard_idx]}"
            )
            raise ValueError(msg)
        return dataset


def _validate_shard_sizes(shard_sizes: Sequence[int]) -> tuple[int, ...]:
    """Check that *shard_sizes* is non-empty and strictly positive."""
    sizes = tuple(int(s) for s in shard_sizes)
    if not sizes:
        msg = "shard_sizes must not be empty"
        raise ValueError(msg)
    if any(s <= 0 for s in sizes):
        msg = f"shard_sizes must all be positive, got {list(sizes)}"
        raise ValueError(msg)
    return sizes


def _cumulative_offsets(sizes: Sequence[int]) -> tuple[int, ...]:
    """Return `(0, s0, s0 + s1, ...)`."""
    return (0, *accumulate(sizes))


class ShardShuffleSampler(Sampler[int]):
    """Sampler that shuffles within and across shards without crossing shard boundaries.

    Designed for `MeshShardedDataset`: each iteration visits every
    index exactly once, but indices from different shards are never
    interleaved, avoiding expensive shard swaps. Shards may have
    different sizes.

    Call `set_epoch` before each epoch for a different permutation.
    """

    def __init__(
        self,
        shard_sizes: Sequence[int],
        seed: int = 0,
        *,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
    ) -> None:
        self.shard_sizes = _validate_shard_sizes(shard_sizes)
        self.shard_offsets = _cumulative_offsets(self.shard_sizes)
        self.length = self.shard_offsets[-1]
        self.num_shards = len(self.shard_sizes)
        self.seed = seed
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.epoch = 0

    @classmethod
    def uniform(
        cls,
        length: int,
        shard_size: int,
        seed: int = 0,
        *,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
    ) -> ShardShuffleSampler:
        """Construct a sampler for *length* items in shards of *shard_size*.

        The last shard holds the remainder if *shard_size* does not divide
        *length*.
        """
        if length <= 0:
            msg = "length must be positive"
            raise ValueError(msg)
        if shard_size <= 0:
            msg = "shard_size must be positive"
            raise ValueError(msg)
        sizes = [min(shard_size, length - s) for s in range(0, length, shard_size)]
        return cls(
            sizes,
            seed=seed,
            shuffle_shards=shuffle_shards,
            shuffle_within_shard=shuffle_within_shard,
        )

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
            dataset.shard_sizes,
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

        for shard_idx in shard_order.tolist():
            shard_start = self.shard_offsets[shard_idx]
            shard_len = self.shard_sizes[shard_idx]

            if self.shuffle_within_shard:
                local_perm = torch.randperm(shard_len, generator=g)
            else:
                local_perm = torch.arange(shard_len)

            for local_idx in local_perm.tolist():
                yield shard_start + local_idx
