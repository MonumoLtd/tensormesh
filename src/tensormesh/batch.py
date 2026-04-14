"""Batched mesh storage: multiple meshes packed into contiguous tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tensormesh.mesh import Mesh, concat

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from torch import Tensor


@dataclass(frozen=True, eq=False, init=False)
class MeshBatch:
    """A batch of meshes stored as concatenated tensors with CSR pointer arrays.

    Analogous to PyTorch Geometric's `Batch`: individual meshes share
    contiguous `xy`, `cell_indices`, and feature tensors, with
    `vertex_ptr` / `cell_ptr` recording where each mesh begins and ends.
    """

    meshes: Mesh
    """The concatenated mesh holding all vertices, cells, and features."""

    vertex_ptr: Tensor
    """(num_meshes + 1,) int64 tensor of cumulative vertex counts."""

    cell_ptr: Tensor
    """(num_meshes + 1,) int64 tensor of cumulative cell counts."""

    def __init__(self, *, meshes: Mesh, vertex_ptr: Tensor, cell_ptr: Tensor) -> None:
        if vertex_ptr.ndim != 1 or cell_ptr.ndim != 1:
            msg = "vertex_ptr and cell_ptr must be 1-D tensors"
            raise ValueError(msg)
        if vertex_ptr.shape[0] != cell_ptr.shape[0]:
            msg = (
                "vertex_ptr and cell_ptr must have the same length, "
                f"got {vertex_ptr.shape[0]} and {cell_ptr.shape[0]}"
            )
            raise ValueError(msg)
        if vertex_ptr.shape[0] < 2:
            msg = "pointer arrays must have at least 2 elements (for one mesh)"
            raise ValueError(msg)
        if vertex_ptr[-1].item() != meshes.num_vertices:
            msg = (
                f"vertex_ptr[-1] ({vertex_ptr[-1].item()}) does not match "
                f"mesh.num_vertices ({meshes.num_vertices})"
            )
            raise ValueError(msg)
        if cell_ptr[-1].item() != meshes.num_cells:
            msg = (
                f"cell_ptr[-1] ({cell_ptr[-1].item()}) does not match "
                f"mesh.num_cells ({meshes.num_cells})"
            )
            raise ValueError(msg)

        object.__setattr__(self, "meshes", meshes)
        object.__setattr__(self, "vertex_ptr", vertex_ptr)
        object.__setattr__(self, "cell_ptr", cell_ptr)

    @property
    def num_meshes(self) -> int:
        """Number of meshes in the batch."""
        return self.vertex_ptr.shape[0] - 1

    # -- Construction ---------------------------------------------------

    @classmethod
    def from_meshes(cls, meshes: Sequence[Mesh]) -> MeshBatch:
        """Build a batch from individual meshes.

        All meshes must share the same feature keys.
        """
        if not meshes:
            msg = "Cannot create a batch from an empty sequence of meshes"
            raise ValueError(msg)

        vertex_counts = [m.num_vertices for m in meshes]
        cell_counts = [m.num_cells for m in meshes]

        vertex_ptr = torch.zeros(len(meshes) + 1, dtype=torch.int64)
        cell_ptr = torch.zeros(len(meshes) + 1, dtype=torch.int64)
        torch.cumsum(
            torch.tensor(vertex_counts, dtype=torch.int64), dim=0, out=vertex_ptr[1:]
        )
        torch.cumsum(
            torch.tensor(cell_counts, dtype=torch.int64), dim=0, out=cell_ptr[1:]
        )

        return cls(meshes=concat(meshes), vertex_ptr=vertex_ptr, cell_ptr=cell_ptr)

    # -- Sequence protocol ----------------------------------------------

    def __len__(self) -> int:
        return self.num_meshes

    def __getitem__(self, idx: int) -> Mesh:
        if idx < 0:
            idx += self.num_meshes
        if not 0 <= idx < self.num_meshes:
            msg = f"index {idx} out of range for batch of {self.num_meshes} meshes"
            raise IndexError(msg)

        start_v = int(self.vertex_ptr[idx].item())
        end_v = int(self.vertex_ptr[idx + 1].item())
        start_c = int(self.cell_ptr[idx].item())
        end_c = int(self.cell_ptr[idx + 1].item())

        return Mesh(
            xy=self.meshes.xy[start_v:end_v],
            cell_indices=self.meshes.cell_indices[start_c:end_c] - start_v,
            vertex_features={
                k: v[start_v:end_v] for k, v in self.meshes.vertex_features.items()
            },
            cell_features={
                k: v[start_c:end_c] for k, v in self.meshes.cell_features.items()
            },
            global_features={k: v[idx] for k, v in self.meshes.global_features.items()},
        )

    def to(
        self,
        device: torch.device | str | None = None,
        float_dtype: torch.dtype | None = None,
    ) -> MeshBatch:
        """Move all tensors to the specified device and/or dtype."""
        return MeshBatch(
            meshes=self.meshes.to(device=device, float_dtype=float_dtype),
            vertex_ptr=self.vertex_ptr.to(device=device),
            cell_ptr=self.cell_ptr.to(device=device),
        )

    # -- Persistence ----------------------------------------------------

    def save(self, path: Path) -> None:
        """Save the batch to *path*."""
        torch.save(
            {
                "format": "tensormesh.MeshBatch",
                "version": 1,
                "mesh": self.meshes,
                "vertex_ptr": self.vertex_ptr,
                "cell_ptr": self.cell_ptr,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, *, mmap: bool = False) -> MeshBatch:
        """Load a batch from *path*.

        Expects a file created by `save`. If *mmap* is `True`, the
        tensors will be memory-mapped (only supported for CPU tensors).
        """
        data = torch.load(path, weights_only=False, mmap=mmap)

        if not isinstance(data, dict) or data.get("format") != "tensormesh.MeshBatch":
            msg = f"Unrecognised file format in {path}"
            raise ValueError(msg)

        return cls(
            meshes=data["mesh"],
            vertex_ptr=data["vertex_ptr"],
            cell_ptr=data["cell_ptr"],
        )
