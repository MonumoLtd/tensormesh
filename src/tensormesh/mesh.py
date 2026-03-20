from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from frozendict import frozendict
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@typing.final
@dataclass(kw_only=True, frozen=True, eq=False)
class Mesh:
    """Tensor representation of a 2D triangular mesh.

    The constructor enforces consistency on
    * the shapes of the tensors relative to each other;
    * the bounds of the cell vertex indices;
    * the device on which all tensors reside.
    """

    xy: Tensor
    """(num_vertices, 2) float tensor with the mesh vertex coordinates."""

    cell_indices: Tensor
    """(num_cells, 3) long tensor with the vertex indices of each triangular cell."""

    vertex_features: frozendict[str, Tensor] = field(
        default_factory=frozendict[str, Tensor]
    )
    """Attributes defined at the mesh nodes; values of shape (num_vertices, ...)."""

    cell_features: frozendict[str, Tensor] = field(
        default_factory=frozendict[str, Tensor]
    )
    """Attributes defined at the mesh elements; values of shape (num_cells, ...)."""

    global_features: frozendict[str, Tensor] = field(
        default_factory=frozendict[str, Tensor]
    )
    """Attributes defined at the mesh level; values of shape (...)."""

    def __post_init__(self) -> None:
        _validate_shapes(self)
        _validate_index_bounds(self)
        _validate_device(self)

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the mesh."""
        return self.xy.shape[0]

    @property
    def num_cells(self) -> int:
        """Number of triangular cells in the mesh."""
        return self.cell_indices.shape[0]

    @property
    def device(self) -> torch.device:
        """Device on which all mesh tensors reside."""
        return self.xy.device

    def to(
        self,
        device: torch.device | str | None = None,
        float_dtype: torch.dtype | None = None,
    ) -> Mesh:
        """Move all tensors to the specified device and/or dtype.

        `torch.long` and `torch.bool` tensors always preserve their dtype
        regardless of *float_dtype*.

        Args:
            device: Target device (e.g. `"cuda:0"`).  `None` leaves the
                device unchanged.
            float_dtype: Target floating-point dtype for float tensors.  `None`
                leaves the dtype unchanged.

        Returns:
            A new :class:`Mesh` with all tensors on the requested device/dtype.
        """

        def convert(x: Tensor) -> Tensor:
            if float_dtype is not None and x.is_floating_point():
                return x.to(device=device, dtype=float_dtype)
            return x.to(device=device)

        vertices = frozendict({k: convert(v) for k, v in self.vertex_features.items()})
        cells = frozendict({k: convert(v) for k, v in self.cell_features.items()})
        glob = frozendict({k: convert(v) for k, v in self.global_features.items()})
        return Mesh(
            xy=convert(self.xy),
            cell_indices=convert(self.cell_indices),
            vertex_features=vertices,
            cell_features=cells,
            global_features=glob,
        )

    def with_features(
        self,
        *,
        vertex_features: Mapping[str, torch.Tensor] = frozendict(),
        cell_features: Mapping[str, torch.Tensor] = frozendict(),
        global_features: Mapping[str, torch.Tensor] = frozendict(),
    ) -> Mesh:
        """Create a new `Mesh` with additional features."""
        return Mesh(
            xy=self.xy,
            cell_indices=self.cell_indices,
            vertex_features=self.vertex_features | frozendict(vertex_features),
            cell_features=self.cell_features | frozendict(cell_features),
            global_features=self.global_features | frozendict(global_features),
        )

    def delete_features(
        self,
        *,
        vertex_features: Sequence[str] = (),
        cell_features: Sequence[str] = (),
        global_features: Sequence[str] = (),
    ) -> Mesh:
        """Create a new `Mesh` without the specified features."""
        new_vertex_features = frozendict(
            {k: v for k, v in self.vertex_features.items() if k not in vertex_features}
        )
        new_cell_features = frozendict(
            {k: v for k, v in self.cell_features.items() if k not in cell_features}
        )
        new_global_features = frozendict(
            {k: v for k, v in self.global_features.items() if k not in global_features}
        )
        return Mesh(
            xy=self.xy,
            cell_indices=self.cell_indices,
            vertex_features=new_vertex_features,
            cell_features=new_cell_features,
            global_features=new_global_features,
        )

    def rename_features(
        self,
        *,
        vertex_mapping: Mapping[str, str] | None = None,
        cell_mapping: Mapping[str, str] | None = None,
        global_mapping: Mapping[str, str] | None = None,
    ) -> Mesh:
        """Create a new `Mesh` with renamed features."""
        vertex_mapping = vertex_mapping or {}
        cell_mapping = cell_mapping or {}
        global_mapping = global_mapping or {}

        new_vertex_features = frozendict(
            {vertex_mapping.get(k, k): v for k, v in self.vertex_features.items()}
        )
        new_cell_features = frozendict(
            {cell_mapping.get(k, k): v for k, v in self.cell_features.items()}
        )
        new_global_features = frozendict(
            {global_mapping.get(k, k): v for k, v in self.global_features.items()}
        )
        return Mesh(
            xy=self.xy,
            cell_indices=self.cell_indices,
            vertex_features=new_vertex_features,
            cell_features=new_cell_features,
            global_features=new_global_features,
        )

    def clone(self) -> Mesh:
        """Return a deep copy with all tensors cloned."""
        return Mesh(
            xy=self.xy.clone(),
            cell_indices=self.cell_indices.clone(),
            vertex_features=frozendict(
                {k: v.clone() for k, v in self.vertex_features.items()}
            ),
            cell_features=frozendict(
                {k: v.clone() for k, v in self.cell_features.items()}
            ),
            global_features=frozendict(
                {k: v.clone() for k, v in self.global_features.items()}
            ),
        )


# ----------------------------------------------------------------------
# validation helpers
# ----------------------------------------------------------------------


def _validate_shapes(mesh: Mesh) -> None:
    """Ensure compatibility of tensor shapes within the mesh object."""
    if mesh.xy.ndim != 2 or mesh.xy.shape[1] != 2:
        msg = f"xy must have shape (num_vertices, 2), got {tuple(mesh.xy.shape)}"
        raise ValueError(msg)

    if mesh.cell_indices.ndim != 2 or mesh.cell_indices.shape[1] != 3:
        msg = (
            "cell_indices must have shape (num_cells, 3), "
            f"got {tuple(mesh.cell_indices.shape)}"
        )
        raise ValueError(msg)

    for feature_name, feature_tensor in mesh.vertex_features.items():
        if feature_tensor.shape[0] != mesh.num_vertices:
            msg = (
                f"Vertex feature '{feature_name}' has incompatible shape; "
                f"found {tuple(feature_tensor.shape)}, "
                f"expected ({mesh.num_vertices}, ...)"
            )
            raise ValueError(msg)

    for feature_name, feature_tensor in mesh.cell_features.items():
        if feature_tensor.shape[0] != mesh.num_cells:
            msg = (
                f"Cell feature '{feature_name}' has incompatible shape; "
                f"found {tuple(feature_tensor.shape)}, "
                f"expected ({mesh.num_cells}, ...)"
            )
            raise ValueError(msg)


def _validate_device(mesh: Mesh) -> None:
    """Ensure all tensors reside on the same device."""
    if mesh.cell_indices.device != mesh.device:
        msg = (
            f"'cell_indices' is on device {mesh.cell_indices.device} "
            f"but 'xy' is on {mesh.xy.device}"
        )
        raise ValueError(msg)

    for name in ("vertex_features", "cell_features", "global_features"):
        for t in getattr(mesh, name).values():
            if t.device != mesh.device:
                msg = f"'{name}' is on device {t.device} but mesh is on {mesh.device}"
                raise ValueError(msg)


def _validate_index_bounds(mesh: Mesh) -> None:
    """Ensure the cell indices are compatible with the vertex locations."""
    if mesh.cell_indices.numel() != 0:
        n_v = int(mesh.xy.shape[0])
        lo, hi = mesh.cell_indices.aminmax()
        if int(lo.item()) < 0:
            msg = f"'cell_indices' contains negative index {int(lo.item())}"
            raise ValueError(msg)
        if int(hi.item()) >= n_v:
            msg = f"'cell_indices' contains index {int(hi.item())} >= n_vertices={n_v}"
            raise ValueError(msg)


# ----------------------------------------------------------------------
# Mesh related methods
# ----------------------------------------------------------------------


def concat(meshes: Sequence[Mesh]) -> Mesh:
    """Concatenate meshes that share the same schema.

    Vertex indices are offset incrementally.

    Args:
        meshes: sequence of meshes to concatenate.

    Returns:
        A new mesh containing the concatenated data from *meshes*.
    """
    if not meshes:
        msg = "Cannot concatenate an empty sequence of meshes"
        raise ValueError(msg)

    num_vertices = 0
    vertex_names = list(meshes[0].vertex_features)
    cell_names = list(meshes[0].cell_features)
    global_names = list(meshes[0].global_features)

    all_xy, all_cell_indices = [], []
    all_vertex_features = {key: [] for key in vertex_names}
    all_cell_features = {key: [] for key in cell_names}
    all_global_features = {key: [] for key in global_names}
    for mesh in meshes:
        all_xy.append(mesh.xy)
        all_cell_indices.append(mesh.cell_indices + num_vertices)
        num_vertices += mesh.num_vertices

        if set(mesh.vertex_features) != set(vertex_names):
            msg = "All meshes must have the same vertex feature keys"
            raise ValueError(msg)
        for key in vertex_names:
            all_vertex_features[key].append(mesh.vertex_features[key])

        if set(mesh.cell_features) != set(cell_names):
            msg = "All meshes must have the same cell feature keys"
            raise ValueError(msg)
        for key in cell_names:
            all_cell_features[key].append(mesh.cell_features[key])

        if set(mesh.global_features) != set(global_names):
            msg = "All meshes must have the same global feature keys"
            raise ValueError(msg)
        for key in global_names:
            all_global_features[key].append(mesh.global_features[key])

    return Mesh(
        xy=torch.cat(all_xy, dim=0),
        cell_indices=torch.cat(all_cell_indices, dim=0),
        vertex_features=frozendict(
            {k: torch.cat(v, dim=0) for k, v in all_vertex_features.items()}
        ),
        cell_features=frozendict(
            {k: torch.cat(v, dim=0) for k, v in all_cell_features.items()}
        ),
        global_features=frozendict(
            {k: torch.cat(v, dim=0) for k, v in all_global_features.items()}
        ),
    )


def align_schema(
    mesh: Mesh,
    target_mesh: Mesh,
    fill_float: float = 0.0,
    fill_int: int = 0,
    *,
    fill_bool: bool = False,
) -> Mesh:
    """Align the schema of *mesh* to match *target_mesh*.

    This is useful when concatenating meshes with different sets of features.
    Missing features in *mesh* are filled with the specified default values.

    Args:
        mesh: The mesh to be aligned.
        target_mesh: The mesh whose schema we want to match.
        fill_float: Default value for missing float features.
        fill_int: Default value for missing integer features.
        fill_bool: Default value for missing boolean features.

    Returns:
        A new mesh with the same schema as *target_mesh*.
    """

    def consolidate_dictionary_keys(
        features: Mapping[str, torch.Tensor],
        target_features: Mapping[str, torch.Tensor],
        num_items: int,
    ) -> dict[str, torch.Tensor]:
        consolidated = dict(**features)
        for key in target_features:
            if key not in features:
                shape = (num_items, *target_features[key].shape[1:])
                dtype = target_features[key].dtype
                if dtype.is_floating_point:
                    consolidated[key] = torch.full(
                        shape, fill_float, dtype=dtype, device=mesh.device
                    )
                elif dtype == torch.long:
                    consolidated[key] = torch.full(
                        shape, fill_int, dtype=dtype, device=mesh.device
                    )
                elif dtype == torch.bool:
                    consolidated[key] = torch.full(
                        shape, fill_bool, dtype=dtype, device=mesh.device
                    )
                else:
                    msg = f"Unsupported dtype {dtype} for feature '{key}'"
                    raise ValueError(msg)
        return consolidated

    vertex_features = consolidate_dictionary_keys(
        mesh.vertex_features, target_mesh.vertex_features, mesh.num_vertices
    )
    cell_features = consolidate_dictionary_keys(
        mesh.cell_features, target_mesh.cell_features, mesh.num_cells
    )
    global_features = consolidate_dictionary_keys(
        mesh.global_features, target_mesh.global_features, 1
    )

    return Mesh(
        xy=mesh.xy,
        cell_indices=mesh.cell_indices,
        vertex_features=frozendict(vertex_features),
        cell_features=frozendict(cell_features),
        global_features=frozendict(global_features),
    )
