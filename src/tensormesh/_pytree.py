"""Register Mesh as a pytree node for torch.compile support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

import torch.utils._pytree as pytree

from tensormesh._frozen_dict import FrozenDict
from tensormesh.mesh import Mesh


def _mesh_flatten(
    mesh: Mesh,
) -> tuple[list[Any], tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]]:
    vf_keys = tuple(sorted(mesh.vertex_features.keys()))
    cf_keys = tuple(sorted(mesh.cell_features.keys()))
    gf_keys = tuple(sorted(mesh.global_features.keys()))

    children = [
        mesh.xy,
        mesh.cell_indices,
        *[mesh.vertex_features[k] for k in vf_keys],
        *[mesh.cell_features[k] for k in cf_keys],
        *[mesh.global_features[k] for k in gf_keys],
    ]
    aux = (vf_keys, cf_keys, gf_keys)
    return children, aux


def _mesh_unflatten(
    values: Iterable[Any], aux: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]
) -> Mesh:
    vals = list(values)
    vf_keys, cf_keys, gf_keys = aux

    xy = vals[0]
    cell_indices = vals[1]
    offset = 2

    vf = FrozenDict(
        dict(zip(vf_keys, vals[offset : offset + len(vf_keys)], strict=True))
    )
    offset += len(vf_keys)

    cf = FrozenDict(
        dict(zip(cf_keys, vals[offset : offset + len(cf_keys)], strict=True))
    )
    offset += len(cf_keys)

    gf = FrozenDict(
        dict(zip(gf_keys, vals[offset : offset + len(gf_keys)], strict=True))
    )

    return Mesh(
        xy=xy,
        cell_indices=cell_indices,
        vertex_features=vf,
        cell_features=cf,
        global_features=gf,
    )


pytree.register_pytree_node(
    Mesh, _mesh_flatten, _mesh_unflatten, serialized_type_name="tensormesh.Mesh"
)
