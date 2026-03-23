from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from tensormesh import Mesh

if TYPE_CHECKING:
    from torch import Tensor


def _make_mesh() -> Mesh:
    """Simple 2-triangle, 4-vertex mesh built with dict (backward compat)."""
    xy = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cell_indices = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    return Mesh(
        xy=xy,
        cell_indices=cell_indices,
        vertex_features={"vf": torch.arange(4, dtype=torch.float32)},
        cell_features={"cf": torch.ones(2)},
        global_features={"gf": torch.tensor(42.0)},
    )


class SimpleModel(nn.Module):
    """Minimal model that takes a Mesh and returns a scalar."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, mesh: Mesh) -> Tensor:
        return self.linear(mesh.xy).sum() + mesh.vertex_features["vf"].sum()


class TestCompile:
    def test_compiled_model_with_mesh_input(self) -> None:
        model = SimpleModel()
        compiled = torch.compile(model, fullgraph=True)

        mesh = _make_mesh()
        eager_out = model(mesh)
        compiled_out = compiled(mesh)

        assert torch.allclose(eager_out, compiled_out)
