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


class TestExport:
    def test_export_simple_model(self) -> None:
        model = SimpleModel()
        mesh = _make_mesh()

        ep = torch.export.export(model, (mesh,))
        exported_out = ep.module()(mesh)
        eager_out = model(mesh)

        assert torch.allclose(exported_out, eager_out)

    def test_export_mesh_without_features(self) -> None:
        class BareModel(nn.Module):
            def forward(self, mesh: Mesh) -> Tensor:
                return mesh.xy.sum()

        xy = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        cell_indices = torch.tensor([[0, 1, 2]], dtype=torch.long)
        mesh = Mesh(xy=xy, cell_indices=cell_indices)
        model = BareModel()

        ep = torch.export.export(model, (mesh,))
        exported_out = ep.module()(mesh)

        assert torch.allclose(exported_out, model(mesh))

    def test_export_accesses_all_feature_types(self) -> None:
        class AllFeaturesModel(nn.Module):
            def forward(self, mesh: Mesh) -> Tensor:
                return (
                    mesh.xy.sum()
                    + mesh.vertex_features["vf"].sum()
                    + mesh.cell_features["cf"].sum()
                    + mesh.global_features["gf"]
                )

        mesh = _make_mesh()
        model = AllFeaturesModel()

        ep = torch.export.export(model, (mesh,))
        exported_out = ep.module()(mesh)

        assert torch.allclose(exported_out, model(mesh))
