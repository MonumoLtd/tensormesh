# MeshTensor

The `MeshTensor` library provides tooling for representing and manipulating 2D triangular meshes
as collections of PyTorch tensors.

If you'd like a practical introduction, head to the [Getting Started tutorial](tutorials/getting_started.md).

## Core concept

The `Mesh` dataclass uses PyTorch tensors to represent 2D triangular mesh. `Mesh` objects 
have the following attributes

| Field | type | tensor(s) shape | Description |
|---|---|---|---|
| `xy` | Tensor | (num_vertex, 2) | Vertex coordinates |
| `cell_indices` | Tensor | (num_cells, 3) | vertex indices for each element |
| `vertex_features` | frozendict[str, Tensor] | (num_vertex, ...) | Per-vertex features |
| `cell_features` | frozendict[str, Tensor]| (num_cells, ...) | Per-cell features |
| `global_features` | frozendict[str, Tensor] | (...) | Mesh-level attributes |


## Capabilities

**Device and dtype conversion** — `mesh.to(device, float_dtype)` moves all tensors;
integer and boolean tensors preserve their dtype.

**Mesh operations** (`meshtensor.ops`) — stateless functions for cell areas,
unique undirected edges, vertex-to-cell interpolation, and tensor column
manipulation.

**Plotting** (`meshtensor.plots`) — Plotly-based visualisation of cells, vertex features, vector
fields, and wireframe plots.

**Persistence** — uses the build in `torch.save()` and `torch.load()`.

## Copyright Notice
Copyright 2026 Monumo Limited.  All rights reserved.