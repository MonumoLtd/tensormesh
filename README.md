# TensorMesh

PyTorch-based representation of 2D triangular meshes.

If you'd like a practical introduction, run `uv sync` and head to the [getting started tutorial](tutorials/getting_started.ipynb)!

## Core concepts

### `Mesh`

A dataclass using PyTorch tensors to represent 2D triangular mesh. `Mesh` objects 
have the following attributes

| Field | type | tensor(s) shape | Description |
|---|---|---|---|
| `xy` | Tensor (float) | (num_vertex, 2) | Vertex coordinates |
| `cell_indices` | Tensor (int64) | (num_cells, 3) | vertex indices for each element |
| `vertex_features` | dict[str, Tensor] | (num_vertex, ...) | Per-vertex features |
| `cell_features` | dict[str, Tensor]| (num_cells, ...) | Per-cell features |
| `global_features` | dict[str, Tensor] | (...) | Mesh-level attributes |

**Device and dtype conversion** — `mesh.to(device, dtype)` moves all tensors;
integer and boolean tensors preserve their dtype.

**Mesh operations** (`tensormesh.ops`) — stateless functions for cell areas,
unique undirected edges, vertex-to-cell interpolation, and tensor column
manipulation.

**Plotting** (`tensormesh.plots`) — Plotly-based visualisation of cells, vertex features, vector
fields, and wireframe plots

**Persistence** — uses the build in `torch.save()` and `torch.load()`.

## Development

```bash
uv sync                  # install / sync all dependencies
make lint                # ruff check + format, taplo fmt
make typecheck           # pyright (strict) over src/ and tests/
make test                # pytest with 4 parallel workers + coverage
```

To build the documentation run `make docs`. This will create a `build_docs/index.html`
that you can open in your browser.

## Contributors
This library benefited from contributions, feedback, and support from the following people (listed in alphabetical order):

- Chris Hull
- Colin Gravill
- Lisa Maria Kreusser
- Marcel Nonnenmacher
- Nicolas Durrande
- Tom Gillam
- William Gallafent

## Copyright Notice
Copyright 2026 Monumo Limited.  All rights reserved.