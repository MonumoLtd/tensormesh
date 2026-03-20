"""Execute tutorial notebooks and convert to markdown for docs.

Run this script before `zensical build` to regenerate docs/tutorials/*.md.
Generated files are gitignored — this script is the single source of truth.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import ExecutePreprocessor, TagRemovePreprocessor

_TUTORIALS = Path(__file__).parent / "tutorials"
_OUTPUT = _TUTORIALS

# Use the CDN renderer so Plotly charts are interactive without embedding 3 MB of
# JS per figure.  The script tag is injected once per figure and cached by the browser.
_SETUP = "import plotly.io as pio; pio.renderers.default = 'notebook_connected'"


def _build(nb_path: Path) -> None:
    nb = nbformat.read(nb_path, as_version=4)

    # Inject renderer config as a hidden cell before the notebook's own cells.
    setup_cell = nbformat.v4.new_code_cell(_SETUP)
    setup_cell["metadata"]["tags"] = ["remove-cell"]
    nb.cells.insert(0, setup_cell)

    ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(_TUTORIALS)}})

    tag_remover = TagRemovePreprocessor()
    tag_remover.remove_cell_tags = {"remove-cell"}
    nb, _ = tag_remover.preprocess(nb, {})

    exporter = MarkdownExporter()
    body, resources = exporter.from_notebook_node(nb)

    _OUTPUT.mkdir(exist_ok=True)
    (_OUTPUT / (nb_path.stem + ".md")).write_text(body, encoding="utf-8")

    if resources.get("outputs"):
        res_dir = _OUTPUT / f"{nb_path.stem}_files"
        res_dir.mkdir(exist_ok=True)
        for fname, data in resources["outputs"].items():
            (res_dir / fname).write_bytes(data)


for _nb in sorted(_TUTORIALS.glob("*.ipynb")):
    print(f"Building {_nb.name}...")  # noqa: T201
    _build(_nb)
