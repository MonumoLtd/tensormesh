"""Pre-generate docs/api/ pages and update the API Reference nav in mkdocs.yml.

Run this script before `zensical build` to keep API docs in sync with
the source tree. Generated docs/api/*.md files are gitignored — this
script is the single source of truth. The API Reference section of
mkdocs.yml is rewritten in-place on every run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml  # transitive dep via mkdocstrings-python → mkdocs → pyyaml

_ACRONYMS = {"ml"}


def _humanize(segment: str) -> str:
    """Convert a snake_case package segment to a human-readable title.

    Known short acronyms (e.g. 'ml') are uppercased; everything else is
    title-cased word by word.
    """
    return " ".join(
        word.upper() if word in _ACRONYMS else word.title()
        for word in segment.split("_")
    )


def _overview_label(parts: tuple[str, ...]) -> str:
    """Return the page title for a package overview, e.g. 'TensorMesh Overview'."""
    return " ".join(_humanize(p) for p in parts) + " Overview"


def _module_title(parts: tuple[str, ...]) -> str:
    """Return the page title for a module, e.g. 'TensorMesh: Model'."""
    parent = " ".join(_humanize(p) for p in parts[:-1])
    name = _humanize(parts[-1])
    return f"{parent}: {name}"


def _module_docstring(path: Path) -> str | None:
    """Return the module-level docstring from a Python source file, or None."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return ast.get_docstring(tree)


def _build_pkg_nav(pkg_path: Path, src: Path) -> list[dict[str, str | list[str]]]:
    """Recursively build the mkdocs nav entry list for a package."""
    parts = pkg_path.relative_to(src).parts
    api_prefix = "api/" + "/".join(parts)

    entries: list[dict[str, str | list[str]]] = [{"Overview": f"{api_prefix}/index.md"}]

    for mod in sorted(pkg_path.glob("*.py")):
        if mod.stem in ("__init__", "__main__"):
            continue
        entries.append({mod.stem: f"{api_prefix}/{mod.stem}.md"})

    entries.extend(
        {subpkg.name: _build_pkg_nav(subpkg, src)}
        for subpkg in sorted(
            d for d in pkg_path.iterdir() if d.is_dir() and (d / "__init__.py").exists()
        )  # pyright: ignore[reportArgumentType]
    )

    return entries


src = Path(__file__).parent.parent / "src"
api_dir = Path(__file__).parent / "api"
mkdocs_base = Path(__file__).parent.parent / "mkdocs-base.yml"
mkdocs_yml = Path(__file__).parent.parent / "mkdocs.yml"

# ── Generate docs/api/*.md ────────────────────────────────────────────────

if api_dir.exists():
    for f in api_dir.rglob("*.md"):
        f.unlink()

api_dir.mkdir(exist_ok=True)

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    parts = tuple(module_path.parts)

    if parts[-1] == "__main__":
        continue

    is_package = parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]
        doc_path = api_dir / Path(*parts) / "index.md"
    else:
        doc_path = api_dir / Path(*parts[:-1]) / f"{parts[-1]}.md"

    doc_path.parent.mkdir(parents=True, exist_ok=True)

    if is_package:
        content = f"# {_overview_label(parts)}\n\n"
        docstring = _module_docstring(path)
        if docstring:
            content += docstring + "\n\n"
    else:
        content = f"# {_module_title(parts)}\n\n"

    module_id = ".".join(parts)
    content += f"::: {module_id}\n"

    if not is_package:
        content += (
            "\n## Internal API\n\n"
            f"::: {module_id}\n"
            "    options:\n"
            "      filters:\n"
            '        - "^_[^_]"\n'
            "      show_root_heading: false\n"
            "      show_root_toc_entry: false\n"
        )

    doc_path.write_text(content, encoding="utf-8")

# ── Update API Reference nav in mkdocs.yml ────────────────────────────────

config = yaml.safe_load(mkdocs_base.read_text(encoding="utf-8"))

api_nav = _build_pkg_nav(src / "tensormesh", src)

for i, entry in enumerate(config["nav"]):
    if isinstance(entry, dict) and "API Reference" in entry:
        config["nav"][i] = {"API Reference": api_nav}
        break

mkdocs_yml.write_text(
    yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
