.PHONY: install
install:
	uv sync

.PHONY: lint
lint:
	-uv run ruff check --fix
	-uv run ruff format
	-RUST_LOG=warn uv run taplo fmt

.PHONY: lint_unsafe
lint_unsafe:
	-uv run ruff check --fix --unsafe-fixes
	-uv run ruff format
	-RUST_LOG=warn uv run taplo fmt

.PHONY: typecheck
typecheck:
	uv run pyright
	find . -name "*.ipynb" -exec uv run --locked nbqa pyright {} +;

.PHONY: test
test:
	COVERAGE_CORE=sysmon OMP_NUM_THREADS=1 uv run pytest --numprocesses 4 --cov=tensormesh --cov-report=xml

.PHONY: nbstripout
nbstripout:
	uv run nbstripout `find docs/tutorials -type f -name *.ipynb` --drop-empty-cells --extra-keys "metadata.language_info.version"

.PHONY: docs
docs:
	uv run python docs/gen_api_pages.py
	uv run python docs/gen_tutorials.py
	uv run zensical build