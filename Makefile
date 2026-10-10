UV ?= uv
ARGS ?= --help
TEST_ARGS ?=

.DEFAULT_GOAL := run
.PHONY: install run debug clean lint test

install:
	$(UV) sync

run:
	$(UV) run python -m src $(ARGS)

debug:
	$(UV) run python -m pdb -m src $(ARGS)

test:
	$(UV) run python -m pytest tests $(TEST_ARGS)

clean:
	find . -type d \( -name .git -o -name .venv -o -name data \) -prune -o \
		-type d \( -name __pycache__ -o -name .mypy_cache -o \
		-name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +

lint:
	$(UV) run --with flake8 flake8 . --extend-exclude=data,.venv
	$(UV) run --with mypy mypy . \
		--warn-return-any --warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs \
		--explicit-package-bases --exclude '(^|/)(data|\.venv)/'
