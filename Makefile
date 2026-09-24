PYTHON = python
NAME = -m src

DEFINITION = data/input/functions_definition.json
INPUT = data/input/function_calling_tests.json
OUTPUT = data/output/function_calling_results.json

install:
	uv sync

run:
	uv run $(PYTHON) $(NAME) \
		--functions_definition $(DEFINITION) \
		--input $(INPUT) \
		--output $(OUTPUT)

debug:
	@$(PYTHON) -m pdb src

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src
	uv run mypy src --strict

.PHONY: install run debug clean lint lint-strict%
