PYTHON = python
NAME = -m src
DIFINITION = call-me-maybe/data/input/function_calling_tests.json
INPUT = call-me-maybe/data/input/functions_definition.json
OUTPUT = call-me-maybe/data/output/function_calling_results.json

install:
	uv tool pip install flake8 mypy

run:
	uv run $(PYTHON) $(NAME) --functions_definition $(DIFINITTION) --input $(INPUT) --output $(OUTPUT)

debug:
	@$(PYTHON) -m pdb src

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict

.PHONY: install run debug clean lint lint-strict%
