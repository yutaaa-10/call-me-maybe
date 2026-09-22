PYTHON = python
NAME = -m src

DEFINITION = data/input/functions_definition.json
INPUT = data/input/function_calling_tests.json
OUTPUT = data/output/function_calling_results.json

install:
	uv tool pip install flake8 mypy

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
	flake8 src
	mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 src
	mypy src --strict

.PHONY: install run debug clean lint lint-strict%
