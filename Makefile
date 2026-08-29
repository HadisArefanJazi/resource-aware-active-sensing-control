.PHONY: install test lint experiment

install:
	python -m pip install -e ".[dev]"

test:
	pytest --cov=resource_aware_control --cov-report=term-missing

lint:
	ruff check .
	ruff format --check .

experiment:
	resource-control experiment
