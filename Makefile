.PHONY: run install clean check runner lint test

.DEFAULT_GOAL := runner

# Run the app
run: install
	poetry run python -m freq_app.runner

# Install dependencies
install: pyproject.toml
	poetry install

# clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +

check:
	poetry run flake8 src/

runner: check run clean

	
# Run linting
lint:
	poetry run flake8 src/

# Run tests
test:
	poetry run pytest -v
