.PHONY: lint test security

# Development commands
lint:
	@echo "Running linter..."
	uv run --with pre-commit pre-commit run --all-files
	@echo "Linting complete."

test:
	@echo "Running tests..."
	uv run --group dev pytest --cov=src --cov-report=term-missing --cov-report=html
	@echo "Tests complete."

security:
	@echo "Running security scan..."
	uv run --group dev bandit -r src/
	@echo "Security scan complete."
