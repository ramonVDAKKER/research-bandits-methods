# research-bandits-methods

Python package implementing sampling policies for contextual bandits and statistical inference methods.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Make

## Installation

```bash
# Clone the repository
git clone https://github.com/ramonvdakker/research-bandits-methods.git
cd research-bandits-methods

# Install dependencies
uv sync

# Install with development dependencies
uv sync --group dev
```

## Usage

TODO

## Development

TODO

### Running Tests

```bash
# Run all tests with coverage
make test

# Run specific test
uv run --group dev pytest tests/test_main.py::test_main

# View coverage report
open htmlcov/index.html
```

### Linting and Code Quality

```bash
# Run all pre-commit hooks
make lint

# Install pre-commit hooks (runs automatically on commit)
uv run --with pre-commit pre-commit install
```

The project enforces code quality through pre-commit hooks:
- **ruff**: Linting and formatting
- **isort**: Import sorting
- **mypy**: Type checking
- **pydocstyle**: Docstring conventions
- **interrogate**: Docstring coverage
- **bandit**: Security checks
- **detect-secrets**: Secret detection

## License

See [LICENSE](LICENSE) file for details.
