# research-bandits-methods

A Python package for Monte Carlo simulation of multi-armed bandit problems with support for various sampling policies and statistical inference methods.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

`research-bandits-methods` provides a comprehensive framework for simulating and analyzing multi-armed bandit algorithms. The package emphasizes:

- **Efficient vectorized simulations** across multiple Monte Carlo replications
- **Fair policy comparison** using pre-generated counterfactual outcomes
- **Flexible reward distributions** including Gaussian, Bernoulli, Student's t, and custom mixtures

## Features

### Bandit Policies

- **ε-Greedy**: Simple exploration-exploitation tradeoff with tunable exploration rate
- **UCB (Upper Confidence Bound)**: Optimistic action selection with logarithmic regret guarantees
- **Gaussian Thompson Sampling**: Bayesian approach with Gaussian priors and assuming Gaussian rewards

Forthcoming:
- **translation-invariant Gaussian Thompson sampling**

All policies support:
- Vectorized operations across R parallel runs
- Forced initial exploration to ensure all arms are sampled

### Reward Distributions

- **GaussianRewards**: Gaussian distributions with configurable per-arm means and variances
- **BernoulliRewards**: Binary outcomes for click/conversion modeling
- **StudentTRewards**: Heavy-tailed distributions for robustness testing
- **MixtureDistribution**: Discrete mixtures for modeling heterogeneous scenarios
- **PerArmDistribution**: Heterogeneous distributions across arms

### Simulation Modes

1. **Simulation mode**: Pre-generates all counterfactual outcomes to facilitate policy comparison in Monte Carlo studies
2. **Online mode**: Samples rewards only for selected arm

## Installation

### Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Install dependencies

```bash
uv sync
```

### Install with development dependencies

```bash
uv sync --group dev
```

## Quick Start

TODO: starter notebook


## Development

### Running Tests

```bash
# Run all tests with coverage
make test

# Or using uv directly
uv run --group dev pytest --cov=src --cov-report=term-missing --cov-report=html

# Run specific test
uv run --group dev pytest tests/test_markovian_policies.py::test_epsilon_greedy
```

### Code Quality

The project enforces strict code quality standards via pre-commit hooks:

```bash
# Run all linters
make lint

# Install pre-commit hooks
uv run --with pre-commit pre-commit install
```

Quality checks include:
- **isort**: Import sorting (black profile, 120 char line length)
- **ruff**: Fast linting and formatting (Python + Jupyter)
- **mypy**: Static type checking
- **pydocstyle**: Docstring conventions
- **interrogate**: Docstring coverage enforcement
- **detect-secrets**: Secret detection

### Project Structure

```
research-bandits-methods/
├── src/research_bandits_methods/
│   ├── bandits/
│   │   ├── policies/
│   │   │   ├── markovian_policies.py
│   │   │   └── __init__.py
│   │   ├── distributions.py
│   ├── contextual_bandits/            # (Future: contextual bandits)
│   └── constants.py                   # Numerical constants
├── tests/
│   ├── test_markovian_policies.py
│   ├── test_distributions.py
├── pyproject.toml                     # Package configuration
└── Makefile                           # Developer shortcuts
```

## License

MIT License - See LICENSE file for details
