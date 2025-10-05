"""Test reward distribution models with statistical validation."""

import numpy as np
import pytest
from scipy import stats

from research_bandits_methods.bandits.distributions import (
    BernoulliRewards,
    GaussianRewards,
    MixtureDistribution,
    PerArmDistribution,
    StudentTRewards,
)


@pytest.mark.parametrize(
    "means,variances,expected_variances",
    [
        # Scalar variance (broadcast)
        (np.array([0.3, 0.5, 0.8]), 1.0, np.array([1.0, 1.0, 1.0])),
        # Per-arm variances
        (np.array([0.3, 0.5, 0.8]), np.array([1.0, 2.0, 0.5]), np.array([1.0, 2.0, 0.5])),
    ],
)
def test_gaussian_rewards_basic(means, variances, expected_variances):
    """Test basic GaussianRewards functionality with scalar and per-arm variances."""
    dist = GaussianRewards(means=means, variances=variances)
    sims = dist.generate_counterfactuals(T=100, R=50, rng=np.random.default_rng(42))

    assert dist.K == 3
    assert sims.shape == (100, 3, 50)
    assert np.array_equal(dist.means, means)
    assert np.array_equal(dist.variances, expected_variances)


def test_gaussian_rewards_statistical_properties():
    """Behavior test: verify Gaussian distribution statistical properties."""
    means = np.array([1.0, 2.0, 3.0])
    variances = np.array([1.0, 2.0, 0.5])
    dist = GaussianRewards(means=means, variances=variances)
    rng = np.random.default_rng(42)

    T = 10_000
    cf = dist.generate_counterfactuals(T=T, R=1, rng=rng)

    for k in range(3):
        arm_data = cf[:, k, 0]

        # Test mean
        sample_mean = arm_data.mean()
        assert abs(sample_mean - means[k]) < 3 * np.sqrt(variances[k] / T), f"Arm {k} mean mismatch"

        # Test normality using Shapiro-Wilk test
        _, p_value = stats.shapiro(arm_data)
        assert p_value > 0.01, f"Arm {k} fails normality test"