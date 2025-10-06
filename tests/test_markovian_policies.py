"""Comprehensive tests for markovian_policies module."""
import numpy as np
import pytest
from research_bandits_methods.bandits.policies.markovian_policies import (
    EpsilonGreedy, UCB, GaussianThompson
)


class TestMarkovianPolicies:
    """Comprehensive test suite for Markovian bandit policies."""

    @pytest.fixture
    def rng(self):
        """Fixed RNG for reproducibility."""
        return np.random.default_rng(42)

    # Test: Welford's algorithm numerical stability
    def test_welford_numerical_stability(self, rng):
        """Verify Welford's algorithm maintains precision for large values."""
        policy = EpsilonGreedy(K=3, R=1, epsilon=0.1, rng=rng)
        
        # Add large values that would overflow cumulative sum
        large_rewards = np.array([1e10])
        for _ in range(1000):
            policy.update(np.array([0]), large_rewards)
            
        # Mean should still be approximately 1e10
        assert np.abs(policy.means[0, 0] - 1e10) < 1e5
        assert np.isfinite(policy.means[0, 0])

    # Test: UCB cache size limit
    def test_ucb_cache_bounded(self, rng):
        """Verify UCB cache doesn't grow unbounded."""
        policy = UCB(K=3, R=1, c=1.0, cache_size=100)
        
        # Run for many rounds
        for t in range(1, 1001):
            arms = policy.select_arm(t)
            policy.update(arms, rng.random(1))
            
        # Cache should be limited
        assert len(policy._log_cache) <= 100

    # Test: Thompson sampling variance floor
    def test_thompson_variance_floor(self, rng):
        """Verify Thompson sampling maintains positive variance."""
        policy = GaussianThompson(K=3, R=5, prior_var=1e-6, rng=rng)
        
        # Many updates to drive variance toward zero
        for t in range(1, 10_001):
            arms = policy.select_arm(t)
            policy.update(arms, np.ones(5))
            
        # Should still produce finite samples
        arms = policy.select_arm(10_001)
        assert np.all(np.isfinite(arms))

    # Test: Property methods return copies
    def test_property_methods_defensive_copy(self, rng):
        """Verify property methods return copies, not references."""
        policy = EpsilonGreedy(K=3, R=2, epsilon=0.1, rng=rng)
        
        means = policy.empirical_means
        counts = policy.pull_counts
        
        # Modify returned arrays
        means[0, 0] = 999.0
        counts[0, 0] = 999
        
        # Internal state should be unchanged
        assert policy.means[0, 0] != 999.0
        assert policy.counts[0, 0] != 999
