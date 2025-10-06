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


class TestGaussianRewards:
    def _assert_gaussian_statistics(
        self,
        samples: np.ndarray,
        expected_mean: float,
        expected_variance: float,
        alpha: float = 0.01,
    ) -> None:
        """Assert that samples follow Gaussian distribution with given parameters.

        Parameters
        ----------
        samples : np.ndarray
            1D array of samples to test.
        expected_mean : float
            Expected mean of the distribution.
        expected_variance : float
            Expected variance of the distribution.
        alpha : float, default=0.01
            Significance level for hypothesis tests.
        """
        n = len(samples)
        df = n - 1
        z_norm = stats.norm.ppf(1 - alpha / 2)
        z_var_l = stats.chi2.ppf(alpha / 2, df)
        z_var_r = stats.chi2.ppf(1 - alpha / 2, df)

        # Test mean in confidence interval
        sample_mean = samples.mean()
        assert abs(sample_mean - expected_mean) < z_norm * np.sqrt(
            expected_variance / n
        ), f"Mean {sample_mean:.4f} outside CI for expected {expected_mean}"

        # Test variance in confidence interval
        sample_var = samples.var(ddof=1)
        assert (df * sample_var / z_var_r) < expected_variance, (
            f"Variance {sample_var:.4f} outside CI for expected {expected_variance}"
        )
        assert (df * sample_var / z_var_l) > expected_variance, (
            f"Variance {sample_var:.4f} outside CI for expected {expected_variance}"
        )

        # Test normality using Jarque-Bera test
        _, p_value = stats.jarque_bera(samples)
        assert p_value > alpha, f"Failed normality test (p={p_value:.4f})"

    @pytest.mark.parametrize(
        "means,variances,expected_variances",
        [
            # Variance
            (np.array([0.3, 0.5, 0.8]), 1.0, np.array([1.0, 1.0, 1.0])),
            # Per-arm variances
            (
                np.array([0.3, 0.5, 0.8]),
                np.array([1.0, 2.0, 0.5]),
                np.array([1.0, 2.0, 0.5]),
            ),
        ],
    )
    def test_gaussian_rewards_basic(self, means, variances, expected_variances):
        """Test basic GaussianRewards functionality with scalar and per-arm variances."""
        dist = GaussianRewards(means=means, variances=variances)
        sims = dist.generate_counterfactuals(T=100, R=50, rng=np.random.default_rng(42))

        assert dist.K == 3
        assert sims.shape == (100, 3, 50)
        assert np.array_equal(dist.means, means)
        assert np.array_equal(dist.variances, expected_variances)

    def test_gaussian_rewards_statistical_properties(self):
        """Behavior test: verify Gaussian distribution statistical properties."""
        means = np.array([1.0, 2.0, 3.0])
        variances = np.array([1.0, 2.0, 0.5])
        dist = GaussianRewards(means=means, variances=variances)
        rng = np.random.default_rng(42)

        T = 100_000
        cf = dist.generate_counterfactuals(T=T, R=1, rng=rng)

        for k in range(len(means)):
            arm_data = cf[:, k, 0]
            self._assert_gaussian_statistics(arm_data, means[k], variances[k])

    def test_gaussian_rewards_sample_online(self):
        """Test sample_online method returns correct distribution."""
        means = np.array([1.0, 2.0, 3.0])
        variances = np.array([1.0, 2.0, 0.5])
        dist = GaussianRewards(means=means, variances=variances)
        rng = np.random.default_rng(42)

        n_samples = 100_000
        for k in range(len(means)):
            samples = np.array([dist.sample_online(k, rng) for _ in range(n_samples)])

            # Verify samples are floats
            assert isinstance(samples[0], (float, np.floating))

            # Verify statistical properties
            self._assert_gaussian_statistics(samples, means[k], variances[k])

    def test_gaussian_small_variance(self):
        """Test Gaussian with very small variance."""
        dist = GaussianRewards(means=[5.0], variances=1e-6)
        sims = dist.generate_counterfactuals(
            T=1000, R=10, rng=np.random.default_rng(42)
        )

        # All samples should be very close to mean
        assert np.abs(sims.mean() - 5.0) < 0.01
        assert sims.std() < 0.01

    def test_gaussian_rewards_empty_means(self):
        """Test GaussianRewards rejects empty means."""
        with pytest.raises((ValueError, IndexError)):
            GaussianRewards(means=[])


class TestBernoulliRewards:
    def _assert_bernoulli_statistics(
        self,
        samples: np.ndarray,
        expected_prob: float,
        alpha: float = 0.01,
    ) -> None:
        """Assert that samples follow Bernoulli distribution with given probability.

        Parameters
        ----------
        samples : np.ndarray
            1D array of binary samples (0 or 1) to test.
        expected_prob : float
            Expected success probability of the distribution.
        alpha : float, default=0.01
            Significance level for hypothesis tests.
        """
        n = len(samples)

        # Verify samples are binary
        assert np.all((samples == 0) | (samples == 1)), "Samples must be 0 or 1"

        # Chi-square goodness of fit test for Bernoulli
        observed = np.array([np.sum(samples == 0), np.sum(samples == 1)])
        expected = np.array([(1 - expected_prob) * n, expected_prob * n])
        _, p_value = stats.chisquare(observed, expected)
        assert p_value > alpha, f"Failed chi-square test (p={p_value:.4f})"

    def test_bernoulli_rewards_basic(self):
        """Test basic BernoulliRewards functionality."""
        probs = np.array([0.2, 0.5, 0.8])
        dist = BernoulliRewards(probs=probs)
        sims = dist.generate_counterfactuals(T=100, R=50, rng=np.random.default_rng(42))

        assert dist.K == 3
        assert sims.shape == (100, 3, 50)
        assert np.array_equal(dist.probs, probs)

        # Verify all values are 0 or 1
        assert np.all((sims == 0) | (sims == 1)), "All samples must be 0 or 1"

    def test_bernoulli_rewards_invalid_probs(self):
        """Test BernoulliRewards rejects invalid probabilities."""
        # Probability > 1
        with pytest.raises(ValueError, match="All probabilities must be in"):
            BernoulliRewards(probs=[0.5, 1.5])

        # Negative probability
        with pytest.raises(ValueError, match="All probabilities must be in"):
            BernoulliRewards(probs=[0.5, -0.1])

    def test_bernoulli_rewards_statistical_properties(self):
        """Behavior test: verify Bernoulli distribution statistical properties."""
        probs = np.array([0.2, 0.5, 0.8])
        dist = BernoulliRewards(probs=probs)
        rng = np.random.default_rng(42)

        T = 100_000
        cf = dist.generate_counterfactuals(T=T, R=1, rng=rng)

        for k in range(len(probs)):
            arm_data = cf[:, k, 0]
            self._assert_bernoulli_statistics(arm_data, probs[k])

    def test_bernoulli_rewards_sample_online(self):
        """Test sample_online method returns correct distribution."""
        probs = np.array([0.2, 0.5, 0.8])
        dist = BernoulliRewards(probs=probs)
        rng = np.random.default_rng(42)

        n_samples = 100_000
        for k in range(len(probs)):
            samples = np.array([dist.sample_online(k, rng) for _ in range(n_samples)])

            # Verify samples are floats
            assert isinstance(samples[0], (float, np.floating))

            # Verify statistical properties
            self._assert_bernoulli_statistics(samples, probs[k])

    def test_bernoulli_extreme_probs(self):
        """Test Bernoulli with extreme probabilities (0.0 and 1.0)."""
        dist = BernoulliRewards(probs=[0.0, 0.5, 1.0])
        sims = dist.generate_counterfactuals(T=100, R=10, rng=np.random.default_rng(42))

        # Arm 0 should always be 0
        assert np.all(sims[:, 0, :] == 0.0)
        # Arm 2 should always be 1
        assert np.all(sims[:, 2, :] == 1.0)


class TestMixtureDistribution:
    def test_mixture_basic_broadcast_weights(self):
        """Test basic MixtureDistribution with broadcast weights (same for all arms)."""
        components = [
            GaussianRewards(means=[0.3, 0.5, 0.8], variances=1.0),
            GaussianRewards(means=[0.5, 0.7, 1.0], variances=2.0),
        ]
        weights = np.array([0.7, 0.3])
        dist = MixtureDistribution(components=components, weights=weights)

        assert dist.K == 3
        assert dist.n_components == 2
        assert dist.weights.shape == (3, 2)

        # Verify broadcast worked correctly
        assert np.allclose(dist.weights[0], weights)
        assert np.allclose(dist.weights[1], weights)
        assert np.allclose(dist.weights[2], weights)

        # Test counterfactuals generation
        sims = dist.generate_counterfactuals(T=100, R=50, rng=np.random.default_rng(42))
        assert sims.shape == (100, 3, 50)

    def test_mixture_per_arm_weights(self):
        """Test MixtureDistribution with different weights per arm."""
        components = [
            GaussianRewards(means=[1.0, 2.0, 3.0], variances=1.0),
            GaussianRewards(means=[2.0, 3.0, 4.0], variances=1.0),
        ]
        weights = np.array(
            [
                [0.9, 0.1],  # Arm 0: 90% component 1, 10% component 2
                [0.5, 0.5],  # Arm 1: 50% component 1, 50% component 2
                [0.2, 0.8],  # Arm 2: 20% component 1, 80% component 2
            ]
        )
        dist = MixtureDistribution(components=components, weights=weights)

        assert dist.K == 3
        assert dist.n_components == 2
        assert np.array_equal(dist.weights, weights)

        # Test counterfactuals generation
        sims = dist.generate_counterfactuals(T=100, R=50, rng=np.random.default_rng(42))
        assert sims.shape == (100, 3, 50)

    def test_mixture_invalid_empty_components(self):
        """Test MixtureDistribution rejects empty components list."""
        with pytest.raises(ValueError, match="components cannot be empty"):
            MixtureDistribution(components=[], weights=[])

    def test_mixture_invalid_mismatched_k(self):
        """Test MixtureDistribution rejects components with different K values."""
        components = [
            GaussianRewards(means=[1.0, 2.0], variances=1.0),  # K=2
            GaussianRewards(means=[1.0, 2.0, 3.0], variances=1.0),  # K=3
        ]
        with pytest.raises(ValueError, match="All components must have same K"):
            MixtureDistribution(components=components, weights=[0.5, 0.5])

    def test_mixture_invalid_weights_length(self):
        """Test MixtureDistribution rejects weights with wrong length."""
        components = [
            GaussianRewards(means=[1.0, 2.0], variances=1.0),
            GaussianRewards(means=[2.0, 3.0], variances=1.0),
        ]
        # 3 weights for 2 components
        with pytest.raises(ValueError, match="Length of weights .* does not match"):
            MixtureDistribution(components=components, weights=[0.5, 0.3, 0.2])

    def test_mixture_invalid_weights_shape(self):
        """Test MixtureDistribution rejects weights with wrong 2D shape."""
        components = [
            GaussianRewards(means=[1.0, 2.0], variances=1.0),
            GaussianRewards(means=[2.0, 3.0], variances=1.0),
        ]
        # Wrong shape: (3, 2) instead of (2, 2)
        with pytest.raises(ValueError, match="Weights shape .* does not match"):
            MixtureDistribution(
                components=components,
                weights=np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
            )

    def test_mixture_invalid_negative_weights(self):
        """Test MixtureDistribution rejects negative weights."""
        components = [
            GaussianRewards(means=[1.0, 2.0], variances=1.0),
            GaussianRewards(means=[2.0, 3.0], variances=1.0),
        ]
        with pytest.raises(ValueError, match="All weights must be non-negative"):
            MixtureDistribution(components=components, weights=[0.5, -0.5])

    def test_mixture_invalid_weights_sum(self):
        """Test MixtureDistribution rejects weights that don't sum to 1."""
        components = [
            GaussianRewards(means=[1.0, 2.0], variances=1.0),
            GaussianRewards(means=[2.0, 3.0], variances=1.0),
        ]
        with pytest.raises(ValueError, match="Weights must sum to 1"):
            MixtureDistribution(components=components, weights=[0.5, 0.3])

    def test_mixture_invalid_3d_weights(self):
        """Test MixtureDistribution rejects 3D weights array."""
        components = [
            GaussianRewards(means=[1.0, 2.0], variances=1.0),
            GaussianRewards(means=[2.0, 3.0], variances=1.0),
        ]
        with pytest.raises(ValueError, match="Weights must be 1D or 2D array"):
            MixtureDistribution(components=components, weights=np.zeros((2, 2, 2)))

    def test_mixture_statistical_properties_broadcast(self):
        """Test mixture produces expected mean (broadcast weights)."""
        # Two Gaussians with different means, equal weight
        components = [
            GaussianRewards(means=[1.0], variances=0.1),
            GaussianRewards(means=[3.0], variances=0.1),
        ]
        weights = np.array([0.5, 0.5])
        dist = MixtureDistribution(components=components, weights=weights)
        rng = np.random.default_rng(42)

        T = 100_000
        cf = dist.generate_counterfactuals(T=T, R=1, rng=rng)
        arm_data = cf[:, 0, 0]

        # Expected mean of mixture: 0.5 * 1.0 + 0.5 * 3.0 = 2.0
        expected_mean = 2.0
        sample_mean = arm_data.mean()
        se = arm_data.std() / np.sqrt(T)
        z_norm = stats.norm.ppf(0.995)

        assert abs(sample_mean - expected_mean) < z_norm * se, (
            f"Mixture mean {sample_mean:.4f} outside CI for expected {expected_mean}"
        )

    def test_mixture_statistical_properties_per_arm(self):
        """Test mixture produces expected means with per-arm weights."""
        # Two Bernoulli with different probs
        components = [
            BernoulliRewards(probs=[0.2, 0.3]),
            BernoulliRewards(probs=[0.8, 0.9]),
        ]
        # Arm 0: 100% component 0, Arm 1: 100% component 1
        weights = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        dist = MixtureDistribution(components=components, weights=weights)
        rng = np.random.default_rng(42)

        T = 100_000
        cf = dist.generate_counterfactuals(T=T, R=1, rng=rng)

        # Arm 0 should match component 0 (prob=0.2)
        arm0_data = cf[:, 0, 0]
        expected_prob0 = 0.2
        assert abs(arm0_data.mean() - expected_prob0) < 0.01

        # Arm 1 should match component 1 (prob=0.9)
        arm1_data = cf[:, 1, 0]
        expected_prob1 = 0.9
        assert abs(arm1_data.mean() - expected_prob1) < 0.01

    def test_mixture_sample_online(self):
        """Test sample_online method for mixture distribution."""
        components = [
            GaussianRewards(means=[1.0, 2.0], variances=0.1),
            GaussianRewards(means=[3.0, 4.0], variances=0.1),
        ]
        weights = np.array([0.5, 0.5])
        dist = MixtureDistribution(components=components, weights=weights)
        rng = np.random.default_rng(42)

        n_samples = 100_000
        for k in range(2):
            samples = np.array([dist.sample_online(k, rng) for _ in range(n_samples)])

            # Verify samples are floats
            assert isinstance(samples[0], (float, np.floating))

            # Expected mean: 0.5 * means[k] from comp0 + 0.5 * means[k] from comp1
            expected_mean = 0.5 * components[0].means[k] + 0.5 * components[1].means[k]
            sample_mean = samples.mean()
            se = samples.std() / np.sqrt(n_samples)
            z_norm = stats.norm.ppf(0.995)

            assert abs(sample_mean - expected_mean) < z_norm * se, (
                f"Arm {k} sample_online mean {sample_mean:.4f} outside CI for {expected_mean}"
            )

    def test_mixture_with_bernoulli_components(self):
        """Test mixture with Bernoulli components."""
        components = [
            BernoulliRewards(probs=[0.1, 0.3, 0.5]),
            BernoulliRewards(probs=[0.9, 0.7, 0.5]),
        ]
        weights = np.array([0.5, 0.5])
        dist = MixtureDistribution(components=components, weights=weights)

        assert dist.K == 3
        assert dist.n_components == 2

        sims = dist.generate_counterfactuals(T=100, R=50, rng=np.random.default_rng(42))
        assert sims.shape == (100, 3, 50)
        # All values should still be 0 or 1 (Bernoulli)
        assert np.all((sims == 0) | (sims == 1))

    def test_mixture_weights_near_boundary(self):
        """Test mixture with weights very close to 0/1."""
        components = [
            GaussianRewards([1.0], variances=1.0),
            GaussianRewards([10.0], variances=1.0),
        ]
        # Component 0 has 99.99% weight
        weights = np.array([1.0 - 1e-4, 1e-4])
        dist = MixtureDistribution(components=components, weights=weights)

        sims = dist.generate_counterfactuals(
            T=10000, R=1, rng=np.random.default_rng(42)
        )
        # Mean should be very close to component 0's mean
        assert np.abs(sims.mean() - 1.0) < 0.1


class TestPerArmDistribution:
    def test_per_arm_basic_homogeneous(self):
        """Test basic PerArmDistribution with same distribution type."""
        arm_dists = [
            GaussianRewards(means=[1.0], variances=1.0),
            GaussianRewards(means=[2.0], variances=2.0),
            GaussianRewards(means=[3.0], variances=0.5),
        ]
        dist = PerArmDistribution(arm_distributions=arm_dists)

        assert dist.K == 3
        assert len(dist.arm_distributions) == 3

        # Test counterfactuals generation
        sims = dist.generate_counterfactuals(T=100, R=50, rng=np.random.default_rng(42))
        assert sims.shape == (100, 3, 50)

    def test_per_arm_heterogeneous(self):
        """Test PerArmDistribution with different distribution types per arm."""
        arm_dists = [
            BernoulliRewards(probs=[0.3]),
            GaussianRewards(means=[0.5], variances=1.0),
            GaussianRewards(means=[0.8], variances=2.0),
        ]
        dist = PerArmDistribution(arm_distributions=arm_dists)

        assert dist.K == 3

        # Test counterfactuals generation
        sims = dist.generate_counterfactuals(T=100, R=50, rng=np.random.default_rng(42))
        assert sims.shape == (100, 3, 50)

        # Arm 0 should be Bernoulli (0 or 1)
        assert np.all((sims[:, 0, :] == 0) | (sims[:, 0, :] == 1))

    def test_per_arm_invalid_empty(self):
        """Test PerArmDistribution rejects empty distributions list."""
        with pytest.raises(ValueError, match="arm_distributions cannot be empty"):
            PerArmDistribution(arm_distributions=[])

    def test_per_arm_single_arm(self):
        """Test PerArmDistribution with single arm."""
        arm_dists = [GaussianRewards(means=[1.0], variances=1.0)]
        dist = PerArmDistribution(arm_distributions=arm_dists)

        assert dist.K == 1

        sims = dist.generate_counterfactuals(T=100, R=50, rng=np.random.default_rng(42))
        assert sims.shape == (100, 1, 50)

    def test_per_arm_statistical_properties(self):
        """Test PerArmDistribution preserves individual arm distributions."""
        arm_dists = [
            GaussianRewards(means=[1.0], variances=0.1),
            GaussianRewards(means=[2.0], variances=0.1),
            BernoulliRewards(probs=[0.8]),
        ]
        dist = PerArmDistribution(arm_distributions=arm_dists)
        rng = np.random.default_rng(42)

        T = 50_000
        cf = dist.generate_counterfactuals(T=T, R=1, rng=rng)

        # Check arm 0: Gaussian with mean=1.0
        arm0_data = cf[:, 0, 0]
        assert abs(arm0_data.mean() - 1.0) < 0.02

        # Check arm 1: Gaussian with mean=2.0
        arm1_data = cf[:, 1, 0]
        assert abs(arm1_data.mean() - 2.0) < 0.02

        # Check arm 2: Bernoulli with prob=0.8
        arm2_data = cf[:, 2, 0]
        assert np.all((arm2_data == 0) | (arm2_data == 1))
        assert abs(arm2_data.mean() - 0.8) < 0.01

    def test_per_arm_sample_online(self):
        """Test sample_online method for PerArmDistribution."""
        arm_dists = [
            GaussianRewards(means=[1.0], variances=0.1),
            BernoulliRewards(probs=[0.7]),
            GaussianRewards(means=[3.0], variances=0.1),
        ]
        dist = PerArmDistribution(arm_distributions=arm_dists)
        rng = np.random.default_rng(42)

        n_samples = 50_000

        # Test arm 0: Gaussian
        samples_0 = np.array([dist.sample_online(0, rng) for _ in range(n_samples)])
        assert isinstance(samples_0[0], (float, np.floating))
        assert abs(samples_0.mean() - 1.0) < 0.02

        # Test arm 1: Bernoulli
        samples_1 = np.array([dist.sample_online(1, rng) for _ in range(n_samples)])
        assert isinstance(samples_1[0], (float, np.floating))
        assert np.all((samples_1 == 0) | (samples_1 == 1))
        assert abs(samples_1.mean() - 0.7) < 0.01

        # Test arm 2: Gaussian
        samples_2 = np.array([dist.sample_online(2, rng) for _ in range(n_samples)])
        assert isinstance(samples_2[0], (float, np.floating))
        assert abs(samples_2.mean() - 3.0) < 0.02

    def test_per_arm_with_mixture_component(self):
        """Test PerArmDistribution with a mixture as one of the arms."""
        # Arm 0: Simple Gaussian
        # Arm 1: Mixture of two Gaussians
        mixture = MixtureDistribution(
            components=[
                GaussianRewards(means=[1.0], variances=0.1),
                GaussianRewards(means=[3.0], variances=0.1),
            ],
            weights=[0.5, 0.5],
        )

        arm_dists = [
            GaussianRewards(means=[0.5], variances=0.1),
            mixture,
        ]
        dist = PerArmDistribution(arm_distributions=arm_dists)

        assert dist.K == 2

        sims = dist.generate_counterfactuals(T=100, R=50, rng=np.random.default_rng(42))
        assert sims.shape == (100, 2, 50)

        # Test that arm 1 has the mixture mean (approximately 2.0)
        rng = np.random.default_rng(42)
        T = 50_000
        cf = dist.generate_counterfactuals(T=T, R=1, rng=rng)
        arm1_data = cf[:, 1, 0]
        expected_mixture_mean = 2.0  # 0.5 * 1.0 + 0.5 * 3.0
        assert abs(arm1_data.mean() - expected_mixture_mean) < 0.05

    def test_per_arm_all_bernoulli(self):
        """Test PerArmDistribution with all Bernoulli arms."""
        arm_dists = [
            BernoulliRewards(probs=[0.2]),
            BernoulliRewards(probs=[0.5]),
            BernoulliRewards(probs=[0.8]),
        ]
        dist = PerArmDistribution(arm_distributions=arm_dists)

        assert dist.K == 3

        sims = dist.generate_counterfactuals(
            T=1000, R=10, rng=np.random.default_rng(42)
        )
        assert sims.shape == (1000, 3, 10)

        # All arms should be binary
        assert np.all((sims == 0) | (sims == 1))

    def test_per_arm_many_arms(self):
        """Test PerArmDistribution with many arms (10 arms)."""
        arm_dists = [
            GaussianRewards(means=[float(i)], variances=1.0) for i in range(10)
        ]
        dist = PerArmDistribution(arm_distributions=arm_dists)

        assert dist.K == 10

        sims = dist.generate_counterfactuals(T=100, R=5, rng=np.random.default_rng(42))
        assert sims.shape == (100, 10, 5)


class TestStudentTRewards:
    @pytest.mark.parametrize(
        "means,df,expected_df",
        [
            # Scalar df (broadcast)
            (np.array([0.3, 0.5, 0.8]), 3.0, np.array([3.0, 3.0, 3.0])),
            # Per-arm df
            (
                np.array([0.3, 0.5, 0.8]),
                np.array([3.0, 5.0, 10.0]),
                np.array([3.0, 5.0, 10.0]),
            ),
        ],
    )
    def test_student_t_rewards_basic(self, means, df, expected_df):
        """Test basic StudentTRewards functionality with scalar and per-arm df."""
        dist = StudentTRewards(means=means, df=df)
        sims = dist.generate_counterfactuals(T=100, R=50, rng=np.random.default_rng(42))

        assert dist.K == 3
        assert sims.shape == (100, 3, 50)
        assert np.array_equal(dist.means, means)
        assert np.array_equal(dist.df, expected_df)

    def test_student_t_rewards_invalid_df_length(self):
        """Test StudentTRewards rejects df array with wrong length."""
        with pytest.raises(
            ValueError, match="Length df .* does not match means length"
        ):
            StudentTRewards(means=[1.0, 2.0, 3.0], df=[3.0, 5.0])

    def test_student_t_rewards_invalid_negative_df(self):
        """Test StudentTRewards rejects negative df."""
        with pytest.raises(
            ValueError, match="All degrees of freedom must be strictly positive"
        ):
            StudentTRewards(means=[1.0, 2.0], df=-1.0)

    def test_student_t_rewards_invalid_zero_df(self):
        """Test StudentTRewards rejects zero df."""
        with pytest.raises(
            ValueError, match="All degrees of freedom must be strictly positive"
        ):
            StudentTRewards(means=[1.0, 2.0], df=0.0)

    def test_student_t_rewards_statistical_properties(self):
        """Test StudentTRewards produces samples with correct mean."""
        means = np.array([1.0, 2.0, 3.0])
        df = np.array([5.0, 10.0, 20.0])  # Higher df for more stable estimates
        dist = StudentTRewards(means=means, df=df)
        rng = np.random.default_rng(42)

        T = 100_000
        cf = dist.generate_counterfactuals(T=T, R=1, rng=rng)

        for k in range(len(means)):
            arm_data = cf[:, k, 0]

            # Test mean (Student's t with df > 1 has mean = location parameter)
            sample_mean = arm_data.mean()
            se = arm_data.std() / np.sqrt(T)
            z_norm = stats.norm.ppf(0.995)  # 99% CI

            assert abs(sample_mean - means[k]) < z_norm * se, (
                f"Arm {k} mean {sample_mean:.4f} outside CI for expected {means[k]}"
            )

    def test_student_t_rewards_heavy_tails(self):
        """Test StudentTRewards with low df produces heavier tails than Gaussian."""
        # Compare Student's t (df=3) vs Gaussian with same location/scale
        mean = 0.0
        df = 3.0

        dist_t = StudentTRewards(means=[mean], df=df)
        dist_gauss = GaussianRewards(
            means=[mean], variances=df / (df - 2)
        )  # Match variance

        rng_t = np.random.default_rng(42)
        rng_gauss = np.random.default_rng(42)

        T = 50_000
        cf_t = dist_t.generate_counterfactuals(T=T, R=1, rng=rng_t)
        cf_gauss = dist_gauss.generate_counterfactuals(T=T, R=1, rng=rng_gauss)

        data_t = cf_t[:, 0, 0]
        data_gauss = cf_gauss[:, 0, 0]

        # Student's t should have higher kurtosis (heavier tails)
        kurtosis_t = stats.kurtosis(data_t)
        kurtosis_gauss = stats.kurtosis(data_gauss)

        assert kurtosis_t > kurtosis_gauss, (
            f"Student's t kurtosis {kurtosis_t:.4f} should be > "
            f"Gaussian kurtosis {kurtosis_gauss:.4f}"
        )

    def test_student_t_rewards_sample_online(self):
        """Test sample_online method returns correct distribution."""
        means = np.array([1.0, 2.0, 3.0])
        df = np.array([10.0, 20.0, 30.0])  # Higher df for stable tests
        dist = StudentTRewards(means=means, df=df)
        rng = np.random.default_rng(42)

        n_samples = 100_000
        for k in range(len(means)):
            samples = np.array([dist.sample_online(k, rng) for _ in range(n_samples)])

            # Verify samples are floats
            assert isinstance(samples[0], (float, np.floating))

            # Test mean
            sample_mean = samples.mean()
            se = samples.std() / np.sqrt(n_samples)
            z_norm = stats.norm.ppf(0.995)

            assert abs(sample_mean - means[k]) < z_norm * se, (
                f"Arm {k} sample_online mean {sample_mean:.4f} outside CI for {means[k]}"
            )

    def test_student_t_rewards_different_df_per_arm(self):
        """Test StudentTRewards with very different df values per arm."""
        means = np.array([0.0, 0.0])
        df = np.array([2.5, 30.0])  # Very heavy tails vs nearly Gaussian

        dist = StudentTRewards(means=means, df=df)
        rng = np.random.default_rng(42)

        T = 50_000
        cf = dist.generate_counterfactuals(T=T, R=1, rng=rng)

        # Arm 0 (df=2.5) should have heavier tails than Arm 1 (df=30)
        data_0 = cf[:, 0, 0]
        data_1 = cf[:, 1, 0]

        kurtosis_0 = stats.kurtosis(data_0)
        kurtosis_1 = stats.kurtosis(data_1)

        assert kurtosis_0 > kurtosis_1, (
            f"Arm 0 (df=2.5) kurtosis {kurtosis_0:.4f} should be > "
            f"Arm 1 (df=30) kurtosis {kurtosis_1:.4f}"
        )

    def test_student_t_rewards_high_df_approaches_gaussian(self):
        """Test that high df Student's t approaches Gaussian distribution."""
        mean = 1.0
        df = 100.0  # Very high df should be nearly Gaussian

        dist = StudentTRewards(means=[mean], df=df)
        rng = np.random.default_rng(42)

        T = 50_000
        cf = dist.generate_counterfactuals(T=T, R=1, rng=rng)
        data = cf[:, 0, 0]

        # High df Student's t should pass normality test
        _, p_value = stats.jarque_bera(data)
        assert p_value > 0.01, (
            f"High df Student's t should be close to normal (p={p_value:.4f})"
        )

    def test_student_t_rewards_variance_formula(self):
        """Test Student's t variance formula: var = df/(df-2) for df > 2."""
        mean = 0.0
        df = 10.0  # df > 2 so variance exists

        dist = StudentTRewards(means=[mean], df=df)
        rng = np.random.default_rng(42)

        T = 100_000
        cf = dist.generate_counterfactuals(T=T, R=1, rng=rng)
        data = cf[:, 0, 0]

        # Expected variance for standard Student's t
        expected_var = df / (df - 2)
        sample_var = data.var(ddof=1)

        # Allow 5% tolerance due to sampling variability
        assert abs(sample_var - expected_var) / expected_var < 0.05, (
            f"Sample variance {sample_var:.4f} should be close to "
            f"theoretical {expected_var:.4f}"
        )
