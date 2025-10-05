"""Reward distribution models for bandit simulations.

This module supports:
- Simulation of counterfactual outcomes (T x K x R arrays) for Monte Carlo studies
- Online/real-data mode with on-demand reward sampling
"""

from abc import ABC, abstractmethod

import numpy as np

_FLOAT_TOL = 1e-10


class RewardDistribution(ABC):
    """Abstract base class for reward distribution models.

    Subclasses must implement `generate_counterfactuals` for simulation mode.
    Optionally override `sample_online` for efficient real-data sampling.
    """

    @abstractmethod
    def generate_counterfactuals(
        self, T: int, R: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate counterfactual outcomes for all (time, arm, replication) tuples.

        Parameters
        ----------
        T : int
            Number of time periods.
        R : int
            Number of Monte Carlo replications.
        rng : np.random.Generator
            Random number generator for reproducibility.

        Returns
        -------
        np.ndarray, shape (T, K, R)
            Counterfactual outcomes. Element [t, k, r] is the reward that would be
            observed if arm k were selected at time t in replication r.
            K is determined by the distribution's initialization.
        """

    @abstractmethod
    def sample_online(self, arm: int, rng: np.random.Generator) -> float:
        """Sample a reward for online/real-data mode (optional override).

        Default implementation generates counterfactuals and extracts the needed value
        (wasteful). Subclasses should override for efficiency in online mode.

        Parameters
        ----------
        arm : int
            Arm index (0-indexed).
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        float
            Sampled reward for the given arm.
        """


class GaussianRewards(RewardDistribution):
    """Gaussian reward distributions with per-arm means and variances.

    Parameters
    ----------
    means : np.ndarray, shape (K,)
        Mean reward for each arm.
    variances : np.ndarray, shape (K,), or float, default=1.0
        Variance for each arm. If float, same variance used for all arms.
        All variances must be strictly positive.

    Examples
    --------
    >>> dist = GaussianRewards(means=[0.3, 0.5, 0.8], variances=1.0)
    >>> cf = dist.generate_counterfactuals(T=10, R=5, rng=np.random.default_rng(42))
    >>> cf.shape
    (10, 3, 5)

    >>> # Different variance per arm
    >>> dist = GaussianRewards(means=[0.5, 0.8], variances=[1.0, 2.0])
    """

    def __init__(
        self,
        means: np.ndarray,
        variances: np.ndarray | float = 1.0,
    ):
        """Initialize GaussianRewards distribution."""
        self.means = np.asarray(means, dtype=float)
        self.K = len(self.means)

        if np.isscalar(variances):
            self.variances = np.full(self.K, float(variances), dtype=float)
        else:
            self.variances = np.asarray(variances, dtype=float)
            if len(self.variances) != self.K:
                raise ValueError(
                    f"Length variances {len(self.variances)} does not match means length {self.K}"
                )

        if np.any(self.variances <= _FLOAT_TOL):
            raise ValueError("All variances must be strictly positive")

    def generate_counterfactuals(
        self, T: int, R: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate Gaussian counterfactuals."""
        stds = np.sqrt(self.variances)
        counterfactuals = rng.normal(
            self.means[None, :, None], stds[None, :, None], size=(T, self.K, R)
        )

        return counterfactuals

    def sample_online(self, arm: int, rng: np.random.Generator) -> float:
        """Efficiently sample a single reward (no waste)."""
        return rng.normal(self.means[arm], np.sqrt(self.variances[arm]))

    def __repr__(self) -> str:
        """Return string representation."""
        return f"GaussianRewards(means={self.means}, variances={self.variances})"


class BernoulliRewards(RewardDistribution):
    """Bernoulli reward distributions (binary outcomes: 0 or 1).

    Parameters
    ----------
    probs : np.ndarray, shape (K,)
        Success probability for each arm. Each element must be in [0, 1].

    Examples
    --------
    >>> dist = BernoulliRewards(probs=[0.2, 0.5, 0.8])
    >>> cf = dist.generate_counterfactuals(T=10, R=5, rng=np.random.default_rng(42))
    """

    def __init__(self, probs: np.ndarray):
        """Initialize BernoulliRewards distribution."""
        self.probs = np.asarray(probs, dtype=float)
        self.K = len(self.probs)
        if not np.all((self.probs >= 0.0) & (self.probs <= 1.0)):
            raise ValueError("All probabilities must be in [0, 1]")

    def generate_counterfactuals(
        self, T: int, R: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate Bernoulli counterfactuals."""
        counterfactuals = rng.binomial(
            1, self.probs[None, :, None], size=(T, self.K, R)
        ).astype(float)
        return counterfactuals

    def sample_online(self, arm: int, rng: np.random.Generator) -> float:
        """Efficiently sample a single Bernoulli reward."""
        return float(rng.binomial(1, self.probs[arm]))

    def __repr__(self) -> str:
        """Return string representation."""
        return f"BernoulliRewards(probs={self.probs})"


class StudentTRewards(RewardDistribution):
    """Student's t-distributed rewards with per-arm parameters.

    Useful for studying robustness to heavy-tailed reward distributions.
    Each arm has location parameter (mean) and degrees of freedom.

    Parameters
    ----------
    means : np.ndarray, shape (K,)
        Location parameter (mean) for each arm.
    df : np.ndarray, shape (K,), or float, default=3.0
        Degrees of freedom for each arm. If float, same df used for all arms.
        Must be strictly positive. Lower values give heavier tails.

    Examples
    --------
    >>> dist = StudentTRewards(means=[0.3, 0.5, 0.8], df=3.0)
    >>> cf = dist.generate_counterfactuals(T=10, R=5, rng=np.random.default_rng(42))

    >>> # Different df per arm
    >>> dist = StudentTRewards(means=[0.5, 0.8], df=[3.0, 5.0])
    """

    def __init__(
        self,
        means: np.ndarray,
        df: np.ndarray | float = 3.0,
    ):
        """Initialize StudentTRewards distribution."""
        self.means = np.asarray(means, dtype=float)
        self.K = len(self.means)

        if np.isscalar(df):
            self.df = np.full(self.K, float(df), dtype=float)
        else:
            self.df = np.asarray(df, dtype=float)
            if len(self.df) != self.K:
                raise ValueError(
                    f"Length df {len(self.df)} does not match means length {self.K}"
                )

        if np.any(self.df <= _FLOAT_TOL):
            raise ValueError("All degrees of freedom must be strictly positive")

    def generate_counterfactuals(
        self, T: int, R: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate Student's t counterfactuals."""
        counterfactuals = np.zeros((T, self.K, R))
        for k in range(self.K):
            counterfactuals[:, k, :] = self.means[k] + rng.standard_t(
                self.df[k], size=(T, R)
            )
        return counterfactuals

    def sample_online(self, arm: int, rng: np.random.Generator) -> float:
        """Efficiently sample a single Student's t reward."""
        return self.means[arm] + rng.standard_t(self.df[arm])

    def __repr__(self) -> str:
        """Return string representation."""
        return f"StudentTRewards(means={self.means}, df={self.df})"


class MixtureDistribution(RewardDistribution):
    """Discrete mixture of reward distributions.

    At each sample, randomly selects one component based on weights,
    then samples from that component distribution.

    Parameters
    ----------
    components : list of RewardDistribution
        Component distributions. All must have the same K.
    weights : np.ndarray, shape (n_components,) or (K, n_components)
        Mixing weights. If 1D, same weights for all arms (broadcast).
        If 2D, per-arm weights. Each row must sum to 1.

    Examples
    --------
    >>> # Same mixture for all 3 arms (scalar broadcast)
    >>> mixture = MixtureDistribution(
    ...     components=[
    ...         GaussianRewards([0.3, 0.5, 0.8], variances=1.0),
    ...         StudentTRewards([0.3, 0.5, 0.8], df=2.0)
    ...     ],
    ...     weights=[0.9, 0.1]  # 90% Gaussian, 10% StudentT for all arms
    ... )

    >>> # Different mixture per arm (per-arm weights)
    >>> mixture = MixtureDistribution(
    ...     components=[
    ...         GaussianRewards([0.3, 0.5, 0.8], variances=1.0),
    ...         StudentTRewards([0.3, 0.5, 0.8], df=2.0)
    ...     ],
    ...     weights=np.array([
    ...         [0.95, 0.05],  # Arm 0: 95% Gaussian, 5% StudentT
    ...         [0.80, 0.20],  # Arm 1: 80% Gaussian, 20% StudentT
    ...         [0.50, 0.50],  # Arm 2: 50% Gaussian, 50% StudentT
    ...     ])
    ... )

    >>> # Single arm for use with PerArmDistribution
    >>> mixture_arm = MixtureDistribution(
    ...     components=[
    ...         GaussianRewards([0.5], variances=1.0),
    ...         StudentTRewards([0.5], df=2.0)
    ...     ],
    ...     weights=[0.9, 0.1]
    ... )
    >>> dist = PerArmDistribution([
    ...     mixture_arm,
    ...     GaussianRewards([0.8], variances=1.0)
    ... ])
    """

    def __init__(self, components: list, weights: np.ndarray):
        """Initialize MixtureDistribution."""
        if not components:
            raise ValueError("components cannot be empty")

        self.components = components
        self.n_components = len(components)

        # Validate all components have same K
        K_values = [comp.K for comp in components]
        if len(set(K_values)) > 1:
            raise ValueError(f"All components must have same K, got {K_values}")
        self.K = K_values[0]

        # Handle weights: 1D (broadcast) or 2D (per-arm)
        weights = np.asarray(weights, dtype=float)
        if weights.ndim == 1:
            # Broadcast to all arms
            if len(weights) != self.n_components:
                raise ValueError(
                    f"Length of weights {len(weights)} does not match "
                    f"number of components {self.n_components}"
                )
            self.weights = np.tile(weights, (self.K, 1))  # (K, n_components)
        elif weights.ndim == 2:
            # Per-arm weights
            if weights.shape != (self.K, self.n_components):
                raise ValueError(
                    f"Weights shape {weights.shape} does not match "
                    f"expected shape ({self.K}, {self.n_components})"
                )
            self.weights = weights
        else:
            raise ValueError(f"Weights must be 1D or 2D array, got {weights.ndim}D")

        # Validate weights: non-negative and sum to 1 per arm
        if np.any(self.weights < 0.0):
            raise ValueError("All weights must be non-negative")

        row_sums = self.weights.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=_FLOAT_TOL):
            raise ValueError(
                f"Weights must sum to 1 for each arm, got sums: {row_sums}"
            )

    def generate_counterfactuals(
        self, T: int, R: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate mixture counterfactuals."""
        # Generate counterfactuals from all components
        all_counterfactuals = np.stack(
            [comp.generate_counterfactuals(T, R, rng) for comp in self.components],
            axis=0,
        )  # (n_components, T, K, R)

        # Generate component selections per arm
        counterfactuals = np.zeros((T, self.K, R))
        for k in range(self.K):
            # Select components for arm k at all (t, r)
            selections = rng.choice(self.n_components, size=(T, R), p=self.weights[k])
            # Extract selected components
            for t in range(T):
                for r in range(R):
                    counterfactuals[t, k, r] = all_counterfactuals[
                        selections[t, r], t, k, r
                    ]

        return counterfactuals

    def sample_online(self, arm: int, rng: np.random.Generator) -> float:
        """Sample from mixture for online mode."""
        # Select component based on weights for this arm
        component_idx = rng.choice(self.n_components, p=self.weights[arm])
        # Sample from selected component
        return self.components[component_idx].sample_online(arm, rng)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MixtureDistribution(K={self.K}, n_components={self.n_components}, "
            f"components={self.components})"
        )


class PerArmDistribution(RewardDistribution):
    """Heterogeneous distributions per arm.

    Allows each arm to have a completely different reward distribution
    (e.g., arm 0: Bernoulli, arm 1: Gaussian, arm 2: Student's t).

    Parameters
    ----------
    arm_distributions : list of RewardDistribution
        One distribution object per arm. Length must equal K.

    Examples
    --------
    >>> arm_dists = [
    ...     BernoulliRewards([0.3]),
    ...     GaussianRewards([0.5], variance=1.0),
    ...     GaussianRewards([0.8], variance=2.0),
    ... ]
    >>> dist = PerArmDistribution(arm_dists)
    >>> cf = dist.generate_counterfactuals(T=10, K=3, R=5, rng=np.random.default_rng(42))
    """

    def __init__(self, arm_distributions: list):
        """Initialize PerArmDistribution."""
        if not arm_distributions:
            raise ValueError("arm_distributions cannot be empty")
        self.arm_distributions = arm_distributions
        self.K = len(arm_distributions)

    def generate_counterfactuals(
        self, T: int, R: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate heterogeneous counterfactuals per arm."""
        counterfactuals = np.zeros((T, self.K, R))
        for k, dist in enumerate(self.arm_distributions):
            # Generate T×R for this arm and extract
            arm_data = dist.generate_counterfactuals(T, R, rng)
            counterfactuals[:, k, :] = arm_data[:, 0, :]

        return counterfactuals

    def sample_online(self, arm: int, rng: np.random.Generator) -> float:
        """Sample from the specific arm's distribution."""
        return self.arm_distributions[arm].sample_online(0, rng)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PerArmDistribution(K={self.K}, distributions={self.arm_distributions})"
