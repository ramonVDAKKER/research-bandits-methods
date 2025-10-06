"""Markovian bandit policies for multi-armed bandit problems.

This module implements bandit algorithms that use Markovian state
(sufficient statistics), via cumulative rewards and cumulative arm pulls, to decide
upon the arm. The implementation supports vectorised parallel Monte Carlo simulations.

Mathematical Background
-----------------------
In the K-armed bandit problem, an agent repeatedly selects one of K arms (actions)
over T rounds. Each arm k has an unknown mean reward μ_k. The goal is to maximise
cumulative reward by learning which arms are best while balancing exploration
and exploitation.

All policies in this module:
- Maintain sufficient statistics (counts, cumulative rewards, means)
- Support vectorised operations across R parallel independent runs
- Perform forced exploration for the first K rounds (one pull per arm)

Policies
--------
EpsilonGreedy
    Explores uniformly with probability ε, exploits best empirical arm otherwise.
    Simple baseline with tunable exploration rate.

UCB
    Upper Confidence Bound (UCB1) algorithm. Optimistically estimates arm values
    using confidence intervals. Achieves O(log T) regret for bounded rewards.

GaussianThompson
    Bayesian Thompson sampling assuming Gaussian rewards with known variance.
    Samples arm means from posterior distribution and selects optimistically.

Complexity
----------
Time: O(K) per round for all policies (arm selection and update)
Space: O(KR) for maintaining statistics across R parallel runs

References
----------
.. [1] Lattimore, T., & Szepesvári, C. (2020). Bandit algorithms.
       Cambridge University Press. https://tor-lattimore.com/downloads/book/book.pdf
.. [2] Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis
       of the multiarmed bandit problem. Machine Learning, 47(2-3), 235-256.
.. [3] Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning:
       An introduction (2nd ed.). MIT Press.
.. [4] Thompson, W. R. (1933). On the likelihood that one unknown probability
       exceeds another in view of the evidence of two samples. Biometrika, 25(3/4), 285-294.

Examples
--------
>>> import numpy as np
>>> rng = np.random.default_rng(42)
>>>
>>> # ε-greedy with 10% exploration
>>> policy = EpsilonGreedy(K=5, R=100, epsilon=0.1, rng=rng)
>>>
>>> # UCB with exploration constant c=2
>>> policy = UCB(K=5, R=100, c=2.0)
>>>
>>> # Thompson sampling with informative prior
>>> policy = GaussianThompson(K=5, R=100, prior_mean=0.5, prior_var=0.25, rng=rng)
"""

from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
from loguru import logger

from ...constants import FLOAT_TOL

# Numerical constants for stability
EPSILON_SAFE_DIV = 1e-10  # Floor for division to prevent overflow
MAX_UCB_BONUS = 1e10  # Maximum UCB exploration bonus
MIN_POSTERIOR_VAR = 1e-10  # Minimum posterior variance for Thompson sampling


class MarkovianBanditPolicy(ABC):
    """Base class for Markovian bandit policies."""

    def __init__(
        self,
        K: int,
        R: int = 1,
        rng: np.random.Generator | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize Markovian bandit policy.

        Parameters
        ----------
        K : int
            Number of arms (must be >= 1).
        R : int, default=1
            Number of parallel runs (must be >= 1).
        rng : np.random.Generator | None, default=None
            Random number generator for reproducibility.

        Raises
        ------
        ValueError
            If K < 1 or R < 1.
        TypeError
            If rng is not None and not a np.random.Generator.
        """
        if K < 1:
            raise ValueError("The number of arms K must be >= 1")
        if R < 1:
            raise ValueError("The number of runs R must be >= 1")
        self.K = K
        self.R = R
        self.counts = np.zeros((K, R), dtype=np.int64)
        self.means = np.zeros((K, R), dtype=float)
        self.m2 = np.zeros((K, R), dtype=float)  # Sum of squared deviations (Welford)

        # Validate RNG type
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError(f"rng must be np.random.Generator, got {type(rng)}")
        self.rng = rng or np.random.default_rng()
        logger.debug(f"Initialized {self.__class__.__name__} with K={K}, R={R}")

    def _forced_exploration_arm(self, t: int) -> np.ndarray | None:
        """Return forced exploration arm if t <= K, else None.

        During the first K rounds, each arm is pulled exactly once to ensure
        all arms have at least one sample before applying the selection strategy.

        Parameters
        ----------
        t : int
            Current round (1-indexed).

        Returns
        -------
        np.ndarray | None
            Array of shape (R,) with arm index (t-1) if t <= K, else None.
        """
        if t <= self.K:
            return np.full(self.R, t - 1, dtype=int)
        return None

    @abstractmethod
    def select_arm(self, t: int) -> np.ndarray:
        """Choose an arm for each run at round t.

        Parameters
        ----------
        t : int
            Current round number (1-indexed: t=1 is the first round).

        Returns
        -------
        np.ndarray, shape (R,)
            Selected arm indices for each of the R runs. Each element is in {0,...,K - 1}.
        """

    def _validate_update_inputs(self, arm: np.ndarray, reward: np.ndarray) -> None:
        """Validate inputs to update method.

        Raises
        ------
        ValueError
            If inputs are invalid.
        """
        if arm.shape != (self.R,):
            raise ValueError(f"arm must have shape ({self.R},), got {arm.shape}")
        if reward.shape != (self.R,):
            raise ValueError(f"reward must have shape ({self.R},), got {reward.shape}")
        if not np.all((arm >= 0) & (arm < self.K)):
            raise ValueError(f"All arm indices must be in [0, {self.K})")
        if not np.all(np.isfinite(reward)):
            raise ValueError("Rewards must be finite (no NaN or Inf)")

    def update(self, arm: np.ndarray, reward: np.ndarray) -> None:
        """Incrementally update sufficient statistics after observing rewards.

        Uses Welford's online algorithm for numerically stable mean and variance computation.

        Notes
        -----
        Welford's algorithm computes the mean and M2 (sum of squared deviations) incrementally:
        - delta = x - mean_old
        - mean_new = mean_old + delta / n
        - delta2 = x - mean_new
        - M2_new = M2_old + delta * delta2
        The sample variance is then M2 / (n - 1).
        """
        self._validate_update_inputs(arm, reward)

        logger.debug(
            f"Update - Arms: {arm}, Rewards: {reward}, "
            f"Counts before: {self.counts[arm, np.arange(self.R)]}"
        )

        run_idx = np.arange(self.R, dtype=int)

        self.counts[arm, run_idx] += 1
        new_counts = self.counts[arm, run_idx]

        delta = reward - self.means[arm, run_idx]
        self.means[arm, run_idx] += delta / new_counts

        delta2 = reward - self.means[arm, run_idx]
        self.m2[arm, run_idx] += delta * delta2

    @property
    def cumulative_rewards(self) -> np.ndarray:
        """Get cumulative rewards for all arms (computed from means and counts).

        Returns
        -------
        np.ndarray, shape (K, R)
            Total rewards for each arm in each run.

        Notes
        -----
        This is computed on-demand from means and counts to avoid maintaining
        redundant state that could drift due to floating-point errors.
        """
        return self.means * self.counts

    @property
    def empirical_means(self) -> np.ndarray:
        """Alias for means.

        Returns
        -------
        np.ndarray, shape (K, R)
            Copy of empirical mean rewards for each arm.

        Notes
        -----
        Returns a defensive copy to prevent external mutation of internal state.
        """
        return self.means.copy()

    @property
    def pull_counts(self) -> np.ndarray:
        """Alias for counts.

        Returns
        -------
        np.ndarray, shape (K, R)
            Copy of pull counts for each arm.

        Notes
        -----
        Returns a defensive copy to prevent external mutation of internal state.
        """
        return self.counts.copy()


class EpsilonGreedy(MarkovianBanditPolicy):
    """ε-greedy policy: explore with probability ε, exploit otherwise.

    The ε-greedy policy selects the arm with the highest empirical mean with
    probability (1-ε), and selects a uniformly random arm with probability ε.
    The first K rounds perform forced exploration (one pull per arm).

    Parameters
    ----------
    K : int
        Number of arms.
    R : int, default=1
        Number of parallel runs.
    epsilon : float, default=0.1
        Exploration probability in (0, 1).
    rng : np.random.Generator | None, default=None
        Random number generator for reproducibility.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> policy = EpsilonGreedy(K=3, R=2, epsilon=0.1, rng=rng)
    >>> arms = policy.select_arm(t=1)  # Returns [0, 0] for forced exploration
    >>> policy.update(arms, np.array([0.5, 0.3]))
    """

    def __init__(
        self,
        K: int,
        R: int = 1,
        epsilon: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize ε-greedy policy.

        Parameters
        ----------
        K : int
            Number of arms.
        R : int, default=1
            Number of parallel runs.
        epsilon : float, default=0.1
            Exploration probability (must be in (0, 1)).
        rng : np.random.Generator | None, default=None
            Random number generator for reproducibility.

        Raises
        ------
        ValueError
            If epsilon not in (0, 1).
        """
        super().__init__(K, R, rng)
        if epsilon <= FLOAT_TOL or epsilon >= 1.0 - FLOAT_TOL:
            raise ValueError("epsilon must be in (0,1)")
        self.epsilon = float(epsilon)

    def select_arm(self, t: int) -> np.ndarray:
        """Select arms using ε-greedy strategy."""
        forced = self._forced_exploration_arm(t)
        if forced is not None:
            return forced

        greedy_arms = np.argmax(self.means, axis=0)
        explore = self.rng.random(self.R) < self.epsilon
        random_arms = self.rng.integers(0, self.K, size=self.R)
        selected = np.where(explore, random_arms, greedy_arms)

        return selected

    def __repr__(self) -> str:
        """Return string representation of policy."""
        return f"EpsilonGreedy(K={self.K}, R={self.R}, epsilon={self.epsilon})"


class UCB(MarkovianBanditPolicy):
    """Upper Confidence Bound (UCB1) policy.

    UCB1 selects arms based on an optimistic estimate of their value, balancing
    exploitation of high-reward arms with exploration of uncertain arms. The UCB
    index for each arm is: mean + c * sqrt(2 * log(t) / n_pulls).

    Parameters
    ----------
    K : int
        Number of arms.
    R : int, default=1
        Number of parallel runs.
    c : float, default=1.0
        Exploration constant (higher = more exploration). Must be > 0.
        Typical values: 1.0 (standard UCB1), 2.0 (more conservative).
    cache_size : int, default=1_000
        Maximum size of LRU cache for log(t) computations to improve performance.

    Raises
    ------
    ValueError
        If c <= 0.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> policy = UCB(K=3, R=2, c=2.0)
    >>> arms = policy.select_arm(t=1)  # Returns [0, 0] for forced exploration
    >>> policy.update(arms, np.array([0.5, 0.3]))

    References
    ----------
    .. [1] Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis
           of the multiarmed bandit problem. Machine Learning, 47(2-3), 235-256.
    """

    def __init__(
        self, K: int, R: int = 1, c: float = 1.0, cache_size: int = 1000
    ) -> None:
        """Initialize UCB1 policy.

        Parameters
        ----------
        K : int
            Number of arms.
        R : int, default=1
            Number of parallel runs.
        c : float, default=1.0
            Exploration constant (must be > 0).
        cache_size : int, default=1_000
            Maximum LRU cache size for log(t) values.

        Raises
        ------
        ValueError
            If c <= 0.
        """
        super().__init__(K, R)
        if c <= FLOAT_TOL:
            raise ValueError("c must be positive")
        self.c = float(c)
        self._log_cache: OrderedDict[int, float] = OrderedDict()
        self._cache_size = cache_size

    def select_arm(self, t: int) -> np.ndarray:
        """Select arms using UCB1 index with LRU-cached logarithms."""
        forced = self._forced_exploration_arm(t)
        if forced is not None:
            return forced

        # LRU cache with size limit
        if t not in self._log_cache:
            if len(self._log_cache) >= self._cache_size:
                self._log_cache.popitem(last=False)  # Remove oldest
            self._log_cache[t] = np.log(t)
        else:
            # Move to end (most recently used)
            self._log_cache.move_to_end(t)

        log_t = self._log_cache[t]

        safe_counts = np.maximum(self.counts, EPSILON_SAFE_DIV)
        uncertainty_loading = self.c * np.sqrt(2.0 * log_t / safe_counts)
        uncertainty_loading = np.minimum(uncertainty_loading, MAX_UCB_BONUS)

        ucb_values = self.means + uncertainty_loading
        return np.argmax(ucb_values, axis=0)

    def __repr__(self) -> str:
        """Return string representation of policy."""
        return f"UCB(K={self.K}, R={self.R}, c={self.c})"


class GaussianThompson(MarkovianBanditPolicy):
    """Thompson sampling with Gaussian prior for arm means.

    Thompson sampling is a Bayesian approach that maintains a posterior distribution
    over arm means and samples from this posterior to make decisions. This implementation
    uses a Normal prior and updates the posterior using observed data and estimated variance.

    Parameters
    ----------
    K : int
        Number of arms.
    R : int, default=1
        Number of parallel runs.
    prior_mean : float, default=0.0
        Prior mean shared across arms and runs.
    prior_var : float, default=1.0
        Prior variance (must be > 0). Larger values indicate less prior certainty.
    rng : np.random.Generator | None, default=None
        Random number generator for reproducibility. If None, uses default RNG.

    Raises
    ------
    ValueError
        If prior_var <= 0.


    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> policy = GaussianThompson(K=3, R=2, prior_mean=0.5, prior_var=0.25, rng=rng)
    >>> arms = policy.select_arm(t=1)  # Returns [0, 0] for forced exploration
    >>> policy.update(arms, np.array([0.6, 0.4]))

    References
    ----------
    .. [1] Thompson, W. R. (1933). On the likelihood that one unknown probability
           exceeds another in view of the evidence of two samples. Biometrika, 25(3/4), 285-294.
    .. [2] Agrawal, S., & Goyal, N. (2012). Analysis of Thompson sampling for the
           multi-armed bandit problem. COLT 2012.
    """

    def __init__(
        self,
        K: int,
        R: int = 1,
        prior_mean: float = 0.0,
        prior_var: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize Thompson sampling policy.

        Parameters
        ----------
        K : int
            Number of arms.
        R : int, default=1
            Number of parallel runs.
        prior_mean : float, default=0.0
            Prior mean for all arms.
        prior_var : float, default=1.0
            Prior variance (must be > 0).
        rng : np.random.Generator | None, default=None
            Random number generator for reproducibility.

        Raises
        ------
        ValueError
            If prior_var <= 0.
        """
        super().__init__(K, R, rng)
        if prior_var <= FLOAT_TOL:
            raise ValueError("prior_var must be positive")
        self.prior_mean = float(prior_mean)
        self.prior_var = float(prior_var)

    def select_arm(self, t: int) -> np.ndarray:
        """Sample arm means from posterior and select optimistically.

        Notes
        -----
        For arms with sufficient data (counts > 1), uses a Normal-Inverse-Gamma
        posterior with the observed sample variance. For arms with insufficient data,
        falls back to the prior distribution.

        The posterior distribution for the mean μ given variance σ² and data:
        - Posterior mean: weighted average of prior mean and sample mean
        - Posterior variance: incorporates both prior uncertainty and sample variance
        """
        forced = self._forced_exploration_arm(t)
        if forced is not None:
            return forced

        prior_precision = 1.0 / self.prior_var

        # Compute sample variance from M2
        sample_var = np.ones((self.K, self.R))  # Default to 1.0 for low-data arms
        valid_mask = self.counts > 1
        sample_var[valid_mask] = self.m2[valid_mask] / (self.counts[valid_mask] - 1)

        # For arms with data, use Bayesian update incorporating sample variance
        # For arms without data, use prior
        post_precision = np.zeros((self.K, self.R))
        post_mean = np.zeros((self.K, self.R))
        post_var = np.zeros((self.K, self.R))

        # Arms with no data: use prior
        no_data_mask = self.counts == 0
        post_mean[no_data_mask] = self.prior_mean
        post_var[no_data_mask] = self.prior_var

        # Arms with data: Bayesian update
        has_data_mask = self.counts > 0
        n = self.counts[has_data_mask]
        prior_prec = prior_precision

        # Avoid division by zero in sample variance
        safe_sample_var = np.maximum(sample_var[has_data_mask], MIN_POSTERIOR_VAR)
        data_prec = n / safe_sample_var

        post_precision[has_data_mask] = prior_prec + data_prec
        post_mean[has_data_mask] = (
            prior_prec * self.prior_mean + data_prec * self.means[has_data_mask]
        ) / post_precision[has_data_mask]

        # Posterior variance (numerical floor for stability)
        post_var[has_data_mask] = np.maximum(
            1.0 / post_precision[has_data_mask], MIN_POSTERIOR_VAR
        )

        theta = self.rng.normal(post_mean, np.sqrt(post_var))
        return np.argmax(theta, axis=0)
