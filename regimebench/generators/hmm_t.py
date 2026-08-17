"""Generator B: Hidden Markov Model with Student-t Emissions."""

import numpy as np
from typing import Tuple, Optional, List

class HMM_StudentT_Simulator:
    """Simulates a Hidden Markov Model with Student-t emission distributions.
    
    Isolates the effect of fat tails without GARCH volatility dynamics.
    """

    def __init__(
        self,
        k: int = 3,
        persistence: float = 0.95,
        means: Optional[List[float]] = None,
        stds: Optional[List[float]] = None,
        df: float = 5.0,
        random_state: Optional[int] = 42
    ):
        self.k = k
        self.persistence = persistence
        self.df = df
        self.rng = np.random.RandomState(random_state)

        # Transition matrix P
        off_diag = (1.0 - persistence) / max(k - 1, 1)
        self.P = np.full((k, k), off_diag)
        np.fill_diagonal(self.P, persistence)

        # State means & standard deviations
        if means is None:
            self.means = np.zeros(k)
        else:
            self.means = np.array(means)

        if stds is None:
            self.stds = np.array([1.5 ** i for i in range(k)])
        else:
            self.stds = np.array(stds)

    def generate(self, T: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates T observations of HMM with Student-t emissions."""
        states = np.zeros(T, dtype=int)
        y = np.zeros(T)
        sigma = np.zeros(T)

        # Stationary initial state distribution
        eigenvals, eigenvecs = np.linalg.eig(self.P.T)
        stationary_idx = np.argmin(np.abs(eigenvals - 1.0))
        pi = np.real(eigenvecs[:, stationary_idx])
        pi /= np.sum(pi)

        states[0] = self.rng.choice(self.k, p=pi)

        for t in range(T):
            if t > 0:
                states[t] = self.rng.choice(self.k, p=self.P[states[t - 1]])

            curr_s = states[t]
            mu = self.means[curr_s]
            s = self.stds[curr_s]
            sigma[t] = s

            if np.isinf(self.df) or self.df > 100:
                eps = self.rng.standard_normal()
            else:
                eps = self.rng.standard_t(self.df) / np.sqrt(self.df / (self.df - 2.0))

            y[t] = mu + s * eps

        return y, sigma, states

def simulate_hmm_t(
    k: int = 3,
    T: int = 5000,
    persistence: float = 0.95,
    stds: Optional[List[float]] = None,
    df: float = 5.0,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience helper function for HMM Student-t simulation."""
    sim = HMM_StudentT_Simulator(k=k, persistence=persistence, stds=stds, df=df, random_state=random_state)
    return sim.generate(T=T)
