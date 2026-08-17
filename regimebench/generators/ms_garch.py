"""Generator A: Markov-Switching GARCH(1,1) Process Simulator."""

import numpy as np
from typing import Tuple, Optional, Dict, Any

class MS_GARCH_Simulator:
    """Simulates a Markov-Switching GARCH(1,1) process with Student-t or Gaussian innovations."""

    def __init__(
        self,
        k: int = 3,
        persistence: float = 0.95,
        vol_ratios: Optional[list] = None,
        df: float = 5.0,
        random_state: Optional[int] = 42
    ):
        self.k = k
        self.persistence = persistence
        self.df = df
        self.rng = np.random.RandomState(random_state)

        # Build symmetric transition matrix P with specified diagonal persistence
        if k == 1:
            self.P = np.array([[1.0]])
        else:
            off_diag = (1.0 - persistence) / max(k - 1, 1)
            self.P = np.full((k, k), off_diag)
            np.fill_diagonal(self.P, persistence)

        # Volatility scale per regime
        if vol_ratios is None:
            self.vol_scales = np.array([1.5 ** i for i in range(k)])
        else:
            self.vol_scales = np.array(vol_ratios)

        # GARCH(1,1) regime parameters: (omega, alpha, beta)
        self.garch_params = []
        for i in range(k):
            scale = self.vol_scales[i]
            omega = 0.05 * (scale ** 2)
            alpha = 0.10
            beta = 0.85
            self.garch_params.append({'omega': omega, 'alpha': alpha, 'beta': beta})

    def generate(self, T: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates T observations of MS-GARCH(1,1)."""
        states = np.zeros(T, dtype=int)
        y = np.zeros(T)
        sigma2 = np.zeros(T)

        if self.k == 1:
            pi = np.array([1.0])
        else:
            eigenvals, eigenvecs = np.linalg.eig(self.P.T)
            stationary_idx = np.argmin(np.abs(eigenvals - 1.0))
            pi = np.real(eigenvecs[:, stationary_idx])
            pi /= np.sum(pi)

        states[0] = self.rng.choice(self.k, p=pi)
        sigma2[0] = self.garch_params[states[0]]['omega'] / max(1.0 - self.garch_params[states[0]]['alpha'] - self.garch_params[states[0]]['beta'], 0.01)

        if np.isinf(self.df) or self.df > 100:
            innovations = self.rng.standard_normal(T)
        else:
            raw = self.rng.standard_t(self.df, size=T)
            innovations = raw / np.sqrt(self.df / (self.df - 2.0))

        y[0] = np.sqrt(sigma2[0]) * innovations[0]

        for t in range(1, T):
            prev_state = states[t - 1]
            curr_state = self.rng.choice(self.k, p=self.P[prev_state])
            states[t] = curr_state

            params = self.garch_params[curr_state]
            s2 = params['omega'] + params['alpha'] * (y[t - 1] ** 2) + params['beta'] * sigma2[t - 1]
            sigma2[t] = max(s2, 1e-6)
            y[t] = np.sqrt(sigma2[t]) * innovations[t]

        return y, np.sqrt(sigma2), states

def simulate_msgarch(
    k: int = 3,
    T: int = 5000,
    persistence: float = 0.95,
    vol_ratios: Optional[list] = None,
    df: float = 5.0,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience helper function for MS-GARCH simulation."""
    sim = MS_GARCH_Simulator(k=k, persistence=persistence, vol_ratios=vol_ratios, df=df, random_state=random_state)
    return sim.generate(T=T)
