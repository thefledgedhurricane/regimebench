"""Generator C: Stochastic Volatility Process with Discrete Jumps (Robustness Check)."""

import numpy as np
from typing import Tuple, Optional

class SV_Jump_Simulator:
    """Simulates a Stochastic Volatility model with Poisson Jumps.
    
    Equations:
        h_t = mu + phi * (h_{t-1} - mu) + eta_t,  eta_t ~ N(0, sigma_eta^2)
        y_t = exp(h_t / 2) * epsilon_t + J_t * N_t
        J_t ~ N(mu_J, sigma_J^2), N_t ~ Poisson(lambda_J)
    """

    def __init__(
        self,
        mu: float = -1.0,
        phi: float = 0.98,
        sigma_eta: float = 0.15,
        jump_lambda: float = 0.02,
        jump_mu: float = -2.0,
        jump_sigma: float = 3.0,
        random_state: Optional[int] = 42
    ):
        self.mu = mu
        self.phi = phi
        self.sigma_eta = sigma_eta
        self.jump_lambda = jump_lambda
        self.jump_mu = jump_mu
        self.jump_sigma = jump_sigma
        self.rng = np.random.RandomState(random_state)

    def generate(self, T: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates T observations of Stochastic Volatility with Jumps."""
        h = np.zeros(T)
        y = np.zeros(T)
        jump_flags = np.zeros(T, dtype=int)

        h[0] = self.mu
        for t in range(1, T):
            h[t] = self.mu + self.phi * (h[t - 1] - self.mu) + self.rng.normal(0, self.sigma_eta)

        vol = np.exp(h / 2.0)
        eps = self.rng.standard_normal(T)
        y = vol * eps

        # Add Poisson jumps
        jumps = self.rng.poisson(self.jump_lambda, size=T)
        for t in range(T):
            if jumps[t] > 0:
                jump_size = self.rng.normal(self.jump_mu, self.jump_sigma) * jumps[t]
                y[t] += jump_size
                jump_flags[t] = 1

        # Continuous regime proxy based on volatility terciles
        q1, q2 = np.percentile(vol, [33.3, 66.7])
        state_proxy = np.zeros(T, dtype=int)
        state_proxy[vol >= q1] = 1
        state_proxy[vol >= q2] = 2

        return y, vol, state_proxy

def simulate_sv_jump(
    T: int = 5000,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience helper function for SV Jump simulation."""
    sim = SV_Jump_Simulator(random_state=random_state)
    return sim.generate(T=T)
