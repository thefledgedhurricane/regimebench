"""Evaluation Metrics Module: k Recovery, Signed Bias, ARI, NMI & Rank Inversion."""

import numpy as np
from typing import Dict, Any, List
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def compute_partition_quality(true_states: np.ndarray, pred_labels: np.ndarray) -> Dict[str, float]:
    """Computes ARI and NMI against true ground-truth regime states."""
    try:
        ari = float(adjusted_rand_score(true_states, pred_labels))
    except Exception:
        ari = 0.0

    try:
        nmi = float(normalized_mutual_info_score(true_states, pred_labels))
    except Exception:
        nmi = 0.0

    return {'ari': ari, 'nmi': nmi}

def compute_recovery_metrics(k_estimated_list: List[int], k_true: int) -> Dict[str, float]:
    """Computes k recovery statistics across experiment replications:
    - exact_recovery_rate: P(k_hat == k_true)
    - signed_bias: E[k_hat - k_true] (Tests H1)
    - mae: E[|k_hat - k_true|]
    """
    k_est = np.array(k_estimated_list)
    exact_recovery = float(np.mean(k_est == k_true))
    bias = float(np.mean(k_est - k_true))
    mae = float(np.mean(np.abs(k_est - k_true)))

    return {
        'exact_recovery_rate': exact_recovery,
        'signed_bias': bias,
        'mae': mae,
        'mean_k_estimated': float(np.mean(k_est))
    }

def wilson_interval(successes: int, n: int, z: float = 1.959963985) -> Dict[str, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation at the small cell sizes used here
    (n = 30 per grid cell), where the Wald interval is badly miscalibrated and can
    leave [0, 1] entirely at rates near 0% or 100%.
    """
    if n <= 0:
        return {'rate': float('nan'), 'lo': float('nan'), 'hi': float('nan'), 'n': 0}

    p = successes / n
    denom = 1.0 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return {
        'rate': float(p),
        'lo': float(max(0.0, centre - half)),
        'hi': float(min(1.0, centre + half)),
        'n': int(n),
    }


def format_rate_ci(successes: int, n: int, pct: bool = True) -> str:
    """Formats a rate with its Wilson interval for direct use in a LaTeX table cell."""
    w = wilson_interval(successes, n)
    if pct:
        return f"{w['rate']*100:.0f} [{w['lo']*100:.0f}, {w['hi']*100:.0f}]"
    return f"{w['rate']:.3f} [{w['lo']:.3f}, {w['hi']:.3f}]"


def mean_regime_duration(persistence: float) -> float:
    """Expected regime dwell time in observations for a self-transition probability.

    A geometric dwell time with self-transition p has mean 1/(1-p): 10 observations at
    p=0.90, 20 at 0.95, 100 at 0.99. Compare against the feature window before reading
    any recovery result -- a 21-day window cannot resolve a 10-day state.
    """
    return 1.0 / (1.0 - persistence)


def compute_rank_inversion(
    ari_at_true_k: List[float], 
    ari_at_selected_k: List[float]
) -> Dict[str, float]:
    """Computes Spearman correlation between method rankings by ARI at true k* vs ARI at selected k (Tests H3)."""
    if len(ari_at_true_k) < 3:
        return {'spearman_rho': 1.0, 'p_value': 1.0}

    rho, pval = spearmanr(ari_at_true_k, ari_at_selected_k)
    return {'spearman_rho': float(rho) if not np.isnan(rho) else 0.0, 'p_value': float(pval) if not np.isnan(pval) else 1.0}
