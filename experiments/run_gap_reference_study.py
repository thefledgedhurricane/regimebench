"""Gap statistic and Prediction Strength: one harmonized design, null and alternative.

This supersedes the earlier split between a null-control run and a power run, which used
different candidate ranges (k>=1 vs k>=2), different series lengths, different subsampling
and different bootstrap counts for the *same* criterion -- and then compared the two
head-to-head. Every cell below shares one design:

    T = 2000, k in {1..6}, 30 replications, 10 bootstrap references, seeded.

The reference distribution is swept as an explicit factor, because it -- not the data --
determines whether the Gap statistic can answer anything but 1. See run_gap_diagnostic.py.

Cells: k*=1 (single-regime GARCH null) plus MS-GARCH k* in {2,3,4} x P_ii in {0.90,0.95,0.99}.
"""

import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from regimebench.generators.ms_garch import simulate_msgarch
from regimebench.methods.clustering_wrappers import fit_kmeans
from regimebench.criteria.k_selection import compute_gap_statistic, compute_prediction_strength

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
REFERENCES = ["uniform", "gaussian", "lognormal"]
K_RANGE = range(1, 7)
T = 2000
REPLICATIONS = 30
N_BOOTSTRAPS = 10


def single_regime_garch(T: int, seed: int) -> np.ndarray:
    """Null process: one GARCH(1,1) regime with Student-t(5) innovations, k*=1."""
    rng = np.random.RandomState(seed)
    omega, alpha, beta, df = 0.05, 0.10, 0.85, 5.0
    y = np.zeros(T)
    s2 = np.zeros(T)
    s2[0] = omega / (1.0 - alpha - beta)
    eps = rng.standard_t(df, size=T) / np.sqrt(df / (df - 2.0))
    y[0] = np.sqrt(s2[0]) * eps[0]
    for t in range(1, T):
        s2[t] = omega + alpha * y[t - 1] ** 2 + beta * s2[t - 1]
        y[t] = np.sqrt(s2[t]) * eps[t]
    return y


def evaluate_cell(k_star: float, p_ii, rep: int):
    seed = 4200 + 137 * int(k_star) + 17 * int((p_ii or 0) * 100) + rep

    if k_star == 1:
        y = single_regime_garch(T, seed)
    else:
        y, _, _ = simulate_msgarch(k=int(k_star), T=T, persistence=p_ii,
                                   vol_ratios=None, df=5.0, random_state=seed)

    X = pd.Series(np.abs(y)).rolling(21, min_periods=1).mean().values.reshape(-1, 1)
    clus = lambda A, k: fit_kmeans(A, k, random_state=seed)[0]

    out = []
    for ref in REFERENCES:
        k_hat, _ = compute_gap_statistic(X, clus, k_range=K_RANGE,
                                         n_bootstraps=N_BOOTSTRAPS,
                                         random_state=seed, reference=ref)
        out.append({'k_star': k_star, 'persistence': p_ii, 'replication': rep,
                    'criterion': f'Gap ({ref})', 'k_hat': k_hat})

    ps_k, _ = compute_prediction_strength(X, k_range=K_RANGE, random_state=seed)
    out.append({'k_star': k_star, 'persistence': p_ii, 'replication': rep,
                'criterion': 'Prediction Strength', 'k_hat': ps_k})
    return out


def run_gap_reference_study():
    print("=" * 90)
    print("GAP REFERENCE STUDY (harmonized null + alternative)")
    print("=" * 90)
    print(f"  T={T}  k in {list(K_RANGE)}  reps={REPLICATIONS}  bootstraps={N_BOOTSTRAPS}")
    print(f"  references swept: {REFERENCES}\n")

    cells = [(1, None)] + [(k, p) for k in (2, 3, 4) for p in (0.90, 0.95, 0.99)]
    tasks = [(k, p, r) for (k, p) in cells for r in range(REPLICATIONS)]

    nested = Parallel(n_jobs=-1, batch_size=2, verbose=10)(
        delayed(evaluate_cell)(k, p, r) for (k, p, r) in tasks
    )
    df = pd.DataFrame([row for group in nested for row in group])
    df['exact_match'] = (df['k_hat'] == df['k_star']).astype(int)
    df['signed_bias'] = df['k_hat'] - df['k_star']
    df['selected_k1'] = (df['k_hat'] == 1).astype(int)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "gap_reference_study.csv")
    df.to_csv(path, index=False)

    print("\n--- P(k_hat = 1) by criterion and true k* ---")
    print(df.pivot_table(index='criterion', columns='k_star',
                         values='selected_k1', aggfunc='mean').mul(100).round(1).to_string())

    print("\n--- Exact recovery P(k_hat = k*) by criterion and true k* ---")
    print(df.pivot_table(index='criterion', columns='k_star',
                         values='exact_match', aggfunc='mean').mul(100).round(1).to_string())

    print(f"\nSaved: {path}")
    return df


if __name__ == "__main__":
    run_gap_reference_study()
