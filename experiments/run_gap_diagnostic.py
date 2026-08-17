"""Why the Gap statistic answers k=1: the shape of the gap curve.

The null-control battery reports that the Gap statistic returns k_hat=1 on processes with
no regime structure. Read alone, that looks like the one criterion in the benchmark that
behaves correctly. This diagnostic shows it is not.

Under the classical bounding-box reference of Tibshirani et al. (2001) method (a), the gap
curve on a right-skewed volatility feature is *monotone decreasing*: gap(1) is the global
maximum even on series with four genuine, highly persistent regimes. The one-standard-error
rule ("smallest k with gap(k) >= gap(k+1) - se(k+1)") can then only ever return the smallest
admissible candidate. The criterion is pinned to the floor of its candidate range, exactly
as Silhouette is pinned to k=2 and Calinski-Harabasz to the ceiling.

The mechanism is the null model, not the rule. A uniform reference asks "is this data more
clustered than *uniform* noise?". A rolling mean of |r| is strongly right-skewed and so is
very far from uniform whatever its regime structure, which inflates gap(1) by the marginal
shape alone. This script emits the curve under each available reference so the effect is
visible rather than asserted.
"""

import os
import numpy as np
import pandas as pd

from regimebench.generators.ms_garch import simulate_msgarch
from regimebench.methods.clustering_wrappers import fit_kmeans
from regimebench.criteria.k_selection import compute_gap_curve

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
REFERENCES = ["uniform", "gaussian", "lognormal"]


def run_gap_diagnostic(replications: int = 10, T: int = 2000, n_bootstraps: int = 20):
    print("=" * 90)
    print("GAP CURVE DIAGNOSTIC")
    print("=" * 90)

    rows = []
    for k_star in (2, 3, 4):
        for p_ii in (0.90, 0.95, 0.99):
            for r in range(replications):
                seed = 31000 + 311 * k_star + 31 * int(p_ii * 100) + r
                y, _, _ = simulate_msgarch(k=k_star, T=T, persistence=p_ii,
                                           vol_ratios=None, df=5.0, random_state=seed)
                X = pd.Series(np.abs(y)).rolling(21, min_periods=1).mean().values.reshape(-1, 1)
                clus = lambda A, k: fit_kmeans(A, k, random_state=seed)[0]

                for ref in REFERENCES:
                    curve = compute_gap_curve(X, clus, k_range=range(1, 7),
                                              n_bootstraps=n_bootstraps,
                                              random_state=seed, reference=ref)
                    gaps, ses = curve['gap'], curve['se']
                    for k in range(1, 7):
                        rows.append({
                            'k_star': k_star, 'persistence': p_ii, 'replication': r,
                            'reference': ref, 'k': k,
                            'gap': gaps[k], 'se': ses[k], 'log_w': curve['log_w'][k],
                            'argmax_is_1': int(max(gaps, key=gaps.get) == 1),
                        })
            print(f"  k*={k_star}  P_ii={p_ii:.2f}  done")

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "gap_curve_diagnostic.csv")
    df.to_csv(path, index=False)

    print("\n--- Mean gap(k) by reference distribution (pooled over the grid) ---")
    print(df.pivot_table(index='k', columns='reference', values='gap', aggfunc='mean').round(4).to_string())

    print("\n--- Share of series whose gap curve is maximised at k=1 ---")
    share = df[df.k == 1].groupby(['reference', 'k_star'])['argmax_is_1'].mean().mul(100)
    print(share.round(1).to_string())

    print(f"\nSaved: {path}")
    return df


if __name__ == "__main__":
    run_gap_diagnostic()
