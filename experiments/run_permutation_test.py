"""Circular block-shift permutation test for real-data partition agreement (1926-2026).

Rotates the reference chronology by a random circular offset, recomputes the ARI, and
repeats, so the null preserves the serial dependence of the reference rather than
destroying it as an i.i.d. label shuffle would.

Both reference chronologies are tested. Testing only Bry-Boschan/Pagan-Sossounov leaves
the largest observed agreement in the study -- the k=2 partition against NBER recessions --
without any significance statement at all, while the abstract generalizes over both.

The p-value uses the (1 + #{null >= obs}) / (B + 1) convention, which is the unbiased
Monte-Carlo estimator and cannot report an impossible p = 0.
"""

import os
import numpy as np
import pandas as pd

from regimebench.data.fama_french import fetch_fama_french_daily
from regimebench.data.fred import fetch_fred_series
from regimebench.reference.bry_boschan import bry_boschan_daily_dating
from regimebench.methods.clustering_wrappers import fit_kmeans
from regimebench.metrics.evaluators import compute_partition_quality

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
N_PERMUTATIONS = 1000


def circular_shift_test(reference: np.ndarray, labels: np.ndarray,
                        rng: np.random.RandomState, n_permutations: int = N_PERMUTATIONS):
    obs = compute_partition_quality(reference, labels)['ari']
    n = len(reference)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        shift = rng.randint(100, n - 100)
        null[i] = compute_partition_quality(np.roll(reference, shift), labels)['ari']
    p = (1.0 + np.sum(null >= obs)) / (n_permutations + 1.0)
    return {
        'observed_ari': float(obs),
        'null_mean_ari': float(np.mean(null)),
        'null_p95': float(np.percentile(null, 95)),
        'p_value': float(p),
    }


def run_permutation_experiment():
    print("=" * 90)
    print(f"CIRCULAR BLOCK-SHIFT PERMUTATION TEST ({N_PERMUTATIONS} shifts, both references)")
    print("=" * 90)

    ff = fetch_fama_french_daily()
    ff['Cum_Returns'] = (1.0 + ff['Returns']).cumprod() * 100.0
    ff['Bry_Boschan_Bear'] = bry_boschan_daily_dating(ff['Cum_Returns'])
    ff['Realized_Vol_21d'] = ff['Returns'].rolling(21, min_periods=1).std() * np.sqrt(252)
    ff = ff.dropna().copy()

    # NBER recession indicator, expanded from monthly to the daily trading calendar.
    usrec = fetch_fred_series("USREC")
    usrec_daily = usrec.reindex(usrec.index.union(ff.index)).ffill().reindex(ff.index)
    ff['NBER'] = usrec_daily.iloc[:, 0].fillna(0).astype(int).values

    X_vol = ff['Realized_Vol_21d'].values.reshape(-1, 1)
    references = {
        'NBER': ff['NBER'].values,
        'Bry-Boschan': ff['Bry_Boschan_Bear'].values,
    }
    print(f"  {len(ff):,} trading days  |  {ff.index.min().date()} to {ff.index.max().date()}")
    for name, ref in references.items():
        minority = min(np.mean(ref == v) for v in np.unique(ref))
        print(f"  {name}: minority-class share = {minority:.3f}")

    rows = []
    for k in (2, 3, 4, 5):
        labels, _ = fit_kmeans(X_vol, k, random_state=42)
        for ref_name, ref in references.items():
            rng = np.random.RandomState(42)
            res = circular_shift_test(ref, labels, rng)
            rows.append({
                'Partition Strategy': f'Forced k={k}',
                'Reference': ref_name,
                'Observed ARI': res['observed_ari'],
                'Null Mean ARI': res['null_mean_ari'],
                'Null 95th Percentile': res['null_p95'],
                'Empirical p-value': res['p_value'],
                'N shifts': N_PERMUTATIONS,
            })
            print(f"    k={k}  vs {ref_name:<12} ARI={res['observed_ari']:.4f}  p={res['p_value']:.3f}")

    res_df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "permutation_test_results.csv")
    res_df.to_csv(csv_path, index=False)

    print("\nRESULTS:")
    print(res_df.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    return res_df


if __name__ == "__main__":
    run_permutation_experiment()
