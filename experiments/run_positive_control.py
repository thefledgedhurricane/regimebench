"""Positive control: does the evaluation harness recover a partition when one is visible?

Every other experiment in this benchmark reports a failure. Without a positive control a
reader cannot distinguish "these pipelines fail" from "this evaluation code has a bug",
and a referee is trained to suspect the latter.

The generators already return the true conditional volatility ``sigma_t`` alongside the
returns, and every other experiment discards it. Here we cluster on a ladder of features
running from the oracle to the standard proxy:

  1. sigma_true      -- the latent conditional volatility itself (upper bound)
  2. sigma_log       -- log sigma_t (the scale on which regimes are additive)
  3. roll_sigma_21   -- 21-day rolling mean of sigma_t (window, but no estimation noise)
  4. roll_abs_r_21   -- 21-day rolling mean of |r_t| (the standard proxy used throughout)
  5. roll_abs_r_opt  -- rolling mean of |r_t| with the window matched to regime duration

If (1) recovers the partition and (5) does not, the failure localizes precisely to the
feature representation rather than to the harness, the metric, or the clustering method.
"""

import os
import numpy as np
import pandas as pd

from regimebench.generators.ms_garch import simulate_msgarch
from regimebench.generators.hmm_t import simulate_hmm_t
from regimebench.methods.clustering_wrappers import fit_kmeans, fit_gmm
from regimebench.metrics.evaluators import compute_partition_quality, mean_regime_duration

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Matches factor_grid_pilot.yaml so the positive control is directly comparable to the grid.
VOL_RATIOS = {2: [1.0, 2.5], 3: [1.0, 2.5, 5.0], 4: [1.0, 2.5, 5.0, 10.0]}


def build_features(y: np.ndarray, sigma: np.ndarray, duration: float) -> dict:
    matched = int(max(3, min(252, round(duration))))
    abs_y = np.abs(y)
    return {
        'sigma_true': sigma.reshape(-1, 1),
        'sigma_log': np.log(np.maximum(sigma, 1e-12)).reshape(-1, 1),
        'roll_sigma_21': pd.Series(sigma).rolling(21, min_periods=1).mean().values.reshape(-1, 1),
        'roll_abs_r_21': pd.Series(abs_y).rolling(21, min_periods=1).mean().values.reshape(-1, 1),
        'roll_abs_r_matched': pd.Series(abs_y).rolling(matched, min_periods=1).mean().values.reshape(-1, 1),
    }


def run_positive_control(replications: int = 20, T: int = 5000):
    print("=" * 90)
    print("POSITIVE CONTROL: ORACLE-FEATURE RECOVERY")
    print("=" * 90)

    rows = []
    for gen_name, gen_fn in [('ms_garch', simulate_msgarch), ('hmm_t', simulate_hmm_t)]:
        for k_star in (2, 3, 4):
            for p_ii in (0.90, 0.95, 0.99):
                duration = mean_regime_duration(p_ii)
                for r in range(replications):
                    seed = 77000 + 977 * k_star + 71 * int(p_ii * 100) + r
                    ratio_kw = ('vol_ratios' if gen_name == 'ms_garch' else 'stds')
                    kwargs = {'k': k_star, 'T': T, 'persistence': p_ii, 'df': 5.0,
                              'random_state': seed, ratio_kw: VOL_RATIOS[k_star]}
                    y, sigma, states = gen_fn(**kwargs)

                    for feat_name, X in build_features(y, sigma, duration).items():
                        labels_km, _ = fit_kmeans(X, k_star, random_state=seed)
                        labels_gmm, _ = fit_gmm(X, k_star, random_state=seed)
                        rows.append({
                            'Generator': gen_name, 'True_k': k_star, 'Persistence': p_ii,
                            'Mean_Regime_Duration': duration, 'Replication': r,
                            'Feature': feat_name,
                            'ARI_KMeans_oracle_k': compute_partition_quality(states, labels_km)['ari'],
                            'ARI_GMM_oracle_k': compute_partition_quality(states, labels_gmm)['ari'],
                        })
                print(f"  {gen_name}  k*={k_star}  P_ii={p_ii:.2f}  ({replications} reps) done")

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "positive_control_results.csv")
    df.to_csv(path, index=False)

    print("\n--- Mean ARI at the oracle k*, by feature (K-Means) ---")
    order = ['sigma_true', 'sigma_log', 'roll_sigma_21', 'roll_abs_r_matched', 'roll_abs_r_21']
    piv = df.pivot_table(index='Feature', columns='Persistence',
                         values='ARI_KMeans_oracle_k', aggfunc='mean').reindex(order)
    print(piv.round(4).to_string())

    print("\n--- Same, split by generator ---")
    piv2 = df.pivot_table(index='Feature', columns='Generator',
                          values='ARI_KMeans_oracle_k', aggfunc='mean').reindex(order)
    print(piv2.round(4).to_string())

    print(f"\nSaved: {path}")
    return df


if __name__ == "__main__":
    run_positive_control()
