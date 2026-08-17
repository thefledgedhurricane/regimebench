"""Cross-Family Robustness Check: Stochastic Volatility + Jumps (Generator C).

Unlike MS-GARCH and HMM-t, the SV+Jump process (regimebench.generators.sv_jump) has
NO discrete latent regime-switching mechanism: log-volatility follows a continuous
AR(1) process with occasional Poisson jumps. Any discrete k selected by a clustering
criterion here is, by construction, not recovering a "true" generative regime count -
it can only be evaluated against a post-hoc volatility-tercile proxy (k=3 by definition)
or treated as a null-control-style check for spurious regime detection on a smooth,
continuously-varying, jump-diffusion volatility process.
"""

import os
import numpy as np
import pandas as pd

from regimebench.generators.sv_jump import simulate_sv_jump
from regimebench.methods.clustering_wrappers import fit_kmeans, fit_gmm
from regimebench.criteria.k_selection import compute_internal_indices, compute_information_criteria, compute_gap_statistic
from regimebench.metrics.evaluators import compute_partition_quality

def run_sv_jump_robustness():
    print("=" * 90)
    print("RUNNING CROSS-FAMILY ROBUSTNESS CHECK: STOCHASTIC VOLATILITY + JUMPS")
    print("=" * 90)

    reps = 30
    T = 2500
    k_candidates = list(range(2, 7))

    results = []

    for r in range(reps):
        seed = 20000 + r
        y, vol, tercile_proxy = simulate_sv_jump(T=T, random_state=seed)
        abs_y = np.abs(y)
        roll_vol = pd.Series(abs_y).rolling(21, min_periods=1).mean().values.reshape(-1, 1)

        sil_k, bic_k = {}, {}
        ari_at_k3 = 0.0
        for k in k_candidates:
            labels_km, _ = fit_kmeans(roll_vol, k, random_state=seed)
            ind = compute_internal_indices(roll_vol, labels_km)
            sil_k[k] = ind['silhouette']
            if k == 3:
                ari_at_k3 = compute_partition_quality(tercile_proxy, labels_km)['ari']

            labels_gmm, info = fit_gmm(roll_vol, k, random_state=seed)
            ic = compute_information_criteria(info['log_likelihood'], info['num_params'], len(roll_vol), info['posteriors'])
            bic_k[k] = ic['bic']

        k_hat_sil = max(sil_k, key=sil_k.get)
        k_hat_bic = min(bic_k, key=bic_k.get)

        def km_func(X, k):
            labels, _ = fit_kmeans(X, k, random_state=seed)
            return labels

        best_k_gap, _ = compute_gap_statistic(roll_vol[::20], km_func, k_range=range(1, 7), n_bootstraps=3)

        results.append({
            'Replication': r,
            'Silhouette_k_hat': k_hat_sil,
            'BIC_k_hat': k_hat_bic,
            'Gap_Statistic_k_hat': best_k_gap,
            'ARI_vs_VolTercile_at_k3': ari_at_k3,
        })

    res_df = pd.DataFrame(results)

    print("SUMMARY (True Discrete Regime Count: NONE - continuous SV+Jump process):")
    print(f"   Silhouette k_hat Distribution: {res_df['Silhouette_k_hat'].value_counts().to_dict()}")
    print(f"   BIC k_hat Distribution: {res_df['BIC_k_hat'].value_counts().to_dict()}")
    print(f"   Gap Statistic k_hat Distribution: {res_df['Gap_Statistic_k_hat'].value_counts().to_dict()}")
    print(f"   Mean ARI vs Volatility-Tercile Proxy (k=3): {res_df['ARI_vs_VolTercile_at_k3'].mean():.4f}")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "sv_jump_robustness_results.csv")
    res_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 90)
    print(f"SV+JUMP ROBUSTNESS CHECK COMPLETE! Results CSV: {csv_path}")
    print("=" * 90)
    return res_df

if __name__ == "__main__":
    run_sv_jump_robustness()
