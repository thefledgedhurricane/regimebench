"""Three Null Controls Experiment & Gap Statistic Remedy.

Null 1: i.i.d. Gaussian N(0,1)
Null 2: i.i.d. Student-t(5)
Null 3: Single-Regime GARCH(1,1) with t(5) innovations

Tests Silhouette, Davies-Bouldin, BIC, and Tibshirani's Gap Statistic (k in {1,2,3,4,5,6}).
"""

import os
import numpy as np
import pandas as pd

from regimebench.methods.clustering_wrappers import fit_kmeans, fit_gmm
from regimebench.criteria.k_selection import compute_internal_indices, compute_information_criteria, compute_gap_statistic

def generate_null_iid_gaussian(T: int = 5000, seed: int = 42) -> np.ndarray:
    """Null 1: i.i.d. Standard Gaussian N(0,1)."""
    rng = np.random.RandomState(seed)
    return rng.standard_normal(T)

def generate_null_iid_student_t(T: int = 5000, df: float = 5.0, seed: int = 42) -> np.ndarray:
    """Null 2: i.i.d. Student-t(5)."""
    rng = np.random.RandomState(seed)
    raw = rng.standard_t(df, size=T)
    return raw / np.sqrt(df / (df - 2.0))

def generate_null_single_garch(T: int = 5000, seed: int = 42) -> np.ndarray:
    """Null 3: Single-Regime GARCH(1,1) with Student-t(5) innovations (k*=1)."""
    rng = np.random.RandomState(seed)
    omega, alpha, beta, df = 0.05, 0.10, 0.85, 5.0
    y = np.zeros(T)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / (1.0 - alpha - beta)

    raw_t = rng.standard_t(df, size=T)
    innovations = raw_t / np.sqrt(df / (df - 2.0))

    y[0] = np.sqrt(sigma2[0]) * innovations[0]
    for t in range(1, T):
        sigma2[t] = omega + alpha * (y[t - 1] ** 2) + beta * sigma2[t - 1]
        y[t] = np.sqrt(sigma2[t]) * innovations[t]

    return y

def run_three_null_controls():
    print("=" * 90)
    print("RUNNING THREE NULL CONTROLS & GAP STATISTIC REMEDY")
    print("=" * 90)

    reps = 200
    T = 2500
    k_candidates = list(range(2, 7))

    null_generators = {
        'Null 1: i.i.d. Gaussian': generate_null_iid_gaussian,
        'Null 2: i.i.d. Student-t(5)': generate_null_iid_student_t,
        'Null 3: Single-Regime GARCH(1,1)': generate_null_single_garch
    }

    all_results = []

    for null_name, gen_func in null_generators.items():
        print(f"--- Evaluating {null_name} ({reps} Replications) ---")
        sil_selections = []
        gap_selections = []
        bic_selections = []

        for r in range(reps):
            seed = 10000 + r
            y = gen_func(T=T, seed=seed)
            abs_y = np.abs(y)
            roll_vol = pd.Series(abs_y).rolling(21, min_periods=1).mean().values.reshape(-1, 1)

            sil_k = {}
            bic_k = {}

            # Silhouette & BIC across k in {2,3,4,5,6}
            for k in k_candidates:
                labels_km, _ = fit_kmeans(roll_vol, k, random_state=seed)
                ind = compute_internal_indices(roll_vol, labels_km)
                sil_k[k] = ind['silhouette']

                labels_gmm, info = fit_gmm(roll_vol, k, random_state=seed)
                ic = compute_information_criteria(info['log_likelihood'], info['num_params'], len(roll_vol), info['posteriors'])
                bic_k[k] = ic['bic']

            k_hat_sil = max(sil_k, key=sil_k.get)
            k_hat_bic = min(bic_k, key=bic_k.get)

            # Gap Statistic across k in {1,2,3,4,5,6}
            def km_func(X, k):
                labels, _ = fit_kmeans(X, k, random_state=seed)
                return labels

            best_k_gap, _ = compute_gap_statistic(roll_vol[::20], km_func, k_range=range(1, 7), n_bootstraps=3)

            sil_selections.append(k_hat_sil)
            bic_selections.append(k_hat_bic)
            gap_selections.append(best_k_gap)

            if (r + 1) % 25 == 0:
                print(f"   ... {r + 1}/{reps} replications evaluated for {null_name}")

            all_results.append({
                'Null_Model': null_name,
                'Replication': r,
                'Silhouette_k_hat': k_hat_sil,
                'BIC_k_hat': k_hat_bic,
                'Gap_Statistic_k_hat': best_k_gap
            })

        sil_counts = pd.Series(sil_selections).value_counts().to_dict()
        gap_counts = pd.Series(gap_selections).value_counts().to_dict()
        bic_counts = pd.Series(bic_selections).value_counts().to_dict()

        print(f"   Silhouette k_hat Distribution: {sil_counts}")
        print(f"   Gap Statistic k_hat Distribution: {gap_counts}")
        print(f"   BIC k_hat Distribution: {bic_counts}\n")

    res_df = pd.DataFrame(all_results)
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "three_null_controls_results.csv")
    res_df.to_csv(csv_path, index=False)

    print("=" * 90)
    print(f"THREE NULL CONTROLS COMPLETE! Results CSV: {csv_path}")
    print("=" * 90)
    return res_df

if __name__ == "__main__":
    run_three_null_controls()
