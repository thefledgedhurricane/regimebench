"""The Decisive Experiment: Fast Multivariate Representation Test (4-D Feature Space).

Features: [Returns r_t, Realized Vol_21d, Rolling Absolute Returns_21d, Rolling Squared Returns_21d].
Tests whether Silhouette's collapse survives in higher dimensions under genuine MS-GARCH dynamics!
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from regimebench.generators.ms_garch import simulate_msgarch

def extract_multivariate_features_fast(y: np.ndarray) -> np.ndarray:
    """Fast vectorized 4-D feature space: [r_t, Realized_Vol_21d, Rolling_Abs_21d, Rolling_Sq_21d]."""
    s = pd.Series(y)
    vol = s.rolling(21, min_periods=1).std().fillna(0).values
    abs_r = s.abs().rolling(21, min_periods=1).mean().fillna(0).values
    sq_r = (s ** 2).rolling(21, min_periods=1).mean().fillna(0).values

    X = np.column_stack([y, vol, abs_r, sq_r])
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1.0
    return (X - means) / stds

def run_multivariate_test():
    print("=" * 90)
    print("RUNNING THE DECISIVE EXPERIMENT: FAST MULTIVARIATE TEST (4-D FEATURES)")
    print("=" * 90)

    T = 2500
    reps = 30
    k_candidates = list(range(2, 7))

    scenarios = [
        ('MS-GARCH k*=3 (4-D Features)', 3),
        ('MS-GARCH k*=4 (4-D Features)', 4),
        ('Null Single-Regime (k*=1, 4-D Features)', 1)
    ]

    all_results = []

    for name, true_k in scenarios:
        print(f"--- Evaluating 4-D Features for {name} ({reps} Replications) ---")
        sil_k_hats = []
        bic_k_hats = []

        for r in range(reps):
            seed = 5000 + r
            y, sigma, true_states = simulate_msgarch(k=true_k, T=T, persistence=0.95, df=5.0, random_state=seed)
            X_4d = extract_multivariate_features_fast(y)

            sil_k = {}
            bic_k = {}

            for k in k_candidates:
                km = KMeans(n_clusters=k, random_state=seed, n_init=3)
                labels = km.fit_predict(X_4d)
                sil_k[k] = float(silhouette_score(X_4d[::5], labels[::5]))

                gmm = GaussianMixture(n_components=k, random_state=seed, n_init=1)
                gmm.fit(X_4d)
                log_l = gmm.score(X_4d) * len(X_4d)
                n_params = k * 4 + k * 10 + (k - 1)
                bic_k[k] = -2.0 * log_l + n_params * np.log(len(X_4d))

            k_hat_sil = max(sil_k, key=sil_k.get)
            k_hat_bic = min(bic_k, key=bic_k.get)

            sil_k_hats.append(k_hat_sil)
            bic_k_hats.append(k_hat_bic)

            all_results.append({
                'Scenario': name,
                'Replication': r,
                'Silhouette_k_hat': k_hat_sil,
                'BIC_k_hat': k_hat_bic
            })

        print(f"   4-D Silhouette k_hat Distribution: {pd.Series(sil_k_hats).value_counts().to_dict()}")
        print(f"   4-D BIC k_hat Distribution: {pd.Series(bic_k_hats).value_counts().to_dict()}\n")

    res_df = pd.DataFrame(all_results)
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "multivariate_test_results.csv")
    res_df.to_csv(csv_path, index=False)

    print("=" * 90)
    print(f"FAST MULTIVARIATE TEST COMPLETE! Results CSV: {csv_path}")
    print("=" * 90)
    return res_df

if __name__ == "__main__":
    run_multivariate_test()
