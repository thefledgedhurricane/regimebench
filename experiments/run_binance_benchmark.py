"""Cross-Asset Robustness Check: BTCUSDT 1-Minute Klines from Binance Public Data.

Tests whether the k_hat=2 collapse observed for geometric internal indices (Silhouette,
Davies-Bouldin) on equities and synthetic processes also appears on a structurally
different asset class (crypto) and sampling frequency (1-minute, exchange-traded 24/7,
no overnight gaps). No ground-truth regime labels exist for crypto, so this experiment
reports selection-criteria agreement/disagreement rather than recovery against a reference
partition.
"""

import os
import numpy as np
import pandas as pd

from regimebench.data.binance import fetch_binance_klines
from regimebench.methods.clustering_wrappers import fit_kmeans, fit_gmm
from regimebench.criteria.k_selection import compute_internal_indices, compute_information_criteria, compute_gap_statistic

def run_binance_benchmark(symbol: str = "BTCUSDT", year: int = 2024, month: int = 1):
    print("=" * 90)
    print(f"RUNNING CROSS-ASSET ROBUSTNESS CHECK: {symbol} 1-MINUTE KLINES ({year}-{month:02d})")
    print("=" * 90)

    df = fetch_binance_klines(symbol=symbol, year=year, month=month)
    n_raw = len(df)
    print(f"Raw 1-minute klines fetched (post-dropna): {n_raw:,}")

    X_full = df['Realized_Vol_60m'].values.reshape(-1, 1)

    # Subsample every 10th minute for O(n^2) criteria (Silhouette, Gap Statistic),
    # consistent with the subsampling convention used elsewhere in this benchmark suite.
    X_sub = X_full[::10]
    n_sub = len(X_sub)
    print(f"Subsampled points used for clustering/selection (every 10th minute): {n_sub:,}\n")

    k_candidates = list(range(2, 7))
    sil_k, db_k, bic_k, aic_k, icl_k = {}, {}, {}, {}, {}

    for k in k_candidates:
        labels_km, _ = fit_kmeans(X_sub, k, random_state=42)
        ind = compute_internal_indices(X_sub, labels_km)
        sil_k[k] = ind['silhouette']
        db_k[k] = ind['davies_bouldin']

        labels_gmm, info = fit_gmm(X_sub, k, random_state=42)
        ic = compute_information_criteria(info['log_likelihood'], info['num_params'], n_sub, info['posteriors'])
        bic_k[k] = ic['bic']
        aic_k[k] = ic['aic']
        icl_k[k] = ic['icl']

    k_hat_sil = max(sil_k, key=sil_k.get)
    k_hat_db = min(db_k, key=db_k.get)
    k_hat_bic = min(bic_k, key=bic_k.get)
    k_hat_aic = min(aic_k, key=aic_k.get)
    k_hat_icl = min(icl_k, key=icl_k.get)

    def km_func(X, k):
        labels, _ = fit_kmeans(X, k, random_state=42)
        return labels

    X_gap = X_full[::50]
    best_k_gap, gap_values = compute_gap_statistic(X_gap, km_func, k_range=range(1, 7), n_bootstraps=5)

    print("SELECTION RESULTS:")
    print(f"   Silhouette Selected k_hat        = {k_hat_sil}  (values: {sil_k})")
    print(f"   Davies-Bouldin Selected k_hat     = {k_hat_db}  (values: {db_k})")
    print(f"   GMM BIC Selected k_hat            = {k_hat_bic}  (values: {bic_k})")
    print(f"   GMM AIC Selected k_hat            = {k_hat_aic}  (values: {aic_k})")
    print(f"   GMM ICL Selected k_hat            = {k_hat_icl}  (values: {icl_k})")
    print(f"   Gap Statistic Selected k_hat      = {best_k_gap}  (N={len(X_gap):,}, values: {gap_values})")

    results = [{
        'Symbol': symbol,
        'Year': year,
        'Month': month,
        'N_Raw_Klines': n_raw,
        'N_Subsampled': n_sub,
        'Silhouette_k_hat': k_hat_sil,
        'DaviesBouldin_k_hat': k_hat_db,
        'BIC_k_hat': k_hat_bic,
        'AIC_k_hat': k_hat_aic,
        'ICL_k_hat': k_hat_icl,
        'GapStatistic_k_hat': best_k_gap,
    }]
    res_df = pd.DataFrame(results)

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "binance_benchmark_results.csv")
    res_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 90)
    print(f"BINANCE ROBUSTNESS CHECK COMPLETE! Results CSV: {csv_path}")
    print("=" * 90)
    return res_df

if __name__ == "__main__":
    run_binance_benchmark()
