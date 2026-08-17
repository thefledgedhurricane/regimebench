"""Real-Data Agreement Table Experiment across Full Century (1926-2026) & Extended Range k in {2,...,12}."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regimebench.data.fama_french import fetch_fama_french_daily
from regimebench.data.fred import fetch_fred_series
from regimebench.reference.bry_boschan import bry_boschan_daily_dating
from regimebench.methods.clustering_wrappers import fit_kmeans, fit_gmm
from regimebench.criteria.k_selection import compute_internal_indices, compute_information_criteria
from regimebench.metrics.evaluators import compute_partition_quality

def run_real_data_agreement():
    print("=" * 90)
    print("RUNNING REAL-DATA AGREEMENT TABLE EXPERIMENT (FULL CENTURY 1926-2026, k in {2,...,12})")
    print("=" * 90)

    # 1. Fetch 100-Year Fama-French Daily Returns (1926-2026)
    ff = fetch_fama_french_daily()
    print(f"Loaded Fama-French Century Daily Data: {len(ff)} trading days (1926-2026)")

    # 2. Fetch FRED NBER Recession dates
    usrec = fetch_fred_series("USREC")
    ff = ff.join(usrec, how='left')
    ff['USREC'] = ff['USREC'].ffill().fillna(0).astype(int)

    # 3. Compute Corrected Daily Bry-Boschan Bear Market Regimes
    ff['Cum_Returns'] = (1.0 + ff['Returns']).cumprod() * 100.0
    ff['Bry_Boschan_Bear'] = bry_boschan_daily_dating(ff['Cum_Returns'])
    print(f"Corrected Bry-Boschan Bear Market Days: {ff['Bry_Boschan_Bear'].value_counts().to_dict()}")

    # 4. Feature: 21-day rolling volatility
    ff['Realized_Vol_21d'] = ff['Returns'].rolling(21, min_periods=1).std() * np.sqrt(252)
    ff = ff.dropna().copy()

    X_vol = ff['Realized_Vol_21d'].values.reshape(-1, 1)

    k_range = list(range(2, 13))  # k in {2, ..., 12}
    sil_k = {}
    bic_k = {}
    partition_dict = {}

    for k in k_range:
        labels_km, _ = fit_kmeans(X_vol, k)
        ind_km = compute_internal_indices(X_vol, labels_km)
        sil_k[k] = ind_km['silhouette']

        labels_gmm, info_gmm = fit_gmm(X_vol, k)
        ic_gmm = compute_information_criteria(info_gmm['log_likelihood'], info_gmm['num_params'], len(X_vol), info_gmm['posteriors'])
        bic_k[k] = ic_gmm['bic']

        partition_dict[f'KMeans_k_{k}'] = labels_km
        partition_dict[f'GMM_k_{k}'] = labels_gmm

    k_hat_sil = max(sil_k, key=sil_k.get)
    k_hat_bic = min(bic_k, key=bic_k.get)

    print(f"\nSelection Results across extended range k in {k_range}:")
    print(f"   Silhouette Selected k_hat = {k_hat_sil}")
    print(f"   BIC Selected k_hat = {k_hat_bic}")

    # Build Agreement Table against NBER and Bry-Boschan
    agreement_rows = []

    # Selected Partitions
    p_sil = partition_dict[f'KMeans_k_{k_hat_sil}']
    q_nber_sil = compute_partition_quality(ff['USREC'].values, p_sil)
    q_bb_sil = compute_partition_quality(ff['Bry_Boschan_Bear'].values, p_sil)
    agreement_rows.append({'Partition / Selection Strategy': f'Silhouette-Selected (k={k_hat_sil})', 'ARI vs NBER (USREC)': q_nber_sil['ari'], 'ARI vs Bry-Boschan': q_bb_sil['ari']})

    p_bic = partition_dict[f'GMM_k_{k_hat_bic}']
    q_nber_bic = compute_partition_quality(ff['USREC'].values, p_bic)
    q_bb_bic = compute_partition_quality(ff['Bry_Boschan_Bear'].values, p_bic)
    agreement_rows.append({'Partition / Selection Strategy': f'BIC-Selected (k={k_hat_bic})', 'ARI vs NBER (USREC)': q_nber_bic['ari'], 'ARI vs Bry-Boschan': q_bb_bic['ari']})

    # Forced Partitions k=2, 3, 4, 5
    for k in [2, 3, 4, 5]:
        p_forced = partition_dict[f'KMeans_k_{k}']
        q_nber_f = compute_partition_quality(ff['USREC'].values, p_forced)
        q_bb_f = compute_partition_quality(ff['Bry_Boschan_Bear'].values, p_forced)
        agreement_rows.append({'Partition / Selection Strategy': f'Forced Partition k={k}', 'ARI vs NBER (USREC)': q_nber_f['ari'], 'ARI vs Bry-Boschan': q_bb_f['ari']})

    agreement_df = pd.DataFrame(agreement_rows)

    print("\n" + "=" * 90)
    print("DELIVERABLE: REAL-DATA REFERENCE PARTITION AGREEMENT TABLE (1926-2026)")
    print("=" * 90)
    print(agreement_df.to_string(index=False))

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "real_data_agreement_table.csv")
    agreement_df.to_csv(csv_path, index=False)

    # ------------------------------------------------------------------
    # Figure: Fama-French century market regimes (SAME data as the table).
    # Generated here so the figure and the agreement table share one
    # provenance. Overwrites any stale S&P 500 overlay at the same path.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Panel 1: cumulative return with corrected Bry-Boschan bear regimes + NBER
    axes[0].plot(ff.index, ff['Cum_Returns'], color='#1F4E79', lw=1.0,
                 label='Fama-French US Market (cumulative, 1926=100)')
    axes[0].fill_between(ff.index, 0, ff['Cum_Returns'].max(),
                         where=(ff['Bry_Boschan_Bear'] == 1), color='red', alpha=0.20,
                         label='Bear market (Bry-Boschan / Pagan-Sossounov)')
    axes[0].fill_between(ff.index, 0, ff['Cum_Returns'].max(),
                         where=(ff['USREC'] == 1), color='black', alpha=0.30,
                         label='NBER recession')
    axes[0].set_yscale('log')
    axes[0].set_title("Fama-French US Daily Market Returns (1926-2026) with Corrected Bry-Boschan Bear Regimes & NBER Recessions",
                      fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Cumulative value (log scale)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left', fontsize=8)

    # Panel 2: the exact feature used for clustering (21-day realized vol)
    axes[1].plot(ff.index, ff['Realized_Vol_21d'], color='#4FACFE', lw=0.6,
                 label='21-day realized volatility (annualized)')
    axes[1].set_title("Clustering Feature: 21-Day Rolling Realized Volatility", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Annualized volatility")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "real_data_market_regimes.png")
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(out_dir, f"real_data_market_regimes.{ext}"), dpi=300)
    plt.close()
    print(f"Market Regimes Figure (Fama-French century): {fig_path}")

    print("\n" + "=" * 90)
    print(f"REAL-DATA AGREEMENT EXPERIMENT COMPLETE! Table CSV: {csv_path}")
    print("=" * 90)
    return agreement_df

if __name__ == "__main__":
    run_real_data_agreement()
