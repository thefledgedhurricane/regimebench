"""How much regime separation does the pipeline actually need?

Two quantities are easy to confuse and must be reported separately:

  * the *intended* separation, set by ``volatility_ratios`` in the config;
  * the *realized* separation, i.e. the ratio of realized standard deviations the states
    actually achieve in a finite sample.

Under MS-GARCH they are not the same. With alpha+beta = 0.95 the conditional variance needs
roughly 20 observations to adapt after a switch, so at P_ii = 0.90 (mean dwell 10) or 0.95
(mean dwell 20) the process never reaches its regime's unconditional variance and the
realized separation is compressed far below the intended one. That compression, not the
choice of validity index, is the first-order driver of low partition quality.

This sweep crosses intended separation with persistence and reports both quantities
alongside oracle-k ARI, so a reader can see exactly where recovery becomes possible.
"""

import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from regimebench.generators.ms_garch import simulate_msgarch
from regimebench.generators.hmm_t import simulate_hmm_t
from regimebench.methods.clustering_wrappers import fit_kmeans
from regimebench.criteria.k_selection import compute_internal_indices
from regimebench.metrics.evaluators import compute_partition_quality, mean_regime_duration

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Intended high/low volatility ratio at k*=2, from barely separated to extreme.
SEPARATIONS = [1.5, 2.0, 2.5, 4.0, 6.0, 10.0]
PERSISTENCES = [0.90, 0.95, 0.99]
REPLICATIONS = 20
T = 5000
K_STAR = 2


def evaluate(gen_name: str, ratio: float, p_ii: float, rep: int):
    seed = 61000 + int(ratio * 100) + 31 * int(p_ii * 100) + rep
    vol = [1.0, ratio]
    if gen_name == "ms_garch":
        y, sigma, states = simulate_msgarch(k=K_STAR, T=T, persistence=p_ii,
                                            vol_ratios=vol, df=5.0, random_state=seed)
    else:
        y, sigma, states = simulate_hmm_t(k=K_STAR, T=T, persistence=p_ii,
                                          stds=vol, df=5.0, random_state=seed)

    sds = [float(np.std(y[states == i])) for i in range(K_STAR) if np.sum(states == i) > 1]
    realized = max(sds) / min(sds) if len(sds) == 2 else np.nan

    X = pd.Series(np.abs(y)).rolling(21, min_periods=1).mean().values.reshape(-1, 1)
    labels_oracle, _ = fit_kmeans(X, K_STAR, random_state=seed)

    sil = {}
    for k in range(2, 7):
        lab, _ = fit_kmeans(X, k, random_state=seed)
        sil[k] = compute_internal_indices(X, lab)['silhouette']
    k_hat_sil = max(sil, key=sil.get)

    return {
        'Generator': gen_name,
        'Intended_Vol_Ratio': ratio,
        'Persistence': p_ii,
        'Mean_Regime_Duration': mean_regime_duration(p_ii),
        'Replication': rep,
        'Realized_Vol_Ratio': realized,
        'Oracle_k_ARI': compute_partition_quality(states, labels_oracle)['ari'],
        'Silhouette_k_hat': k_hat_sil,
        'Silhouette_Exact': int(k_hat_sil == K_STAR),
    }


def run_separation_sweep():
    print("=" * 90)
    print("SEPARATION SWEEP (k*=2)")
    print("=" * 90)
    print(f"  intended ratios: {SEPARATIONS}   persistence: {PERSISTENCES}   reps: {REPLICATIONS}\n")

    tasks = [(g, s, p, r)
             for g in ("ms_garch", "hmm_t")
             for s in SEPARATIONS
             for p in PERSISTENCES
             for r in range(REPLICATIONS)]

    rows = Parallel(n_jobs=-1, batch_size=8, verbose=5)(
        delayed(evaluate)(g, s, p, r) for (g, s, p, r) in tasks
    )
    df = pd.DataFrame(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "separation_sweep.csv")
    df.to_csv(path, index=False)

    print("\n--- Intended vs realized volatility ratio (MS-GARCH) ---")
    ms = df[df.Generator == 'ms_garch']
    print(ms.pivot_table(index='Intended_Vol_Ratio', columns='Persistence',
                         values='Realized_Vol_Ratio', aggfunc='mean').round(2).to_string())

    print("\n--- Oracle-k ARI (MS-GARCH) ---")
    print(ms.pivot_table(index='Intended_Vol_Ratio', columns='Persistence',
                         values='Oracle_k_ARI', aggfunc='mean').round(3).to_string())

    print("\n--- Oracle-k ARI (HMM Student-t) ---")
    hm = df[df.Generator == 'hmm_t']
    print(hm.pivot_table(index='Intended_Vol_Ratio', columns='Persistence',
                         values='Oracle_k_ARI', aggfunc='mean').round(3).to_string())

    print("\n--- Silhouette exact-recovery rate at k*=2 (MS-GARCH) ---")
    print(ms.pivot_table(index='Intended_Vol_Ratio', columns='Persistence',
                         values='Silhouette_Exact', aggfunc='mean').mul(100).round(0).to_string())

    print(f"\nSaved: {path}")
    return df


if __name__ == "__main__":
    run_separation_sweep()
