"""Factor grid experiment: recovery, bias and partition quality across the design.

Sweeps the factor grid declared in ``config/factor_grid_pilot.yaml`` and evaluates:
- K-Means  (Silhouette, Davies-Bouldin, Calinski-Harabasz)  on k in {2..K_max}
- GMM      (BIC, AIC, ICL, BIC_Neff)                        on k in {2..K_max}
- HMM      (BIC, AIC, ICL)                                  on k in {2..K_max}

For the model-based families it *additionally* records the selection over the extended
range k in {1..K_max}. The geometric indices are undefined at k=1, but a one-component
GMM/HMM is perfectly well defined, so restricting the information criteria to k>=2 is a
design choice rather than a structural limit. The ``*_k1`` columns answer the question
that choice otherwise forecloses: would BIC have found the null had it been allowed to?

Records exact recovery, signed bias, ARI at the oracle k*, ARI at the estimated k, and
the realized regime separation actually produced by the generator.
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from joblib import Parallel, delayed

from regimebench.generators.ms_garch import simulate_msgarch
from regimebench.generators.hmm_t import simulate_hmm_t
from regimebench.methods.clustering_wrappers import fit_kmeans, fit_gmm, fit_hmm
from regimebench.criteria.k_selection import compute_internal_indices, compute_information_criteria
from regimebench.metrics.evaluators import compute_partition_quality, compute_rank_inversion, mean_regime_duration

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
FEATURE_WINDOW = 21


def realized_separation(y: np.ndarray, states: np.ndarray, k: int) -> float:
    """Realized volatility ratio actually achieved between the most and least volatile state.

    The design *intends* a separation set by ``volatility_ratios``; GARCH persistence and
    finite samples deliver something narrower. Reporting the realized value is what lets a
    reader judge whether a low ARI measures the pipeline or the generator.
    """
    sds = [float(np.std(y[states == i])) for i in range(k) if np.sum(states == i) > 1]
    if len(sds) < 2:
        return float('nan')
    return float(max(sds) / min(sds))


def evaluate_single_series(args):
    (gen_name, k_true, p_ii, df_tails, T, r, cell_id,
     k_candidates, vol_ratios, base_seed) = args
    seed = base_seed + 1000 * cell_id + r

    if gen_name == "ms_garch":
        y, sigma, true_states = simulate_msgarch(
            k=k_true, T=T, persistence=p_ii, vol_ratios=vol_ratios,
            df=df_tails, random_state=seed
        )
    else:
        y, sigma, true_states = simulate_hmm_t(
            k=k_true, T=T, persistence=p_ii, stds=vol_ratios,
            df=df_tails, random_state=seed
        )

    abs_y = np.abs(y)
    roll_vol = pd.Series(abs_y).rolling(FEATURE_WINDOW, min_periods=1).mean().values.reshape(-1, 1)

    row_base = {
        'Generator': gen_name,
        'True_k': k_true,
        'Persistence': p_ii,
        'DegreesOfFreedom': df_tails,
        'SeriesLength': T,
        'Replication': r,
        'Mean_Regime_Duration': mean_regime_duration(p_ii),
        'Feature_Window': FEATURE_WINDOW,
        'Realized_Vol_Ratio': realized_separation(y, true_states, k_true),
    }

    series_results = []
    k_geo = [k for k in k_candidates if k >= 2]
    k_model = [1] + k_geo

    # 1. K-Means: geometric indices, structurally restricted to k >= 2.
    sil_k, db_k, ch_k, ari_km = {}, {}, {}, {}
    for k in k_geo:
        labels, _ = fit_kmeans(roll_vol, k, random_state=seed)
        ind = compute_internal_indices(roll_vol, labels)
        sil_k[k] = ind['silhouette']
        db_k[k] = ind['davies_bouldin']
        ch_k[k] = ind['calinski_harabasz']
        ari_km[k] = compute_partition_quality(true_states, labels)['ari']

    for crit_name, k_hat in [
        ('Silhouette', max(sil_k, key=sil_k.get)),
        ('Davies-Bouldin', min(db_k, key=db_k.get)),
        ('Calinski-Harabasz', max(ch_k, key=ch_k.get)),
    ]:
        series_results.append({
            **row_base, 'Method': 'K-Means', 'Criterion': crit_name,
            'Estimated_k': k_hat, 'Exact_Match': int(k_hat == k_true),
            'Signed_Bias': k_hat - k_true,
            'ARI_at_True_k': ari_km.get(k_true, 0.0),
            'ARI_at_Estimated_k': ari_km.get(k_hat, 0.0),
            'Estimated_k_with_k1': np.nan, 'Selected_k1': np.nan,
        })

    # 2/3. Model-based families, scored on both k >= 2 and k >= 1.
    for method_name, fit_fn, crits in [
        ('GMM', fit_gmm, ['BIC', 'AIC', 'ICL', 'BIC_Neff']),
        ('HMM', fit_hmm, ['BIC', 'AIC', 'ICL']),
    ]:
        ic_curves = {c: {} for c in crits}
        ari_curve = {}
        for k in k_model:
            labels, info = fit_fn(roll_vol, k, random_state=seed)
            ic = compute_information_criteria(
                info['log_likelihood'], info['num_params'], len(roll_vol), info['posteriors'])
            ic_curves['BIC'][k] = ic['bic']
            ic_curves['AIC'][k] = ic['aic']
            ic_curves['ICL'][k] = ic['icl']
            if 'BIC_Neff' in ic_curves:
                ic_curves['BIC_Neff'][k] = ic['bic_neff']
            ari_curve[k] = compute_partition_quality(true_states, labels)['ari']

        for crit in crits:
            curve = ic_curves[crit]
            restricted = {k: v for k, v in curve.items() if k >= 2}
            k_hat = min(restricted, key=restricted.get)
            k_hat_k1 = min(curve, key=curve.get)
            series_results.append({
                **row_base, 'Method': method_name, 'Criterion': crit,
                'Estimated_k': k_hat, 'Exact_Match': int(k_hat == k_true),
                'Signed_Bias': k_hat - k_true,
                'ARI_at_True_k': ari_curve.get(k_true, 0.0),
                'ARI_at_Estimated_k': ari_curve.get(k_hat, 0.0),
                'Estimated_k_with_k1': k_hat_k1, 'Selected_k1': int(k_hat_k1 == 1),
            })

    return series_results


def run_pilot_grid():
    print("=" * 90)
    print("FACTOR GRID EXPERIMENT")
    print("=" * 90)

    cfg_path = os.path.join(os.path.dirname(__file__), "config", "factor_grid_pilot.yaml")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Read the keys the config file actually declares. Every value below is honoured;
    # a config entry that is silently ignored is worse than no config at all.
    grid = cfg['factor_grid']
    true_k_list = grid['true_k']
    persistence_list = grid['persistence']
    df_list = [grid.get('df', 5.0)]
    generators = grid.get('generators', ['ms_garch', 'hmm_t'])
    vol_ratio_map = {len(v): v for v in grid.get('volatility_ratios', [])}
    series_length_list = [cfg['sample_size_T']]
    reps = cfg['replications_per_cell']
    k_candidates = cfg['k_candidates']
    base_seed = cfg['random_seed']

    print(f"  seed={base_seed}  T={series_length_list}  reps/cell={reps}")
    print(f"  k_candidates (geometric) = {[k for k in k_candidates if k >= 2]}")
    print(f"  k_candidates (model-based, extended) = {[1] + [k for k in k_candidates if k >= 2]}")
    for k, v in sorted(vol_ratio_map.items()):
        print(f"  volatility_ratios[k={k}] = {v}")

    tasks = []
    cell_id = 0
    for gen_name in generators:
        for k_true in true_k_list:
            for p_ii in persistence_list:
                for df_tails in df_list:
                    for T in series_length_list:
                        cell_id += 1
                        vol_ratios = vol_ratio_map.get(k_true)
                        for r in range(reps):
                            tasks.append((gen_name, k_true, p_ii, df_tails, T, r, cell_id,
                                          k_candidates, vol_ratios, base_seed))

    n_series = len(tasks)
    print(f"\n  {n_series} unique series "
          f"({len(generators)} generators x {len(true_k_list)} k* x "
          f"{len(persistence_list)} persistence x {reps} reps)\n")

    parallel_results = Parallel(n_jobs=-1, batch_size=4, verbose=10)(
        delayed(evaluate_single_series)(t) for t in tasks
    )

    results = [item for sublist in parallel_results for item in sublist]
    res_df = pd.DataFrame(results)
    res_df.attrs['n_series'] = n_series

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "pilot_grid_results.csv")
    res_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 90)
    print(f"SUMMARY  ({n_series} series x "
          f"{res_df.groupby(['Method','Criterion']).ngroups} method-criterion pairs = {len(res_df)} rows)")
    print("=" * 90)

    summary = res_df.groupby(['Criterion', 'Method', 'True_k']).agg(
        Exact_Recovery_Rate=('Exact_Match', 'mean'),
        Mean_Signed_Bias=('Signed_Bias', 'mean'),
        Mean_ARI_Oracle_k=('ARI_at_True_k', 'mean'),
        Mean_ARI_Estimated_k=('ARI_at_Estimated_k', 'mean')
    ).reset_index()
    print(summary.to_string(index=False))

    print("\n--- Realized regime separation actually produced by the generators ---")
    sep = res_df.drop_duplicates(
        subset=['Generator', 'True_k', 'Persistence', 'Replication']
    ).groupby(['Generator', 'True_k'])['Realized_Vol_Ratio'].mean()
    print(sep.round(3).to_string())

    print("\n--- Oracle-k ARI stratified by persistence (mean regime duration vs 21-day window) ---")
    ded = res_df.drop_duplicates(subset=['Generator', 'True_k', 'Persistence', 'Replication', 'Method'])
    strat = ded.groupby('Persistence').agg(
        Mean_Duration=('Mean_Regime_Duration', 'first'),
        Oracle_ARI=('ARI_at_True_k', 'mean'))
    print(strat.round(4).to_string())

    print("\n--- Would the information criteria have selected k=1 if allowed? ---")
    k1 = res_df.dropna(subset=['Selected_k1'])
    if not k1.empty:
        print(k1.groupby(['Method', 'Criterion'])['Selected_k1'].mean().mul(100).round(1).to_string())

    rank_inv = compute_rank_inversion(res_df['ARI_at_True_k'].values, res_df['ARI_at_Estimated_k'].values)
    series_level = res_df.groupby(
        ['Generator', 'True_k', 'Persistence', 'Replication']
    )[['ARI_at_True_k', 'ARI_at_Estimated_k']].mean()
    rank_series = compute_rank_inversion(series_level['ARI_at_True_k'].values,
                                         series_level['ARI_at_Estimated_k'].values)
    print(f"\nRank-inversion Spearman rho: row-level rho={rank_inv['spearman_rho']:.4f}, "
          f"series-level rho={rank_series['spearman_rho']:.4f} (p={rank_series['p_value']:.3e})")

    pivot = res_df.pivot_table(index=['True_k', 'Persistence'], columns='Criterion',
                               values='Signed_Bias', aggfunc='mean')
    # Diverging ramp: two hues about a neutral gray midpoint, so zero bias reads as
    # "nothing" and the two directions of error are distinguishable at a glance.
    cmap = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#f0efec", "#c1332f"])
    vmax = float(np.nanmax(np.abs(pivot.values)))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(pivot.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"$k^*$={a}, $P_{{ii}}$={b:.2f}" for a, b in pivot.index], fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha='center', va='center', fontsize=7.5,
                    color='white' if abs(v) > 0.6 * vmax else '#0b0b0b')
    fig.colorbar(im, ax=ax, label=r"$E[\hat k - k^*]$", shrink=0.85)
    ax.set_title(r"Signed bias $E[\hat k - k^*]$ by true $k^*$ and persistence",
                 fontsize=10, fontweight='bold', loc='left')
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(OUT_DIR, f"pilot_bias_heatmap.{ext}"), dpi=300)
    plt.close()

    print("=" * 90)
    print(f"GRID COMPLETE. Saved: {csv_path}")
    print("=" * 90)
    return res_df, summary


if __name__ == "__main__":
    run_pilot_grid()
