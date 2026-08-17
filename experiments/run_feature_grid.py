import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from regimebench.generators.ms_garch import simulate_msgarch
from regimebench.criteria.k_selection import compute_internal_indices, compute_gap_statistic

def run_feature_grid():
    print("=" * 90)
    print("RUNNING FEATURE REPRESENTATION GRID EXPERIMENT (CROSSING 7 REPRESENTATIONS)")
    print("=" * 90)

    k_star = 3
    p = 0.95
    replications = 30
    seed = 42
    
    results = []
    
    for i in range(replications):
        rep_seed = seed + i * 100
        returns, sigma, true_states = simulate_msgarch(
            k=k_star,
            T=2000,
            persistence=p,
            df=5.0,
            random_state=rep_seed
        )
        
        reps = {}
        s = pd.Series(returns)
        
        # (a) 21-day rolling |r|
        feat_a = s.abs().rolling(window=21, min_periods=1).mean().values.reshape(-1, 1)
        states_a = true_states
        reps['rolling_abs_r'] = (feat_a, states_a)
        
        # (b) Windowed moment vector
        feat_b_mean = s.rolling(window=21, min_periods=1).mean()
        feat_b_std = s.rolling(window=21, min_periods=1).std().fillna(0)
        feat_b_skew = s.rolling(window=21, min_periods=1).apply(lambda x: pd.Series(x).skew(), raw=True).fillna(0)
        feat_b_kurt = s.rolling(window=21, min_periods=1).apply(lambda x: pd.Series(x).kurt(), raw=True).fillna(0)
        feat_b = pd.concat([feat_b_mean, feat_b_std, feat_b_skew, feat_b_kurt], axis=1).values
        states_b = true_states
        reps['moments'] = (feat_b, states_b)
        
        # (c) Raw return windows
        n_blocks = len(returns) // 21
        feat_c = returns[:n_blocks*21].reshape(-1, 21)
        states_c = true_states[:n_blocks*21:21]
        reps['raw_returns'] = (feat_c, states_c)
        
        # (d) 4-D vector [r_t, Vol_21d, |r|_21d, r²_21d]
        feat_d_vol = s.rolling(window=21, min_periods=1).std().fillna(0)
        feat_d_abs = s.abs().rolling(window=21, min_periods=1).mean().fillna(0)
        feat_d_sq = (s**2).rolling(window=21, min_periods=1).mean().fillna(0)
        feat_d = pd.concat([s, feat_d_vol, feat_d_abs, feat_d_sq], axis=1).values
        states_d = true_states
        reps['4d_vector'] = (feat_d, states_d)

        # The four representations above are the ones the manuscript originally reported.
        # The three below answer the obvious objection to a negative feature result -- that
        # a better-chosen standard estimator would have worked. Each is realizable from the
        # return series alone, so none of them is an oracle.

        # (e) log realized volatility. The positive control shows log(sigma_t) separates the
        # states better than sigma_t does, because the state scales are geometrically spaced;
        # this is the realizable analogue of that transform.
        rv21 = (s**2).rolling(window=21, min_periods=1).mean()
        feat_e = np.log(rv21.clip(lower=1e-12)).values.reshape(-1, 1)
        reps['log_rv_21'] = (feat_e, true_states)

        # (f) HAR-style multi-scale volatility: daily, weekly and monthly realized variance
        # standardized to unit variance. The standard answer to "your window is wrong" is to
        # supply several windows at once and let the clustering choose.
        har = pd.concat([(s**2).rolling(w, min_periods=1).mean() for w in (1, 5, 21)], axis=1)
        har = np.log(har.clip(lower=1e-12))
        feat_f = ((har - har.mean()) / har.std().replace(0, 1)).values
        reps['har_multiscale'] = (feat_f, true_states)

        # (g) RiskMetrics EWMA volatility, lambda = 0.94: an exponentially weighted estimator
        # with no fixed window at all, which is what a practitioner would actually run.
        ewma_var = (s**2).ewm(alpha=1 - 0.94, adjust=False).mean()
        feat_g = np.sqrt(ewma_var.clip(lower=1e-24)).values.reshape(-1, 1)
        reps['ewma_vol'] = (feat_g, true_states)

        for rep_name, (X, target_states) in reps.items():
            best_sil_k = -1
            best_sil = -2
            best_bic_k = -1
            best_bic = float('inf')
            
            sil_aris = {}
            bic_aris = {}
            
            clustering_func = lambda data, k_val: KMeans(n_clusters=k_val, random_state=rep_seed, n_init=3).fit_predict(data)
            
            for k in range(2, 7):
                kmeans = KMeans(n_clusters=k, random_state=rep_seed, n_init=3)
                labels = kmeans.fit_predict(X)
                
                ari = adjusted_rand_score(target_states, labels)
                sil_aris[k] = ari
                
                indices = compute_internal_indices(X, labels)
                sil = indices['silhouette']
                if sil > best_sil:
                    best_sil = sil
                    best_sil_k = k
                    
                gmm = GaussianMixture(n_components=k, random_state=rep_seed)
                gmm.fit(X)
                bic = gmm.bic(X)
                if bic < best_bic:
                    best_bic = bic
                    best_bic_k = k
                    bic_aris[k] = adjusted_rand_score(target_states, gmm.predict(X))
            
            gap_k, _ = compute_gap_statistic(X, clustering_func, k_range=range(2, 7), random_state=rep_seed)
            kmeans_gap = KMeans(n_clusters=gap_k, random_state=rep_seed, n_init=3)
            gap_labels = kmeans_gap.fit_predict(X)
            gap_ari = adjusted_rand_score(target_states, gap_labels)
            
            kmeans_oracle = KMeans(n_clusters=3, random_state=rep_seed, n_init=3)
            oracle_labels = kmeans_oracle.fit_predict(X)
            oracle_ari = adjusted_rand_score(target_states, oracle_labels)
            
            results.append({
                'rep': i,
                'representation': rep_name,
                'sil_k': best_sil_k,
                'bic_k': best_bic_k,
                'gap_k': gap_k,
                'sil_ari': sil_aris.get(best_sil_k, 0.0),
                'bic_ari': bic_aris.get(best_bic_k, 0.0),
                'gap_ari': gap_ari,
                'oracle_k_ari': oracle_ari
            })

        print(f"  ✓ Feature replication {i+1}/{replications} evaluated across 7 representations")

    df = pd.DataFrame(results)
    
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "feature_grid.csv"
    df.to_csv(csv_path, index=False)
    
    summary = df.groupby('representation')[['sil_k', 'bic_k', 'gap_k', 'sil_ari', 'bic_ari', 'gap_ari', 'oracle_k_ari']].mean()
    print("\n" + "=" * 90)
    print("FEATURE REPRESENTATION GRID SUMMARY (Mean across replications):")
    print("=" * 90)
    print(summary.to_string())
    print(f"\nSaved CSV: {csv_path}")

if __name__ == "__main__":
    run_feature_grid()
