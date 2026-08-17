"""Mechanism ablation: which of the three failure mechanisms actually does the damage?

The paper attributes the collapse to three things -- state dwell time short relative to the
feature window, a right-skewed heavy-tailed marginal, and rolling-window smoothing -- and
until now argued for each separately. This experiment measures them jointly, by starting
from the setting in which the criteria are exact (``run_criterion_control.py``) and adding
the three mechanisms back one at a time in a full 2x2x2 factorial.

Every condition shares one latent design: k*=3 states with volatility scales (1, 2.5, 5),
T=2000, 30 replications. The factors are:

  PERSISTENCE  off: the state sequence is i.i.d. uniform over the three states.
               on : a symmetric Markov chain with P_ii = 0.95, i.e. mean dwell time 20,
                    which sits just below the 21-observation feature window.

  SKEW         off: the observable is the scale itself plus small symmetric noise, so the
                    three states are well-separated symmetric clusters in feature space.
               on : the observable is |r_t| with r_t = sigma_{s_t} * standardized t(5),
                    which is the right-skewed heavy-tailed quantity analysts actually have.

  SMOOTHING    off: the feature is the observable.
               on : the feature is its 21-observation rolling mean, the standard proxy.

The (off, off, off) corner is the criterion control in one dimension; the (on, on, on)
corner is the benchmark's standard pipeline. What lies between is the decomposition.

Note that K-Means, GMM and the geometric indices are permutation-invariant: they cannot
see the order of the observations. Persistence therefore cannot affect them directly and
can only act through its interaction with the window, which is precisely the claim the
factorial is built to test.
"""

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from regimebench.criteria.k_selection import (
    compute_gap_statistic,
    compute_information_criteria,
    compute_internal_indices,
    compute_prediction_strength,
)
from regimebench.methods.clustering_wrappers import fit_gmm, fit_hmm, fit_kmeans
from regimebench.metrics.evaluators import compute_partition_quality

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")

K_STAR = 3
SCALES = np.array([1.0, 2.5, 5.0])
T = 2000
REPLICATIONS = 30
PERSISTENCE = 0.95            # mean dwell 20, against a 21-observation window
WINDOW = 21
DF = 5.0
K_GEOMETRIC = list(range(2, 13))
K_EXTENDED = list(range(1, 7))
BASE_SEED = 70000


def draw_states(rng, persistent: bool) -> np.ndarray:
    """The latent state path, with or without temporal persistence."""
    if not persistent:
        return rng.randint(0, K_STAR, size=T)
    off = (1.0 - PERSISTENCE) / (K_STAR - 1)
    states = np.empty(T, dtype=int)
    states[0] = rng.randint(0, K_STAR)
    for t in range(1, T):
        p = np.full(K_STAR, off)
        p[states[t - 1]] = PERSISTENCE
        states[t] = rng.choice(K_STAR, p=p)
    return states


def build_feature(rng, states, skewed: bool, smoothed: bool) -> np.ndarray:
    """The observable, then the feature the analyst actually clusters."""
    scale = SCALES[states]
    if skewed:
        # What an analyst has: the absolute return of a heavy-tailed process. Right-skewed
        # by construction, which is what makes the Gap statistic's uniform reference fail.
        eps = rng.standard_t(DF, size=T) / np.sqrt(DF / (DF - 2.0))
        x = np.abs(scale * eps)
    else:
        # The idealized case: the scale itself, symmetric noise, clusters well separated.
        x = scale + rng.standard_normal(T) * 0.15
    if smoothed:
        x = pd.Series(x).rolling(WINDOW, min_periods=1).mean().values
    return x.reshape(-1, 1)


def evaluate(persistent: bool, skewed: bool, smoothed: bool, rep: int):
    cond = f"{'P' if persistent else '-'}{'K' if skewed else '-'}{'S' if smoothed else '-'}"
    seed = BASE_SEED + 1000 * (persistent * 4 + skewed * 2 + smoothed) + rep
    rng = np.random.RandomState(seed)

    states = draw_states(rng, persistent)
    X = build_feature(rng, states, skewed, smoothed)

    base = {"Condition": cond, "Persistent": int(persistent), "Skewed": int(skewed),
            "Smoothed": int(smoothed), "Replication": rep, "True_k": K_STAR}
    rows = []

    def record(method, criterion, k_hat, ari_at_k_hat, ari_oracle):
        rows.append({**base, "Method": method, "Criterion": criterion,
                     "Estimated_k": int(k_hat), "Exact_Match": int(k_hat == K_STAR),
                     "Signed_Bias": int(k_hat) - K_STAR,
                     "ARI_at_Estimated_k": ari_at_k_hat, "ARI_at_True_k": ari_oracle})

    sil, db, ch, ari_km = {}, {}, {}, {}
    for k in K_GEOMETRIC:
        labels, _ = fit_kmeans(X, k, random_state=seed)
        ind = compute_internal_indices(X, labels)
        sil[k], db[k], ch[k] = ind["silhouette"], ind["davies_bouldin"], ind["calinski_harabasz"]
        ari_km[k] = compute_partition_quality(states, labels)["ari"]

    for name, k_hat in [("Silhouette", max(sil, key=sil.get)),
                        ("Davies-Bouldin", min(db, key=db.get)),
                        ("Calinski-Harabasz", max(ch, key=ch.get))]:
        record("K-Means", name, k_hat, ari_km[k_hat], ari_km[K_STAR])

    for method, fit_fn, crits in [("GMM", fit_gmm, ["BIC", "AIC", "ICL", "BIC_Neff"]),
                                  ("HMM", fit_hmm, ["BIC", "AIC", "ICL"])]:
        curves = {c: {} for c in crits}
        ari_curve = {}
        for k in K_GEOMETRIC:
            labels, info = fit_fn(X, k, random_state=seed)
            ic = compute_information_criteria(info["log_likelihood"], info["num_params"],
                                              len(X), info["posteriors"])
            curves["BIC"][k], curves["AIC"][k], curves["ICL"][k] = ic["bic"], ic["aic"], ic["icl"]
            if "BIC_Neff" in curves:
                curves["BIC_Neff"][k] = ic["bic_neff"]
            ari_curve[k] = compute_partition_quality(states, labels)["ari"]
        for crit in crits:
            k_hat = min(curves[crit], key=curves[crit].get)
            record(method, crit, k_hat, ari_curve[k_hat], ari_curve[K_STAR])

    clus = lambda A, k: fit_kmeans(A, k, random_state=seed)[0]
    k_gap, _ = compute_gap_statistic(X, clus, k_range=range(1, 7), n_bootstraps=10,
                                     random_state=seed, reference="uniform")
    record("K-Means", "Gap (uniform)", k_gap,
           ari_km.get(k_gap, 0.0) if k_gap >= 2 else 0.0, ari_km[K_STAR])

    k_ps, _ = compute_prediction_strength(X, k_range=range(1, 7), random_state=seed)
    record("K-Means", "Prediction Strength", k_ps,
           ari_km.get(k_ps, 0.0) if k_ps >= 2 else 0.0, ari_km[K_STAR])

    return rows


def run_mechanism_ablation():
    print("=" * 90)
    print("MECHANISM ABLATION: PERSISTENCE x SKEW x SMOOTHING")
    print("=" * 90)
    print(f"  k*={K_STAR}, scales {tuple(SCALES)}, T={T}, {REPLICATIONS} reps per condition")
    print(f"  persistence P_ii={PERSISTENCE} (mean dwell {1/(1-PERSISTENCE):.0f}) "
          f"against a {WINDOW}-observation window")
    print("  8 conditions x 30 reps = 240 series, 12 method-criterion pairs each\n")

    tasks = [(p, k, s, r)
             for p in (False, True) for k in (False, True) for s in (False, True)
             for r in range(REPLICATIONS)]
    nested = Parallel(n_jobs=-1, batch_size=2, verbose=10)(
        delayed(evaluate)(p, k, s, r) for p, k, s, r in tasks)
    df = pd.DataFrame([row for block in nested for row in block])

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "mechanism_ablation_results.csv")
    df.to_csv(csv_path, index=False)

    order = ["---", "--S", "-K-", "-KS", "P--", "P-S", "PK-", "PKS"]
    geo = df[df.Criterion.isin(["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"])]
    gap = df[df.Criterion == "Gap (uniform)"]

    print("\n" + "=" * 90)
    print("EXACT k RECOVERY BY CONDITION  (P=persistent, K=skewed, S=smoothed)")
    print("=" * 90)
    summary = pd.DataFrame({
        "geometric_recovery_%": geo.groupby("Condition").Exact_Match.mean().mul(100),
        "all_criteria_recovery_%": df.groupby("Condition").Exact_Match.mean().mul(100),
        "oracle_ARI": df.groupby("Condition").ARI_at_True_k.mean(),
        "gap_k1_rate_%": gap.groupby("Condition").apply(
            lambda s: (s.Estimated_k == 1).mean() * 100, include_groups=False),
    }).reindex(order)
    print(summary.round(2).to_string())

    print("\nMain effects on geometric exact recovery (percentage points):")
    for factor, col in (("persistence", "Persistent"), ("skew", "Skewed"), ("smoothing", "Smoothed")):
        on = geo[geo[col] == 1].Exact_Match.mean() * 100
        off = geo[geo[col] == 0].Exact_Match.mean() * 100
        print(f"  {factor:12s} {off:6.1f} -> {on:6.1f}   ({on - off:+.1f})")

    print("\nSmoothing effect, split by whether the state persists:")
    for p in (0, 1):
        sub = geo[geo.Persistent == p]
        off = sub[sub.Smoothed == 0].Exact_Match.mean() * 100
        on = sub[sub.Smoothed == 1].Exact_Match.mean() * 100
        label = "persistent (dwell 20)" if p else "i.i.d. states"
        print(f"  {label:24s} {off:6.1f} -> {on:6.1f}   ({on - off:+.1f})")

    print("=" * 90)
    print(f"COMPLETE. Saved: {csv_path}")
    print("=" * 90)
    return df


if __name__ == "__main__":
    run_mechanism_ablation()
