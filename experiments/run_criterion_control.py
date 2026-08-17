"""Criterion-level positive control: do the criteria work where they were validated?

The positive control in ``run_positive_control.py`` establishes that the *harness* is
sound -- clustering the latent conditional volatility recovers the true partition
exactly. It says nothing about the criteria themselves, because it hands them the
oracle k. Every criterion number in the benchmark is therefore a failure, and a reader
is entitled to the same objection the manuscript raises against other negative-result
benchmarks: "these criteria fail" is not distinguishable from "this criterion code is
wrong" unless the criteria are shown selecting correctly somewhere.

This experiment supplies that missing control by running the identical criterion code on
the setting the indices were designed and validated for (Milligan & Cooper 1985): i.i.d.
draws from well-separated isotropic Gaussian clusters, no temporal dependence, no
rolling-window smoothing. Nothing changes except the data.

Two blocks, each mirroring a table in the manuscript:

  A. k in {2..12}, the factor-grid range -- K-Means (Silhouette, Davies-Bouldin,
     Calinski-Harabasz), GMM (BIC, AIC, ICL, BIC_Neff), HMM (BIC, AIC, ICL).
  B. k in {1..6}, the harmonized k=1-capable range -- Gap (uniform reference) and
     Prediction Strength.

Separation is reported in the same units as the synthetic grid: the centre distance in
units of the within-cluster standard deviation.
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

TRUE_K = [2, 3, 4]
REPLICATIONS = 30
N_PER_CLUSTER = 400          # 800-1600 points per series, comparable to the k=1 studies
N_FEATURES = 2
SEPARATION = 6.0             # centre spacing in within-cluster standard deviations
K_GEOMETRIC = list(range(2, 13))   # the factor-grid candidate range
K_EXTENDED = list(range(1, 7))     # the harmonized k>=1 range
BASE_SEED = 90000


def simulate_blobs(k: int, seed: int):
    """Well-separated isotropic Gaussian clusters, i.i.d., equal sized.

    Centres are placed on a regular grid along the first axis at ``SEPARATION`` within-
    cluster standard deviations, which is comfortably inside the regime every relative
    validity index was validated on. The observations are exchangeable: there is no
    latent Markov chain, no persistence and no smoothing, so none of the three
    mechanisms the manuscript identifies (dwell time below the feature window,
    compressed realized separation, skew-induced Gap degeneracy) is present.
    """
    rng = np.random.RandomState(seed)
    centres = np.zeros((k, N_FEATURES))
    centres[:, 0] = np.arange(k) * SEPARATION
    labels = np.repeat(np.arange(k), N_PER_CLUSTER)
    X = rng.standard_normal((k * N_PER_CLUSTER, N_FEATURES)) + centres[labels]
    order = rng.permutation(len(labels))
    return X[order], labels[order]


def evaluate_series(k_true: int, rep: int):
    seed = BASE_SEED + 100 * k_true + rep
    X, true_labels = simulate_blobs(k_true, seed)

    base = {"True_k": k_true, "Replication": rep, "Separation": SEPARATION,
            "N_Samples": len(X), "N_Features": N_FEATURES}
    rows = []

    def record(method, criterion, k_hat, k_range, ari):
        rows.append({**base, "Method": method, "Criterion": criterion,
                     "Candidate_Range": f"{min(k_range)}-{max(k_range)}",
                     "Estimated_k": int(k_hat), "Exact_Match": int(k_hat == k_true),
                     "Signed_Bias": int(k_hat) - k_true, "ARI_at_Estimated_k": ari})

    # --- Block A: the factor-grid candidate range -----------------------------------
    sil, db, ch, ari_km = {}, {}, {}, {}
    for k in K_GEOMETRIC:
        labels, _ = fit_kmeans(X, k, random_state=seed)
        ind = compute_internal_indices(X, labels)
        sil[k], db[k], ch[k] = ind["silhouette"], ind["davies_bouldin"], ind["calinski_harabasz"]
        ari_km[k] = compute_partition_quality(true_labels, labels)["ari"]

    for name, k_hat in [("Silhouette", max(sil, key=sil.get)),
                        ("Davies-Bouldin", min(db, key=db.get)),
                        ("Calinski-Harabasz", max(ch, key=ch.get))]:
        record("K-Means", name, k_hat, K_GEOMETRIC, ari_km[k_hat])

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
            ari_curve[k] = compute_partition_quality(true_labels, labels)["ari"]
        for crit in crits:
            k_hat = min(curves[crit], key=curves[crit].get)
            record(method, crit, k_hat, K_GEOMETRIC, ari_curve[k_hat])

    # --- Block B: the harmonized k>=1 range ------------------------------------------
    clus = lambda A, k: fit_kmeans(A, k, random_state=seed)[0]
    k_gap, _ = compute_gap_statistic(X, clus, k_range=range(1, 7), n_bootstraps=10,
                                     random_state=seed, reference="uniform")
    record("K-Means", "Gap (uniform)", k_gap, K_EXTENDED,
           ari_km.get(k_gap, 0.0) if k_gap >= 2 else 0.0)

    k_ps, _ = compute_prediction_strength(X, k_range=range(1, 7), random_state=seed)
    record("K-Means", "Prediction Strength", k_ps, K_EXTENDED,
           ari_km.get(k_ps, 0.0) if k_ps >= 2 else 0.0)

    return rows


def run_criterion_control():
    print("=" * 90)
    print("CRITERION-LEVEL POSITIVE CONTROL: WELL-SEPARATED I.I.D. GAUSSIAN CLUSTERS")
    print("=" * 90)
    print(f"  k* in {TRUE_K}  x  {REPLICATIONS} replications")
    print(f"  {N_PER_CLUSTER} points per cluster, {N_FEATURES}-D, "
          f"centres {SEPARATION:.0f} within-cluster SD apart")
    print(f"  candidate ranges: geometric/model-based {K_GEOMETRIC[0]}-{K_GEOMETRIC[-1]}, "
          f"k>=1-capable {K_EXTENDED[0]}-{K_EXTENDED[-1]}\n")

    tasks = [(k, r) for k in TRUE_K for r in range(REPLICATIONS)]
    nested = Parallel(n_jobs=-1, batch_size=2, verbose=10)(
        delayed(evaluate_series)(k, r) for k, r in tasks)
    df = pd.DataFrame([row for block in nested for row in block])

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "criterion_control_results.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 90)
    print("EXACT RECOVERY BY CRITERION (%)")
    print("=" * 90)
    pivot = df.pivot_table(index=["Method", "Criterion"], columns="True_k",
                           values="Exact_Match", aggfunc="mean").mul(100).round(1)
    print(pivot.to_string())

    print("\nMean selected k-hat:")
    print(df.pivot_table(index=["Method", "Criterion"], columns="True_k",
                         values="Estimated_k", aggfunc="mean").round(2).to_string())

    print(f"\nPooled exact recovery across all criteria and k*: "
          f"{df.Exact_Match.mean() * 100:.1f}%")
    print(f"Mean ARI at the selected k-hat: {df.ARI_at_Estimated_k.mean():.4f}")
    print("=" * 90)
    print(f"COMPLETE. Saved: {csv_path}")
    print("=" * 90)
    return df


if __name__ == "__main__":
    run_criterion_control()
