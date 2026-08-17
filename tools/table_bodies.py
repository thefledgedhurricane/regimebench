"""Prints the LaTeX body rows for each manuscript table, from the CSVs.

This is NOT a manuscript generator. It writes nothing, touches no prose, and never opens
manuscript_array.tex. It prints rows to stdout so you can paste them into the hand-edited
manuscript after re-running an experiment, instead of transcribing numbers by hand.

    python tools/table_bodies.py            # every table
    python tools/table_bodies.py oracle     # one table, by name

`tools/verify_manuscript.py` is what actually enforces agreement: whatever ends up in the
.tex is asserted against these same CSVs, so a paste error or a stale row fails the build.
"""

import os
import sys

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "..", "experiments", "output")


def load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        raise FileNotFoundError(name)
    return pd.read_csv(path)


def factor_grid():
    """Recovery, signed bias and mean k-hat per (method, criterion, k*).

    Mean k-hat is redundant with bias arithmetically -- it is exactly k* + bias -- but it
    is the column the invariance argument in the manuscript is read off, so it stays. The
    Wilson intervals that quantify these rates live in the Uncertainty paragraph, printed
    by ``factor_grid_intervals`` below, rather than in the table.
    """
    df = load("pilot_grid_results.csv")
    lines = []
    for method in ("K-Means", "GMM", "HMM"):
        crits = (["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"] if method == "K-Means"
                 else ["BIC", "BIC_Neff", "AIC", "ICL"])
        for crit in crits:
            sub = df[(df.Method == method) & (df.Criterion == crit)]
            if sub.empty:
                continue
            name = crit.replace("_Neff", r" ($N_{\text{eff}}$)").replace("-", "--")
            row = f"{method} & {name}"
            for k in (2, 3, 4):
                s = sub[sub.True_k == k]
                row += (f" & {s.Exact_Match.mean()*100:.1f}\\% "
                        f"& ${s.Signed_Bias.mean():+.2f}$ "
                        f"& {s.Estimated_k.mean():.2f}")
            lines.append(row + r" \\")
    return lines


def factor_grid_intervals():
    """The Wilson intervals the Uncertainty paragraph quotes, as comments to copy from."""
    from regimebench.metrics.evaluators import wilson_interval

    df = load("pilot_grid_results.csv")
    out = []
    for method, crit in [("K-Means", "Silhouette"), ("GMM", "ICL"), ("GMM", "BIC"),
                         ("K-Means", "Calinski-Harabasz")]:
        cells = []
        for k in (2, 3, 4):
            s = df[(df.Method == method) & (df.Criterion == crit) & (df.True_k == k)]
            w = wilson_interval(int(s.Exact_Match.sum()), len(s))
            cells.append(f"k*={k}: {w['rate']*100:.1f} [{w['lo']*100:.1f}, {w['hi']*100:.1f}]")
        out.append(f"%% {method}/{crit}:  " + "   ".join(cells))
    return out


def oracle():
    df = load("pilot_grid_results.csv")
    g = df.groupby("Method")[["ARI_at_True_k", "ARI_at_Estimated_k"]].mean()
    return [f"{m} & {r.ARI_at_True_k:.4f} & {r.ARI_at_Estimated_k:.4f} \\\\"
            for m, r in g.iterrows()]


def persistence():
    df = load("pilot_grid_results.csv")
    ded = df.drop_duplicates(subset=["Generator", "True_k", "Persistence", "Replication", "Method"])
    lines = []
    for p, sub in ded.groupby("Persistence"):
        ms = sub[sub.Generator == "ms_garch"].ARI_at_True_k.mean()
        hm = sub[sub.Generator == "hmm_t"].ARI_at_True_k.mean()
        lines.append(f"{p:.2f} & {1/(1-p):.0f} & {ms:.4f} & {hm:.4f} & {sub.ARI_at_True_k.mean():.4f} \\\\")
    return lines


def separation():
    df = load("pilot_grid_results.csv")
    s = df.drop_duplicates(subset=["Generator", "True_k", "Persistence", "Replication"])
    intended = {2: 2.5, 3: 5.0, 4: 10.0}
    label = {"ms_garch": "MS-GARCH", "hmm_t": "HMM Student-$t$"}
    lines = []
    for g in ("ms_garch", "hmm_t"):
        for k in (2, 3, 4):
            sub = s[(s.Generator == g) & (s.True_k == k)]
            if sub.empty:
                continue
            cells = " & ".join(f"{sub[sub.Persistence == p].Realized_Vol_Ratio.mean():.2f}"
                               for p in (0.90, 0.95, 0.99))
            lines.append(f"{label[g]} & {k} & {intended[k]:.1f} & {cells} \\\\")
    return lines


def positive():
    df = load("positive_control_results.csv")
    order = ["sigma_true", "sigma_log", "roll_sigma_21", "roll_abs_r_matched", "roll_abs_r_21"]
    label = {
        "sigma_true": r"True $\sigma_t$ (oracle feature)",
        "sigma_log": r"$\log \sigma_t$",
        "roll_sigma_21": r"21-day mean of $\sigma_t$",
        "roll_abs_r_matched": r"$\overline{|r|}$, window matched to dwell time",
        "roll_abs_r_21": r"$\overline{|r|}_{21}$ (standard proxy)",
    }
    g = df.groupby("Feature")[["ARI_KMeans_oracle_k", "ARI_GMM_oracle_k"]].mean()
    byg = df.pivot_table(index="Feature", columns="Generator", values="ARI_KMeans_oracle_k", aggfunc="mean")
    lines = []
    for f in order:
        if f not in g.index:
            continue
        lines.append(f"{label[f]} & {g.loc[f, 'ARI_KMeans_oracle_k']:.4f} "
                     f"& {g.loc[f, 'ARI_GMM_oracle_k']:.4f} "
                     f"& {byg.loc[f, 'hmm_t']:.4f} & {byg.loc[f, 'ms_garch']:.4f} \\\\")
    return lines


def gap_reference():
    df = load("gap_reference_study.csv")
    lines = []
    for c in ["Gap (uniform)", "Gap (gaussian)", "Gap (lognormal)", "Prediction Strength"]:
        sub = df[df.criterion == c]
        if sub.empty:
            continue
        cells = []
        for k in (1, 2, 3, 4):
            s = sub[sub.k_star == k]
            cells.append(f"{s.exact_match.mean()*100:.1f}\\%" if len(s) else "--")
        k1_alt = sub[sub.k_star > 1].selected_k1.mean() * 100
        lines.append(f"{c} & " + " & ".join(cells) + f" & {k1_alt:.1f}\\% \\\\")
    return lines


def null_controls():
    def row(label, sub):
        mode = int(sub.BIC_k_hat.mode().iloc[0])
        return (f"{label} & {len(sub)} & {(sub.Silhouette_k_hat==2).mean()*100:.1f}\\% "
                f"& $\\hat k={mode}$ ({(sub.BIC_k_hat==mode).mean()*100:.1f}\\%) "
                f"& {(sub.Gap_Statistic_k_hat==1).mean()*100:.1f}\\% \\\\")

    df = load("three_null_controls_results.csv")
    lines = [row(nm.split(": ", 1)[-1].replace("Student-t", "Student-$t$"), sub)
             for nm, sub in df.groupby("Null_Model")]
    # The fourth null is a separate artifact (30 reps, k in {2..6}) reported in the same
    # table; the row label must reduce to the same key the verifier builds for it.
    lines.append(row("Stochastic volatility $+$ jumps", load("sv_jump_robustness_results.csv")))
    return lines


def criterion_control():
    df = load("criterion_control_results.csv")
    order = [("K-Means", "Silhouette"), ("K-Means", "Davies-Bouldin"),
             ("K-Means", "Calinski-Harabasz"), ("GMM", "BIC"), ("GMM", "BIC_Neff"),
             ("GMM", "AIC"), ("GMM", "ICL"), ("HMM", "BIC"), ("HMM", "AIC"), ("HMM", "ICL"),
             ("K-Means", "Gap (uniform)"), ("K-Means", "Prediction Strength")]
    lines = []
    for method, crit in order:
        sub = df[(df.Method == method) & (df.Criterion == crit)]
        if sub.empty:
            continue
        name = crit.replace("_Neff", r" ($N_{\text{eff}}$)").replace("Davies-B", "Davies--B") \
                   .replace("Calinski-H", "Calinski--H")
        row = f"{method} & {name}"
        for k in (2, 3, 4):
            s = sub[sub.True_k == k]
            row += f" & {s.Exact_Match.mean()*100:.1f}\\% & {s.Estimated_k.mean():.2f}"
        lines.append(row + r" \\")
        if (method, crit) == ("HMM", "ICL"):
            lines.append(r"\cmidrule(lr){1-8}")   # separates the two candidate ranges
    return lines


FEATURE_LABELS = {
    "rolling_abs_r": r"21-day rolling $|r|$ (baseline)",
    "moments": r"Windowed moments $(\mu,\sigma,\text{skew},\text{kurt})$",
    "raw_returns": "Raw-return windows",
    "4d_vector": "4-D feature vector",
    "log_rv_21": r"$\log \text{RV}_{21}$",
    "har_multiscale": r"HAR multi-scale $\log \text{RV}_{\{1,5,21\}}$",
    "ewma_vol": r"EWMA volatility ($\lambda=0.94$)",
}
# Order of appearance in the manuscript: the four original representations, then the three
# added to answer "a better-chosen standard estimator would have worked".
FEATURE_ORDER = ["rolling_abs_r", "moments", "raw_returns", "4d_vector",
                 "log_rv_21", "har_multiscale", "ewma_vol"]


def feature():
    df = load("feature_grid.csv")
    g = df.groupby("representation")[["sil_ari", "bic_ari", "gap_ari", "oracle_k_ari"]].mean()
    lines = []
    for i, key in enumerate(FEATURE_ORDER):
        if key not in g.index:
            continue
        r = g.loc[key]
        lines.append(f"{FEATURE_LABELS[key]} & {r.sil_ari:.4f} & {r.bic_ari:.4f} "
                     f"& {r.gap_ari:.4f} & {r.oracle_k_ari:.4f} \\\\")
        if i == 3:
            lines.append(r"\cmidrule(lr){1-5}")   # separates original from added
    return lines


ABLATION_ORDER = ["---", "P--", "-K-", "--S", "PK-", "P-S", "-KS", "PKS"]
ABLATION_NAME = {"---": "none (the criterion control)", "P--": "persistence only",
                 "-K-": "skew only", "--S": "smoothing only",
                 "PK-": "persistence $+$ skew", "P-S": "persistence $+$ smoothing",
                 "-KS": "skew $+$ smoothing", "PKS": "all three (the standard pipeline)"}


def mechanism_ablation():
    """One row per corner of the cube, reported per criterion rather than pooled.

    Pooling the three geometric indices would hide the fact that Calinski-Harabasz never
    recovers k in any condition, including the clean one -- which is itself a result.
    """
    df = load("mechanism_ablation_results.csv")
    lines = []
    for cond in ABLATION_ORDER:
        s = df[df.Condition == cond]
        if s.empty:
            continue
        ari = s[(s.Method == "K-Means") & (s.Criterion == "Silhouette")].ARI_at_True_k.mean()
        gap = s[s.Criterion == "Gap (uniform)"]
        cells = " & ".join(f"{s[s.Criterion == c].Exact_Match.mean()*100:.1f}\\%"
                           for c in ("Silhouette", "Davies-Bouldin", "Calinski-Harabasz"))
        lines.append(f"{ABLATION_NAME[cond]} & {ari:.4f} & {cells} "
                     f"& {(gap.Estimated_k == 1).mean()*100:.1f}\\% \\\\")
    return lines


def ablation_effects():
    """Not a manuscript table -- the main effects and the one real interaction."""
    df = load("mechanism_ablation_results.csv")
    geo = df[df.Criterion.isin(["Silhouette", "Davies-Bouldin"])]
    km = df[(df.Method == "K-Means") & (df.Criterion == "Silhouette")]
    out = []
    for label, col in (("persistence", "Persistent"), ("skew", "Skewed"), ("smoothing", "Smoothed")):
        on = geo[geo[col] == 1].Exact_Match.mean() * 100
        off = geo[geo[col] == 0].Exact_Match.mean() * 100
        out.append(f"%% main effect, {label:12s} {off:5.1f} -> {on:5.1f}  ({on-off:+.1f} pp)")
    out.append("%% -- persistence x smoothing on oracle ARI (the dwell-time interaction) --")
    for skew in (0, 1):
        for p in (0, 1):
            v = km[(km.Persistent == p) & (km.Skewed == skew) & (km.Smoothed == 1)].ARI_at_True_k.mean()
            out.append(f"%%   skew={skew} smoothed=1 persistent={p}: ARI {v:.4f}")
    return out


def multivariate():
    df = load("multivariate_test_results.csv")
    label = {"MS-GARCH k*=3 (4-D Features)": r"MS-GARCH $k^*=3$",
             "MS-GARCH k*=4 (4-D Features)": r"MS-GARCH $k^*=4$",
             "Null Single-Regime (k*=1, 4-D Features)": r"Single-cluster null $k^*=1$"}
    lines = []
    for sc in ["MS-GARCH k*=3 (4-D Features)", "MS-GARCH k*=4 (4-D Features)",
               "Null Single-Regime (k*=1, 4-D Features)"]:
        sub = df[df.Scenario == sc]
        if sub.empty:
            continue
        lines.append(f"{label[sc]} & {(sub.Silhouette_k_hat == 2).mean()*100:.1f}\\% "
                     f"& {int(sub.BIC_k_hat.mode().iloc[0])} \\\\")
    return lines


def agreement():
    df = load("real_data_agreement_table.csv")
    strat = [c for c in df.columns if "Strategy" in c][0]
    nber = [c for c in df.columns if "NBER" in c][0]
    bb = [c for c in df.columns if "Bry" in c][0]
    return [f"{r[strat]} & {r[nber]:.4f} & {r[bb]:.4f} \\\\" for _, r in df.iterrows()]


def permutation():
    df = load("permutation_test_results.csv")
    return [f"{r['Partition Strategy']} & {r['Reference']} & {r['Observed ARI']:.4f} "
            f"& {r['Null Mean ARI']:.4f} & {r['Null 95th Percentile']:.4f} "
            f"& {r['Empirical p-value']:.3f} \\\\" for _, r in df.iterrows()]


TABLES = {
    "factor_grid": ("tab:factor_grid", factor_grid),
    "criterion_control": ("tab:criterion_control", criterion_control),
    "factor_grid_intervals": ("(Wilson intervals, comments only)", factor_grid_intervals),
    "oracle": ("tab:oracle", oracle),
    "persistence": ("tab:persistence", persistence),
    "separation": ("tab:separation", separation),
    "positive": ("tab:positive", positive),
    "gap_reference": ("tab:gap_reference", gap_reference),
    "null_controls": ("tab:null_controls", null_controls),
    "feature": ("tab:feature", feature),
    "ablation": ("tab:ablation", mechanism_ablation),
    "ablation_effects": ("(main effects, comments only)", ablation_effects),
    "multivariate": ("tab:multivariate", multivariate),
    "agreement": ("tab:agreement", agreement),
    "permutation": ("tab:permutation", permutation),
}


def main():
    wanted = sys.argv[1:] or list(TABLES)
    for name in wanted:
        if name not in TABLES:
            print(f"unknown table {name!r}; known: {', '.join(TABLES)}")
            continue
        label, fn = TABLES[name]
        print(f"\n%% ---- {name}  ({label}) " + "-" * (54 - len(name) - len(label)))
        try:
            for line in fn():
                print(line)
        except FileNotFoundError as e:
            print(f"%% artifact missing: {e} -- run the pipeline first")


if __name__ == "__main__":
    main()
