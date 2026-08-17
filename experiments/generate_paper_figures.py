"""Generates every figure in the manuscript, from the CSVs in experiments/output/.

Design rules applied (print figures, so no hover/dark-mode layer):
  * categorical hues assigned in fixed order, never cycled -- blue, orange, aqua, violet;
    this 4-slot set validates all-pairs for colour-vision deficiency (worst CVD dE 9.2,
    worst normal-vision dE 16.3) so no chart here needs more than four categories;
  * ordered factors (true k*, persistence, the feature ladder) use a single-hue
    sequential blue ramp, light->dark, never the categorical set;
  * signed bias uses a diverging blue<->red ramp with a neutral gray midpoint;
  * one y-axis per panel, never two scales;
  * a legend whenever two or more series are present, plus direct value labels, so
    identity is never carried by colour alone -- this also discharges the sub-3:1
    surface contrast of the aqua slot;
  * recessive grid and axes, thin marks, text in ink rather than series colour.

Every figure is written as both .png (300 dpi, for drafts) and .pdf (vector, which is
what Elsevier wants for line art and combination artwork -- it sidesteps the 500/1000
dpi raster requirement entirely).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# --- design tokens -----------------------------------------------------------------
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]          # categorical, fixed order
SEQ = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # ordinal blue ramp
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#f0efec", "#c1332f"])

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
})


def seq(n):
    """n evenly spaced steps of the ordinal ramp (never lighter than step 250)."""
    idx = np.linspace(0, len(SEQ) - 1, n).round().astype(int)
    return [SEQ[i] for i in idx]


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  saved {name}.png / {name}.pdf")


def read(name):
    path = os.path.join(OUT_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def tidy(ax, xlab=None, ylab=None, title=None, axis="y"):
    if xlab: ax.set_xlabel(xlab)
    if ylab: ax.set_ylabel(ylab)
    if title: ax.set_title(title, fontweight="bold", loc="left")
    ax.grid(True, axis=axis, alpha=0.5, linewidth=0.6)
    ax.set_axisbelow(True)


# --- figures -----------------------------------------------------------------------

def fig_recovery_rate_comparison():
    df = read("pilot_grid_results.csv")
    if df.empty:
        return
    n_series = df.groupby(['Generator', 'True_k', 'Persistence', 'Replication']).ngroups
    reps = df['Replication'].nunique()
    cells = df.groupby(['Generator', 'True_k', 'Persistence']).ngroups

    s = df.groupby(['Criterion', 'Method', 'True_k'])['Exact_Match'].mean().reset_index()
    s['Label'] = s['Method'] + ' / ' + s['Criterion']
    order = s.groupby('Label')['Exact_Match'].mean().sort_values().index
    ks = sorted(s['True_k'].unique())
    colors = seq(len(ks))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    h = 0.26
    for i, k in enumerate(ks):
        sub = s[s['True_k'] == k].set_index('Label').reindex(order)
        y = np.arange(len(order)) + (i - (len(ks) - 1) / 2) * h
        ax.barh(y, sub['Exact_Match'] * 100, height=h, color=colors[i], label=f"$k^*={k}$")
        for yy, v in zip(y, sub['Exact_Match'] * 100):
            if v > 0:
                ax.text(v + 1.2, yy, f"{v:.0f}", va="center", fontsize=7, color=INK2)

    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlim(0, 105)
    tidy(ax, "Exact recovery rate  $P(\\hat k = k^*)$  (%)", None,
         f"Exact $k$ recovery by method and criterion\n"
         f"{n_series} synthetic series, {cells}-cell factor grid, {reps} replications per cell",
         axis="x")
    ax.legend(title="True regime count", frameon=False, loc="lower right")
    save(fig, "recovery_rate_comparison")


def fig_oracle_ari_by_persistence():
    """The axis the grand mean hides: regime duration against the feature window."""
    df = read("pilot_grid_results.csv")
    if df.empty:
        return
    ded = df.drop_duplicates(subset=['Generator', 'True_k', 'Persistence', 'Replication', 'Method'])
    piv = ded.pivot_table(index='Persistence', columns='Generator',
                          values='ARI_at_True_k', aggfunc='mean')
    gens = {'ms_garch': 'MS-GARCH', 'hmm_t': 'HMM Student-$t$'}
    window = int(df['Feature_Window'].iloc[0]) if 'Feature_Window' in df else 21

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(piv.index))
    w = 0.36
    for i, g in enumerate([c for c in ['ms_garch', 'hmm_t'] if c in piv.columns]):
        vals = piv[g].values
        ax.bar(x + (i - 0.5) * w, vals, w, color=CAT[i], label=gens.get(g, g))
        for xx, v in zip(x + (i - 0.5) * w, vals):
            ax.text(xx, v + 0.008, f"{v:.3f}", ha="center", fontsize=7.5, color=INK2)

    durations = [1 / (1 - p) for p in piv.index]
    ax.set_xticks(x)
    ax.set_xticklabels([f"$P_{{ii}}={p:.2f}$\n(mean dwell {d:.0f} d)" for p, d in zip(piv.index, durations)])
    for xx, d in zip(x, durations):
        if d < window:
            ax.text(xx, ax.get_ylim()[1] * 0.94, "dwell < window", ha="center",
                    fontsize=7, style="italic", color=INK2)
    tidy(ax, None, "Mean ARI at the oracle $k^*$",
         f"Partition quality is governed by regime dwell time relative to the {window}-day feature window")
    ax.legend(frameon=False)
    save(fig, "oracle_ari_by_persistence")


def fig_oracle_ari_by_generator():
    df = read("pilot_grid_results.csv")
    if df.empty:
        return
    piv = df.pivot_table(index='Generator', columns='True_k', values='ARI_at_True_k', aggfunc='mean')
    gens = {'ms_garch': 'MS-GARCH', 'hmm_t': 'HMM Student-$t$'}

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    x = np.arange(len(piv.columns))
    w = 0.36
    for i, g in enumerate(piv.index):
        ax.bar(x + (i - 0.5) * w, piv.loc[g].values, w, color=CAT[i], label=gens.get(g, g))
        for xx, v in zip(x + (i - 0.5) * w, piv.loc[g].values):
            ax.text(xx, v + 0.008, f"{v:.3f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$k^*={c}$" for c in piv.columns])
    tidy(ax, "True regime count", "Mean ARI at the oracle $k^*$",
         "Partition quality at the oracle $k^*$ by generative family")
    ax.legend(frameon=False)
    save(fig, "oracle_ari_by_generator")


def fig_separation_sweep():
    """Intended separation is not delivered separation under MS-GARCH."""
    df = read("separation_sweep.csv")
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5))
    ps = sorted(df['Persistence'].unique())
    colors = seq(len(ps))
    ms = df[df.Generator == 'ms_garch']

    for c, p in zip(colors, ps):
        sub = ms[ms.Persistence == p].groupby('Intended_Vol_Ratio')['Realized_Vol_Ratio'].mean()
        axes[0].plot(sub.index, sub.values, marker="o", ms=4, lw=1.6, color=c,
                     label=f"$P_{{ii}}={p:.2f}$")
    lim = [df['Intended_Vol_Ratio'].min(), df['Intended_Vol_Ratio'].max()]
    axes[0].plot(lim, lim, ls="--", lw=1, color=INK2, label="delivered as designed")
    tidy(axes[0], "Intended volatility ratio", "Realized volatility ratio",
         "(a) MS-GARCH compresses\nthe designed separation", axis="both")
    axes[0].legend(frameon=False, fontsize=7)

    for c, p in zip(colors, ps):
        d = ms[ms.Persistence == p].groupby('Intended_Vol_Ratio').agg(
            r=('Realized_Vol_Ratio', 'mean'), a=('Oracle_k_ARI', 'mean'))
        axes[1].plot(d['r'], d['a'], marker="o", ms=4, lw=1.6, color=c,
                     label=f"$P_{{ii}}={p:.2f}$")
    tidy(axes[1], "Realized volatility ratio", "Mean ARI at the oracle $k^*$",
         "(b) Recovery tracks realized,\nnot intended, separation", axis="both")
    axes[1].legend(frameon=False, fontsize=7)

    # The decisive panel: Silhouette's accuracy runs *backwards* against signal strength.
    for c, p in zip(colors, ps):
        d = ms[ms.Persistence == p].groupby('Intended_Vol_Ratio')['Silhouette_Exact'].mean() * 100
        axes[2].plot(d.index, d.values, marker="o", ms=4, lw=1.6, color=c,
                     label=f"$P_{{ii}}={p:.2f}$")
    axes[2].set_ylim(-4, 104)
    tidy(axes[2], "Intended volatility ratio", "Silhouette exact recovery (%)",
         "(c) Accuracy at $k^*{=}2$ falls as the\nregimes become more detectable", axis="both")
    axes[2].legend(frameon=False, fontsize=7)

    fig.tight_layout()
    save(fig, "separation_sweep")


def fig_criterion_control():
    """The same criterion code on two datasets. This is what licenses every negative result.

    Panel (a) is an exact pairing: identical implementations, identical candidate range
    k in {2..12}, identical number of series (90), pooled over k* in {2,3,4}. The only
    thing that differs between the two marks on a row is the data. Panel (b) isolates the
    Gap statistic's k=1 degeneracy the same way.
    """
    cc, pg = read("criterion_control_results.csv"), read("pilot_grid_results.csv")
    gr = read("gap_reference_study.csv")
    if cc.empty or pg.empty:
        return

    block_a = cc[~cc.Criterion.isin(["Gap (uniform)", "Prediction Strength"])]
    ctrl = block_a.groupby(["Method", "Criterion"])["Exact_Match"].mean().mul(100)
    bench = pg.groupby(["Method", "Criterion"])["Exact_Match"].mean().mul(100)
    paired = pd.DataFrame({"ctrl": ctrl, "bench": bench}).dropna()
    paired["Label"] = [f"{m} / {c}".replace("_Neff", " (N_eff)") for m, c in paired.index]
    paired = paired.sort_values("bench")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax = axes[0]
    y = np.arange(len(paired))
    ax.hlines(y, paired["bench"], paired["ctrl"], color=GRID, lw=2.4, zorder=1)
    ax.scatter(paired["bench"], y, s=42, marker="o", color=CAT[1], zorder=3)
    ax.scatter(paired["ctrl"], y, s=42, marker="s", color=CAT[0], zorder=3)
    for yy, (b, c) in enumerate(zip(paired["bench"], paired["ctrl"])):
        ax.text(b - 2.5, yy, f"{b:.0f}", va="center", ha="right", fontsize=7, color=INK2)
        ax.text(c + 2.5, yy, f"{c:.0f}", va="center", ha="left", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(paired["Label"], fontsize=8)
    ax.set_xlim(-14, 118)
    tidy(ax, "Exact $k$ recovery (%), pooled over $k^*$", None,
         "(a) Identical code, identical candidate range,\n90 series each: only the data differs",
         axis="x")

    ax = axes[1]
    ks = [2, 3, 4]
    if not gr.empty:
        uni = gr[gr.criterion == "Gap (uniform)"]
        vol = [uni[uni.k_star == k].selected_k1.mean() * 100 for k in ks]
    else:
        vol = [np.nan] * len(ks)
    g = cc[cc.Criterion == "Gap (uniform)"]
    ctl = [(g[g.True_k == k].Estimated_k == 1).mean() * 100 for k in ks]

    ax.plot(ks, vol, marker="o", ms=6, lw=1.8, color=CAT[1], label="volatility feature")
    ax.plot(ks, ctl, marker="s", ms=6, lw=1.8, color=CAT[0], label="Gaussian clusters (control)")
    for k, v in zip(ks, vol):
        ax.text(k, v - 7, f"{v:.0f}%", ha="center", fontsize=8, color=INK2)
    for k, v in zip(ks, ctl):
        ax.text(k, v + 4, f"{v:.0f}%", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(ks)
    ax.set_ylim(-10, 112)
    tidy(ax, "True number of clusters $k^*$", r"Rate of answering $\hat k=1$ (%)",
         "(b) The Gap statistic's $\\hat k{=}1$ answer is a\nproperty of the feature, not the estimator")

    # One shared legend below both panels: the two categories are the same in each, and
    # an in-axes box would sit on top of a data point in panel (a) at any corner.
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    handles = [plt.Line2D([], [], marker="o", ls="none", ms=7, color=CAT[1]),
               plt.Line2D([], [], marker="s", ls="none", ms=7, color=CAT[0])]
    fig.legend(handles, ["rolling volatility feature (the benchmark)",
                         "well-separated Gaussian clusters (the control)"],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    save(fig, "criterion_control")


def fig_positive_control():
    """Does the harness recover a partition when one is visible in the feature?"""
    df = read("positive_control_results.csv")
    if df.empty:
        return
    order = ['sigma_true', 'sigma_log', 'roll_sigma_21', 'roll_abs_r_matched', 'roll_abs_r_21']
    label = {
        'sigma_true': r"true $\sigma_t$  (oracle)",
        'sigma_log': r"$\log \sigma_t$",
        'roll_sigma_21': r"21-day mean of $\sigma_t$",
        'roll_abs_r_matched': r"$\overline{|r|}$, window matched to dwell",
        'roll_abs_r_21': r"$\overline{|r|}_{21}$  (standard proxy)",
    }
    g = df.groupby('Feature')['ARI_KMeans_oracle_k'].mean().reindex(order)
    colors = seq(len(order))[::-1]

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ypos = np.arange(len(order))[::-1]
    ax.barh(ypos, g.values, height=0.6, color=colors)
    for y, v in zip(ypos, g.values):
        ax.text(v + 0.012, y, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([label[o] for o in order])
    ax.set_xlim(0, max(g.values) * 1.22)
    tidy(ax, "Mean ARI at the oracle $k^*$", None,
         "Positive control: the harness recovers the partition from the latent volatility\n"
         "and loses it at the standard proxy, localizing the failure to the feature", axis="x")
    save(fig, "positive_control")


def fig_mechanism_ablation():
    """Which of the three mechanisms does the damage, and which only matter in combination.

    Panel (a) is representational: how much of the partition survives in the feature at all,
    scored at the oracle k*. Panel (b) is selection: how often a geometric index then returns
    the right k. Splitting them matters because the two failures have different remedies.
    Conditions are ordered by how many mechanisms are active, so the interaction between
    persistence and smoothing reads off the ordering rather than needing to be asserted.
    """
    df = read("mechanism_ablation_results.csv")
    if df.empty:
        return
    order = ["---", "--S", "-K-", "P--", "-KS", "P-S", "PK-", "PKS"]
    label = {"---": "none", "--S": "smoothing", "-K-": "skew", "P--": "persistence",
             "-KS": "skew + smoothing", "P-S": "persistence + smoothing",
             "PK-": "persistence + skew", "PKS": "all three"}
    order = [c for c in order if c in set(df.Condition)]
    n_on = {c: sum(ch != "-" for ch in c) for c in order}

    # K-Means at the oracle k is the representational measure, matching the table: pooling
    # across methods would fold in the HMM's own misspecification on i.i.d. data.
    km = df[(df.Method == "K-Means") & (df.Criterion == "Silhouette")]
    ari = km.groupby("Condition").ARI_at_True_k.mean().reindex(order)

    ramp = seq(4)
    colours = [ramp[n_on[c]] for c in order]
    y = np.arange(len(order))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), sharey=True)

    axes[0].barh(y, ari.values, height=0.62, color=colours)
    for yy, v in zip(y, ari.values):
        axes[0].text(v + 0.02, yy, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
    axes[0].set_xlim(0, 1.16)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([label[c] for c in order], fontsize=8.5)
    tidy(axes[0], "Mean ARI at the oracle $k^*$ (K-Means)", None,
         "(a) How much structure survives in the feature", axis="x")

    # The three indices are shown separately rather than pooled, because Calinski-Harabasz
    # never recovers k in any condition and pooling would present that as partial success.
    crits = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]
    h = 0.24
    for i, crit in enumerate(crits):
        r = (df[df.Criterion == crit].groupby("Condition").Exact_Match.mean()
             .mul(100).reindex(order))
        yy = y + (1 - i) * h
        axes[1].barh(yy, r.values, height=h, color=CAT[i], label=crit.replace("-", "–"))
        for a, v in zip(yy, r.values):
            axes[1].text(v + 1.5, a, f"{v:.0f}", va="center", fontsize=6.6, color=INK2)
    axes[1].set_xlim(0, 118)
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    tidy(axes[1], "Exact $k$ recovery (%)", None,
         "(b) How often each index then returns the right $k$", axis="x")

    fig.tight_layout()
    save(fig, "mechanism_ablation")


def fig_gap_curve_diagnostic():
    """Why the Gap statistic answers k=1: the curve, under each null model."""
    df = read("gap_curve_diagnostic.csv")
    if df.empty:
        return
    refs = ['uniform', 'gaussian', 'lognormal']
    names = {'uniform': "uniform box\n(Tibshirani method a)",
             'gaussian': "Gaussian\n(single-regime null)",
             'lognormal': "lognormal\n(volatility-matched null)"}

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.4), sharey=True)
    kstars = sorted(df['k_star'].unique())
    colors = seq(len(kstars))
    for ax, ref in zip(axes, refs):
        sub = df[df.reference == ref]
        for c, ks in zip(colors, kstars):
            m = sub[sub.k_star == ks].groupby('k')['gap'].mean()
            ax.plot(m.index, m.values, marker="o", ms=3.5, lw=1.6, color=c, label=f"$k^*={ks}$")
        ax.axvline(1, color=INK2, lw=0.8, ls=":")
        tidy(ax, "candidate $k$", "mean Gap$(k)$" if ax is axes[0] else None,
             names[ref], axis="both")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("The one-standard-error rule can only return the smallest candidate when the curve peaks at $k=1$",
                 fontsize=10, fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "gap_curve_diagnostic")


def fig_gap_reference_selection():
    """What each k=1-capable criterion selects, by true k*, under one harmonized design."""
    df = read("gap_reference_study.csv")
    if df.empty:
        return
    crits = [c for c in ["Gap (uniform)", "Gap (gaussian)", "Gap (lognormal)", "Prediction Strength"]
             if c in set(df.criterion)]

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6))
    ks = sorted(df['k_star'].unique())
    for i, c in enumerate(crits):
        sub = df[df.criterion == c]
        r1 = sub.groupby('k_star')['selected_k1'].mean().reindex(ks) * 100
        rec = sub.groupby('k_star')['exact_match'].mean().reindex(ks) * 100
        axes[0].plot(ks, r1.values, marker="o", ms=4, lw=1.6, color=CAT[i], label=c)
        axes[1].plot(ks, rec.values, marker="o", ms=4, lw=1.6, color=CAT[i], label=c)

    for ax, ttl, yl in [(axes[0], "Selects $\\hat k = 1$", "$P(\\hat k = 1)$  (%)"),
                        (axes[1], "Selects the truth", "$P(\\hat k = k^*)$  (%)")]:
        ax.set_xticks(ks)
        ax.set_ylim(-4, 104)
        tidy(ax, "True regime count $k^*$  ($k^*{=}1$ is the null)", yl, ttl)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("A criterion that answers $\\hat k=1$ everywhere is not detecting the null",
                 fontsize=10, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "gap_reference_selection")


def fig_boundary_pinning():
    """The thesis figure: where in its candidate range each criterion comes to rest."""
    df = read("pilot_grid_results.csv")
    if df.empty:
        return
    k_lo, k_hi = 2, int(df['Estimated_k'].max())
    s = df.groupby(['Method', 'Criterion'])['Estimated_k'].agg(['mean', 'std']).reset_index()
    s['Label'] = (s['Method'] + ' / ' + s['Criterion']).str.replace(
        'BIC_Neff', r'BIC ($N_{eff}$)', regex=False)
    s = s.sort_values('mean')

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    y = np.arange(len(s))

    ax.axvspan(2, 4, color=CAT[2], alpha=0.10, zorder=0)
    ax.axvspan(k_lo - 0.45, k_lo + 0.45, color="#e3e2de", zorder=0)
    ax.axvspan(k_hi - 0.45, k_hi + 0.45, color="#e3e2de", zorder=0)

    ax.errorbar(s['mean'], y, xerr=s['std'], fmt="o", ms=5, lw=1.2, capsize=2.5,
                color=CAT[0], ecolor=INK2, elinewidth=0.9, zorder=3)
    for yy, m in zip(y, s['mean']):
        ax.text(m, yy + 0.3, f"{m:.1f}", ha="center", fontsize=7.5, color=INK2, zorder=4)

    # Extra headroom so the boundary annotations clear both the title and the top row's
    # value label, which sits at len(s)-1+0.3.
    ax.set_ylim(-0.7, len(s) + 0.35)
    ax.text(k_lo, len(s) - 0.05, "candidate floor", ha="center", va="center", fontsize=7,
            color=INK2, style="italic")
    ax.text(k_hi, len(s) - 0.05, "candidate ceiling", ha="center", va="center", fontsize=7,
            color=INK2, style="italic")
    ax.text(3, -0.55, "range of true $k^*$", ha="center", va="center", fontsize=7.5,
            color=INK2, style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(s['Label'])
    ax.set_xlim(k_lo - 0.9, k_hi + 0.9)
    ax.set_xticks(range(k_lo, k_hi + 1))
    tidy(ax, "Selected $\\hat k$  (mean $\\pm$ 1 SD across the grid)", None,
         "Every criterion comes to rest at a boundary of its own candidate range,\n"
         "not at the true regime count", axis="x")
    save(fig, "boundary_pinning")


def fig_feature_representation_ari():
    df = read("feature_grid.csv")
    if df.empty:
        return
    g = df.groupby('representation')['oracle_k_ari'].mean().sort_values()
    label = {'rolling_abs_r': r"21-day rolling $|r|$", 'moments': "windowed moments",
             'raw_returns': "raw-return windows", '4d_vector': "4-D feature vector"}
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    y = np.arange(len(g))
    ax.barh(y, g.values, height=0.6, color=seq(len(g)))
    for yy, v in zip(y, g.values):
        ax.text(v + max(g.values) * 0.02, yy, f"{v:.4f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([label.get(i, i) for i in g.index])
    ax.set_xlim(0, max(g.values) * 1.25)
    tidy(ax, "Mean ARI at the oracle $k^*$", None,
         "No standard representation recovers the partition ($k^*=3$)", axis="x")
    save(fig, "feature_representation_ari")


def fig_null_control_distributions():
    df = read("three_null_controls_results.csv")
    if df.empty:
        return
    melted = df.melt(id_vars=['Null_Model', 'Replication'],
                     value_vars=['Silhouette_k_hat', 'Gap_Statistic_k_hat'],
                     var_name='Criterion', value_name='k_hat')
    melted['Criterion'] = melted['Criterion'].str.replace('_k_hat', '', regex=False).str.replace('_', ' ')
    melted['Null_Model'] = melted['Null_Model'].str.replace(r'^Null \d: ', '', regex=True)

    sv = read("sv_jump_robustness_results.csv")
    if not sv.empty:
        svm = sv.melt(id_vars=['Replication'],
                      value_vars=['Silhouette_k_hat', 'Gap_Statistic_k_hat'],
                      var_name='Criterion', value_name='k_hat')
        svm['Criterion'] = svm['Criterion'].str.replace('_k_hat', '', regex=False).str.replace('_', ' ')
        svm['Null_Model'] = 'SV + jump (continuous)'
        melted = pd.concat([melted, svm], ignore_index=True)

    names = list(melted['Null_Model'].unique())
    fig, axes = plt.subplots(1, len(names), figsize=(2.0 * len(names) + 0.8, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    crits = sorted(melted['Criterion'].unique())
    for ax, nm in zip(axes, names):
        sub = melted[melted['Null_Model'] == nm]
        n_rep = sub.groupby('Criterion')['k_hat'].size().max()
        counts = sub.groupby(['Criterion', 'k_hat']).size().reset_index(name='n')
        counts['pct'] = counts.groupby('Criterion')['n'].transform(lambda s: 100 * s / s.sum())
        ks = sorted(counts['k_hat'].unique())
        w = 0.38
        for i, c in enumerate(crits):
            cc = counts[counts.Criterion == c].set_index('k_hat').reindex(ks).fillna(0)
            ax.bar(np.arange(len(ks)) + (i - 0.5) * w, cc['pct'], w, color=CAT[i], label=c)
        ax.set_xticks(np.arange(len(ks)))
        ax.set_xticklabels(ks)
        ax.axvline(-0.5, color=INK2, lw=0.8, ls="--")
        tidy(ax, "selected $\\hat k$", "share of replications (%)" if ax is axes[0] else None,
             f"{nm}\n($n={int(n_rep)}$)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Selected $\\hat k$ on null processes with no regime structure (truth is $\\hat k=1$)",
                 fontsize=10, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "null_control_distributions")


def fig_binance_regime_overlay():
    from regimebench.data.binance import fetch_binance_klines
    from regimebench.methods.clustering_wrappers import fit_kmeans

    df = fetch_binance_klines(symbol="BTCUSDT", year=2024, month=1)
    X = df['Realized_Vol_60m'].values.reshape(-1, 1)
    labels, _ = fit_kmeans(X, k=2, random_state=42)
    hi = int(np.argmax([X[labels == c].mean() for c in np.unique(labels)]))
    high_vol = (labels == hi).astype(int)
    share = 100 * high_vol.mean()

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.6), sharex=True)
    axes[0].plot(df.index, df['Close'], color=CAT[0], lw=0.6, label="BTCUSDT close (1-min)")
    axes[0].fill_between(df.index, *axes[0].get_ylim(), where=(high_vol == 1),
                         color=CAT[1], alpha=0.18, transform=axes[0].get_xaxis_transform(),
                         label=f"$\\hat k=2$ high-volatility state ({share:.0f}% of bars)")
    tidy(axes[0], None, "Price (USDT)",
         f"BTCUSDT, January 2024, {len(df):,} 1-minute bars, with the $\\hat k=2$ partition")
    axes[0].legend(frameon=False, loc="upper left", fontsize=7)

    axes[1].plot(df.index, df['Realized_Vol_60m'], color=CAT[3], lw=0.5)
    tidy(axes[1], "Date", "60-min realized vol\n(annualized)")
    fig.tight_layout()
    save(fig, "binance_regime_overlay")


FIGURES = [
    ("recovery rate", fig_recovery_rate_comparison),
    ("oracle ARI by persistence", fig_oracle_ari_by_persistence),
    ("oracle ARI by generator", fig_oracle_ari_by_generator),
    ("separation sweep", fig_separation_sweep),
    ("positive control", fig_positive_control),
    ("criterion control", fig_criterion_control),
    ("mechanism ablation", fig_mechanism_ablation),
    ("gap curve diagnostic", fig_gap_curve_diagnostic),
    ("gap reference selection", fig_gap_reference_selection),
    ("boundary pinning", fig_boundary_pinning),
    ("feature representation", fig_feature_representation_ari),
    ("null controls", fig_null_control_distributions),
    ("binance overlay", fig_binance_regime_overlay),
]


def main():
    print("=" * 90)
    print("GENERATING MANUSCRIPT FIGURES (png + vector pdf)")
    print("=" * 90)
    failed = []
    for name, fn in FIGURES:
        try:
            fn()
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"  [SKIP] {name}: {type(e).__name__}: {e}")
    if failed:
        print("\nFigures not produced (missing upstream artifact?):")
        for n, e in failed:
            print(f"  - {n}: {e}")
    print("=" * 90)


if __name__ == "__main__":
    main()
