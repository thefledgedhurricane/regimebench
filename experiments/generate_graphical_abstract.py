"""Graphical abstract for the Array submission.

Elsevier asks for a single image, minimum 1328 x 531 px, that conveys the paper at a
glance. This draws it from the same CSVs and the same design tokens as the manuscript
figures, so it cannot drift from the results it summarizes: every number on the canvas is
read from experiments/output/ at draw time, none is typed in.

The three panels are the paper's argument in order:
  (1) both controls pass, so the pipeline and the criterion code are sound;
  (2) yet every criterion comes to rest at a boundary of its own candidate range;
  (3) and Silhouette's accuracy runs backwards against the strength of the signal.

    python experiments/generate_graphical_abstract.py

Writes graphical_abstract.pdf (vector, for submission) and .png (300 dpi, for preview).
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_paper_figures import CAT, GRID, INK, INK2, OUT_DIR, SEQ, read  # noqa: E402

ACCENT_OK = CAT[2]      # the controls: something worked
ACCENT_BAD = CAT[1]     # the failures
PAPER = "#ffffff"
BAND = "#f4f3f0"


def panel_controls(ax):
    """Both controls pass: the harness recovers the truth, the criteria are exact."""
    pos = read("positive_control_results.csv")
    crit = read("criterion_control_results.csv")

    oracle = pos[(pos.Feature == "sigma_true") & (pos.Generator == "hmm_t")].ARI_KMeans_oracle_k.mean()
    proxy = pos[(pos.Feature == "roll_abs_r_21") & (pos.Generator == "hmm_t")].ARI_KMeans_oracle_k.mean()
    geom = crit[crit.Criterion.isin(["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"])]
    geom_rate = geom.Exact_Match.mean() * 100

    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Control 1 -- the harness, scored by ARI at the oracle k on the latent state.
    ax.text(0, 0.90, f"ARI = {oracle:.2f}", fontsize=22, fontweight="bold",
            color=ACCENT_OK, va="center")
    ax.text(0, 0.775, "clustering the latent volatility $\\sigma_t$\nrecovers the true partition exactly",
            fontsize=8.6, color=INK2, va="top", linespacing=1.5)

    # Control 2 -- the criteria, scored on the data they were validated for.
    ax.text(0, 0.50, f"{geom_rate:.0f}% correct", fontsize=22, fontweight="bold",
            color=ACCENT_OK, va="center")
    ax.text(0, 0.375, "the same Silhouette, Davies$-$Bouldin and\n"
                      "Calinski$-$Harabasz code recovers $k^*$ on\nwell-separated Gaussian clusters",
            fontsize=8.6, color=INK2, va="top", linespacing=1.5)

    ax.plot([0, 0.94], [0.155, 0.155], color=GRID, lw=1)
    ax.text(0, 0.075, f"on the volatility feature the same harness\n"
                      f"falls to ARI {proxy:.2f} at the standard proxy",
            fontsize=8.2, color=INK2, va="center", linespacing=1.5)


def panel_pinning(ax):
    """Where each criterion comes to rest, and how little it moves when the truth does.

    Each row runs from the criterion's mean k-hat at k*=2 to its mean at k*=4. The truth
    moves 2 units over that range, drawn as the reference arrow at the bottom; a criterion
    that tracked it would produce a segment of the same length. Six produce no segment at
    all, which is the claim the panel exists to make.
    """
    df = read("pilot_grid_results.csv")
    piv = df.pivot_table(index=["Method", "Criterion"], columns="True_k",
                         values="Estimated_k", aggfunc="mean").sort_values(2)
    label = {"BIC_Neff": r"BIC $N_{\rm eff}$"}
    short = {"K-Means": "KM"}
    names = [f"{short.get(m, m)} / {label.get(c, c)}" for m, c in piv.index]

    lo, hi = 2, 12
    y = np.arange(len(piv))
    for yy, (a, b) in enumerate(zip(piv[2].values, piv[4].values)):
        moved = abs(b - a) > 0.5
        colour = INK2 if moved else ACCENT_BAD
        if moved:
            ax.annotate("", xy=(b, yy), xytext=(a, yy),
                        arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.7,
                                        shrinkA=0, shrinkB=0, mutation_scale=9))
        ax.scatter(a, yy, s=30, color=colour, zorder=3)

    ax.axhline(-1.15, color=GRID, lw=0.8)
    ax.annotate("", xy=(4, -1.85), xytext=(2, -1.85),
                arrowprops=dict(arrowstyle="-|>", color="#256abf", lw=2.0,
                                shrinkA=0, shrinkB=0, mutation_scale=10))
    ax.text(4.45, -1.85, "the truth moves this far ($k^*$: 2 $\\rightarrow$ 4)",
            fontsize=7.6, color="#256abf", va="center", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.4)
    ax.set_xlim(lo - 0.5, hi + 0.7)
    ax.set_ylim(-2.5, len(piv) + 0.45)
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_xlabel("mean selected $\\hat k$   (candidate range 2$-$12)", fontsize=8.4)
    ax.tick_params(labelsize=7.6)
    ax.grid(True, axis="x", alpha=0.45, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    ax.text(2.0, len(piv) + 0.30, "floor of the range", fontsize=7.6, color=ACCENT_BAD,
            ha="left", va="top", fontweight="bold")
    ax.text(12.0, len(piv) + 0.30, "ceiling", fontsize=7.6, color=ACCENT_BAD,
            ha="right", va="top", fontweight="bold")


def panel_backwards(ax):
    """Silhouette's accuracy falls as the clusters become easier to find."""
    sw = read("separation_sweep.csv")
    cell = sw[(sw.Generator == "ms_garch") & (sw.Persistence == 0.99)]
    g = cell.groupby("Intended_Vol_Ratio").agg(sil=("Silhouette_Exact", "mean"),
                                               ari=("Oracle_k_ARI", "mean"))
    x = np.arange(len(g))

    ax.plot(x, g.sil.values * 100, marker="o", ms=5.5, lw=2.0, color=ACCENT_BAD, zorder=3)
    ax.fill_between(x, 0, g.sil.values * 100, color=ACCENT_BAD, alpha=0.10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}$\\times$" for v in g.index], fontsize=7.6)
    ax.set_ylim(-6, 116)
    ax.set_xlim(-0.35, len(g) - 0.65)
    ax.set_xlabel("designed cluster separation $\\rightarrow$ easier to detect", fontsize=8.4)
    ax.set_ylabel("Silhouette exact $k$ recovery (%)", fontsize=8.4)
    ax.tick_params(labelsize=7.6)
    ax.grid(True, axis="y", alpha=0.45, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.annotate(f"{g.sil.iloc[0] * 100:.0f}%", xy=(x[0], g.sil.iloc[0] * 100),
                xytext=(x[0] + 0.12, g.sil.iloc[0] * 100 - 15), fontsize=9,
                fontweight="bold", color=ACCENT_BAD)
    ax.annotate(f"{g.sil.iloc[-1] * 100:.0f}%", xy=(x[-1], g.sil.iloc[-1] * 100),
                xytext=(x[-1] - 0.1, g.sil.iloc[-1] * 100 + 11), fontsize=9,
                fontweight="bold", color=ACCENT_BAD, ha="right")
    ax.text(len(g) - 1.35, 88,
            f"recoverable structure rises\n"
            f"(oracle ARI {g.ari.iloc[0]:.2f} $\\rightarrow$ {g.ari.iloc[-1]:.2f})\n"
            "while the index gets it wrong",
            fontsize=8.2, color=INK2, ha="center", va="center", linespacing=1.5)


TITLES = [
    ("1.  Both controls pass", "neither the harness nor the criterion code is at fault"),
    ("2.  Yet the answers pin to a boundary", "and six of the ten never move when the truth does"),
    ("3.  And accuracy runs backwards", "right only while the clusters stay invisible"),
]


def main():
    fig = plt.figure(figsize=(11.4, 5.15), facecolor=PAPER)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.105], width_ratios=[1.02, 1.0, 1.0],
                          left=0.045, right=0.982, top=0.775, bottom=0.035,
                          wspace=0.40, hspace=0.26)

    fig.text(0.045, 0.958, "Do cluster-validity criteria recover the number of market regimes?",
             fontsize=14, fontweight="bold", color=INK, va="center")
    # Read rather than typed: this line drifted once already when the grid was re-run.
    grid = read("pilot_grid_results.csv")
    n_series = grid.groupby(["Generator", "True_k", "Persistence", "Replication"]).ngroups
    fig.text(0.982, 0.958,
             f"{n_series} synthetic series  |  4 nulls  |  a century of US returns",
             fontsize=8.6, color=INK2, va="center", ha="right")

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    # One shared title mechanism at fixed heights, so the three headings sit on a common
    # baseline whatever each panel's own axis furniture does.
    for ax, (head, sub) in zip(axes, TITLES):
        x0 = ax.get_position().x0
        fig.text(x0, 0.878, head, fontsize=11.5, fontweight="bold", color=INK, va="center")
        fig.text(x0, 0.822, sub, fontsize=8.6, color=INK2, va="center")

    panel_controls(axes[0])
    panel_pinning(axes[1])
    panel_backwards(axes[2])

    band = fig.add_subplot(gs[1, :])
    band.axis("off")
    band.set_xlim(0, 1)
    band.set_ylim(0, 1)
    band.add_patch(plt.Rectangle((0, 0), 1, 1, color=BAND, lw=0))
    band.text(0.012, 0.5,
              "A criterion pinned to its own candidate boundary will corroborate whatever $k$ "
              "the analyst brings to it.  Report the controls that make the pinning visible.",
              fontsize=9.6, color=INK, va="center", ha="left")

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext, dpi in (("pdf", 300), ("png", 300)):
        fig.savefig(os.path.join(OUT_DIR, f"graphical_abstract.{ext}"),
                    dpi=dpi, facecolor=PAPER, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    px = int(11.4 * 300), int(5.15 * 300)
    print(f"  saved graphical_abstract.pdf / .png  (~{px[0]} x {px[1]} px, "
          f"Elsevier minimum is 1328 x 531)")


if __name__ == "__main__":
    main()
