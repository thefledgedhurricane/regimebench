"""Manuscript integrity verifier.

The manuscript LaTeX is hand-written, not generated. This script is what keeps it
honest: it parses every number out of `manuscript/manuscript_array.tex` and asserts it
against the CSV that produced it, failing loudly on any drift.

It checks three things:

  1. TABLES   -- each registered table is rebuilt from its source CSV and compared to the
                 typeset rows cell by cell, keyed by the row label so a reordering cannot
                 hide a mismatch.
  2. PROSE    -- scalar claims in the running text (headline ARI, sample sizes, rates)
                 are matched by regex and compared to the recomputed value.
  3. STRUCTURE-- submission blockers that are cheap to check and expensive to miss:
                 unescaped %, placeholder DOIs, missing \\linenumbers, figures referenced
                 but absent from disk, and citations that do not resolve.

Self-test:  python tools/verify_manuscript.py --self-test
            perturbs one CSV value in memory and confirms the verifier reports a failure.
            A verifier that cannot fail is not a verifier.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from regimebench.metrics.evaluators import wilson_interval

# Windows consoles default to cp1252 and will raise UnicodeEncodeError on any non-ASCII
# output. Force UTF-8 so the documented standalone invocation works everywhere.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEX = os.path.join(REPO, "manuscript", "manuscript_array.tex")
OUT = os.path.join(REPO, "experiments", "output")

PCT_TOL = 0.06      # percentages are typeset to 1 dp
DEC_TOL = 6e-4      # decimals are typeset to 3-4 dp
NUM = r"[-+]?\d*\.?\d+"
WORD_NUMBERS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
WORD_NUMBERS.update({"fifteen": 15, "twenty": 20, "thirty": 30, "fifty": 50, "hundred": 100})


# --------------------------------------------------------------------------- helpers

def load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing artifact: experiments/output/{name}")
    return pd.read_csv(path)


def strip_tex(cell: str) -> str:
    """Reduces a LaTeX table cell to its bare content."""
    c = cell.strip()
    c = re.sub(r"\\(textbf|textit|emph|mathbf|hat|times|pm)\b", "", c)
    c = re.sub(r"\\[a-zA-Z]+", " ", c)
    c = c.replace("$", "").replace("{", "").replace("}", "")
    c = c.replace("\\%", "").replace("%", "").replace("~", " ")
    c = c.replace("--", "-").replace("\u2013", "-").replace("\u2014", "-")
    return c.strip()


def cell_numbers(cell: str):
    r"""Every number in a cell, in order.

    A cell can legitimately carry two, e.g. "$\hat k=2$ (86.0\%)" states both the modal
    k and its rate, and both are claims the artifact must support.
    """
    return [float(m) for m in re.findall(NUM, strip_tex(cell))]


def extract_table_rows(tex: str, label: str):
    """Body rows of the tabular inside the table environment carrying \\label{label}."""
    blocks = re.findall(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", tex, re.S)
    block = next((b for b in blocks if f"\\label{{{label}}}" in b), None)
    if block is None:
        raise LookupError(f"no table environment carries \\label{{{label}}}")

    body = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", block, re.S)
    if body is None:
        raise LookupError(f"table {label} has no tabular body")

    # Data rows are exactly what sits between the last \midrule and \bottomrule. Column
    # headers legitimately contain digits ("Null 95th pct.", "$k^*{=}2$") and would
    # otherwise be parsed as data.
    inner = body.group(1)
    if r"\midrule" in inner:
        inner = inner.rsplit(r"\midrule", 1)[1]
    inner = inner.split(r"\bottomrule")[0]

    rows = []
    for raw in inner.split(r"\\"):
        line = re.sub(r"(?m)^\s*%.*$", "", raw)      # drop whole-line comments
        if "&" not in line or r"\multicolumn" in line:
            continue                                  # spanning header rows are not data
        line = re.sub(r"\\(toprule|midrule|bottomrule|cmidrule\(?[lr]*\)?(\{[^}]*\})?|hline)", " ", line)
        cells = [c.strip() for c in line.split("&")]
        if not strip_tex(cells[0]):
            continue                                  # a data row always has a row label
        if not any(re.search(NUM, strip_tex(c)) for c in cells[1:]):
            continue
        rows.append(cells)
    return rows


def norm(text: str) -> str:
    """Collapses a label to comparable form: lowercase alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", strip_tex(text).lower())


def key_of(cells, n_key):
    return norm(" ".join(cells[:n_key]))


# --------------------------------------------------------------------------- table specs
# Each spec rebuilds the expected table from CSV as {row-key: [values...]}.

def expect_factor_grid():
    df = load("pilot_grid_results.csv")
    exp = {}
    for (m, c), sub in df.groupby(["Method", "Criterion"]):
        vals = []
        for k in (2, 3, 4):
            s = sub[sub.True_k == k]
            vals += [s.Exact_Match.mean() * 100, s.Signed_Bias.mean(), s.Estimated_k.mean()]
        exp[f"{m} {c}"] = vals
    return exp, 2, [PCT_TOL, DEC_TOL * 20, DEC_TOL * 20] * 3


def expect_oracle():
    df = load("pilot_grid_results.csv")
    g = df.groupby("Method")[["ARI_at_True_k", "ARI_at_Estimated_k"]].mean()
    return {m: [r.ARI_at_True_k, r.ARI_at_Estimated_k]
            for m, r in g.iterrows()}, 1, [DEC_TOL, DEC_TOL]


def expect_persistence():
    df = load("pilot_grid_results.csv")
    ded = df.drop_duplicates(subset=["Generator", "True_k", "Persistence", "Replication", "Method"])
    exp = {}
    for p, sub in ded.groupby("Persistence"):
        row = [1 / (1 - p)]
        for g in ("ms_garch", "hmm_t"):
            row.append(sub[sub.Generator == g].ARI_at_True_k.mean())
        row.append(sub.ARI_at_True_k.mean())
        exp[f"{p:.2f}"] = row
    return exp, 1, [0.6, DEC_TOL, DEC_TOL, DEC_TOL]


def expect_separation():
    df = load("pilot_grid_results.csv")
    s = df.drop_duplicates(subset=["Generator", "True_k", "Persistence", "Replication"])
    intended = {2: 2.5, 3: 5.0, 4: 10.0}
    label = {"ms_garch": "MS-GARCH", "hmm_t": "HMM Student-t"}
    exp = {}
    for (g, k), sub in s.groupby(["Generator", "True_k"]):
        exp[f"{label[g]} {k}"] = [intended[k]] + [
            sub[sub.Persistence == p].Realized_Vol_Ratio.mean() for p in (0.90, 0.95, 0.99)]
    return exp, 2, [0.01] + [0.02] * 3


def expect_positive_control():
    df = load("positive_control_results.csv")
    label = {
        "sigma_true": r"True $\sigma_t$ (oracle feature)",
        "sigma_log": r"$\log \sigma_t$",
        "roll_sigma_21": r"21-day mean of $\sigma_t$",
        "roll_abs_r_matched": r"$\overline{|r|}$, window matched to dwell time",
        "roll_abs_r_21": r"$\overline{|r|}_{21}$ (standard proxy)",
    }
    g = df.groupby("Feature")[["ARI_KMeans_oracle_k", "ARI_GMM_oracle_k"]].mean()
    byg = df.pivot_table(index="Feature", columns="Generator",
                         values="ARI_KMeans_oracle_k", aggfunc="mean")
    return ({label.get(k, k): [r.ARI_KMeans_oracle_k, r.ARI_GMM_oracle_k,
                               byg.loc[k, "hmm_t"], byg.loc[k, "ms_garch"]]
             for k, r in g.iterrows()}, 1, [DEC_TOL] * 4)


def expect_gap_reference():
    df = load("gap_reference_study.csv")
    exp = {}
    for c, sub in df.groupby("criterion"):
        row = []
        for k in (1, 2, 3, 4):
            s = sub[sub.k_star == k]
            row.append(s.exact_match.mean() * 100 if len(s) else np.nan)
        row.append(sub[sub.k_star > 1].selected_k1.mean() * 100)
        exp[c] = row
    return exp, 1, [PCT_TOL] * 5


def _null_row(sub):
    mode = sub.BIC_k_hat.mode().iloc[0]
    return [float(len(sub)),
            (sub.Silhouette_k_hat == 2).mean() * 100,
            float(mode),
            (sub.BIC_k_hat == mode).mean() * 100,
            (sub.Gap_Statistic_k_hat == 1).mean() * 100]


def expect_null_controls():
    df = load("three_null_controls_results.csv")
    exp = {nm.split(": ", 1)[-1]: _null_row(sub) for nm, sub in df.groupby("Null_Model")}
    # The fourth null lives in its own artifact because it is a different design (30 reps,
    # k in {2..6}); it is reported in the same table, so it is asserted in the same place.
    exp["Stochastic volatility jumps"] = _null_row(load("sv_jump_robustness_results.csv"))
    return exp, 1, [0.5, PCT_TOL, 0.01, PCT_TOL, PCT_TOL]


def expect_criterion_control():
    df = load("criterion_control_results.csv")
    label = {"BIC_Neff": r"BIC ($N_{\text{eff}}$)"}
    exp = {}
    for (m, c), sub in df.groupby(["Method", "Criterion"]):
        vals = []
        for k in (2, 3, 4):
            s = sub[sub.True_k == k]
            vals += [s.Exact_Match.mean() * 100, s.Estimated_k.mean()]
        exp[f"{m} {label.get(c, c)}"] = vals
    return exp, 2, [PCT_TOL, DEC_TOL * 20] * 3


def expect_feature():
    df = load("feature_grid.csv")
    g = df.groupby("representation")[["sil_ari", "bic_ari", "gap_ari", "oracle_k_ari"]].mean()
    # Keys are compared after norm(), which drops LaTeX commands and punctuation, so they
    # are written here as the plain words the manuscript's row labels reduce to.
    keymap = {"4d_vector": "4 D feature vector",
              "moments": "Windowed moments skew kurt",
              "raw_returns": "Raw return windows",
              "rolling_abs_r": "21 day rolling r baseline",
              # strip_tex drops \log, \text and \lambda entirely, so these keys are written
              # as what the typeset labels actually reduce to, not as they read on the page.
              "log_rv_21": "RV 21",
              "har_multiscale": "HAR multi scale RV 1 5 21",
              "ewma_vol": "EWMA volatility 0 94"}
    return {keymap.get(k, k): list(r) for k, r in g.iterrows()}, 1, [DEC_TOL] * 4


def expect_ablation():
    """Per criterion, not pooled: pooling would hide Calinski-Harabasz failing everywhere."""
    df = load("mechanism_ablation_results.csv")
    name = {"---": "none the criterion control", "--S": "smoothing only",
            "-K-": "skew only", "-KS": "skew smoothing",
            "P--": "persistence only", "P-S": "persistence smoothing",
            "PK-": "persistence skew", "PKS": "all three the standard pipeline"}
    exp = {}
    for cond, sub in df.groupby("Condition"):
        ari = sub[(sub.Method == "K-Means") & (sub.Criterion == "Silhouette")].ARI_at_True_k.mean()
        gap = sub[sub.Criterion == "Gap (uniform)"]
        row = [ari] + [sub[sub.Criterion == c].Exact_Match.mean() * 100
                       for c in ("Silhouette", "Davies-Bouldin", "Calinski-Harabasz")]
        row.append((gap.Estimated_k == 1).mean() * 100)
        exp[name.get(cond, cond)] = row
    return exp, 1, [DEC_TOL] + [PCT_TOL] * 4


def expect_multivariate():
    df = load("multivariate_test_results.csv")
    label = {"MS-GARCH k*=3 (4-D Features)": "MS-GARCH k 3",
             "MS-GARCH k*=4 (4-D Features)": "MS-GARCH k 4",
             "Null Single-Regime (k*=1, 4-D Features)": "Single cluster null k 1"}
    exp = {}
    for sc, sub in df.groupby("Scenario"):
        exp[label.get(sc, sc)] = [(sub.Silhouette_k_hat == 2).mean() * 100,
                                  float(sub.BIC_k_hat.mode().iloc[0])]
    return exp, 1, [PCT_TOL, 0.01]


def expect_agreement():
    df = load("real_data_agreement_table.csv")
    strat = [c for c in df.columns if "Strategy" in c][0]
    nber = [c for c in df.columns if "NBER" in c][0]
    bb = [c for c in df.columns if "Bry" in c][0]
    return {str(r[strat]): [r[nber], r[bb]] for _, r in df.iterrows()}, 1, [DEC_TOL, DEC_TOL]


def expect_permutation():
    df = load("permutation_test_results.csv")
    exp = {}
    for _, r in df.iterrows():
        k = f"{r['Partition Strategy']} {r['Reference']}"
        exp[k] = [r["Observed ARI"], r["Null Mean ARI"],
                  r["Null 95th Percentile"], r["Empirical p-value"]]
    return exp, 2, [DEC_TOL, DEC_TOL, DEC_TOL, 0.0011]


TABLE_SPECS = [
    ("tab:factor_grid",   expect_factor_grid),
    ("tab:criterion_control", expect_criterion_control),
    ("tab:oracle",        expect_oracle),
    ("tab:persistence",   expect_persistence),
    ("tab:separation",    expect_separation),
    ("tab:positive",      expect_positive_control),
    ("tab:gap_reference", expect_gap_reference),
    ("tab:null_controls", expect_null_controls),
    ("tab:feature",       expect_feature),
    ("tab:ablation",      expect_ablation),
    ("tab:multivariate",  expect_multivariate),
    ("tab:agreement",     expect_agreement),
    ("tab:permutation",   expect_permutation),
]


# --------------------------------------------------------------------------- prose specs

def prose_specs():
    pilot = load("pilot_grid_results.csv")
    ded = pilot.drop_duplicates(subset=["Generator", "True_k", "Persistence", "Replication", "Method"])
    n_series = pilot.groupby(["Generator", "True_k", "Persistence", "Replication"]).ngroups
    n_pairs = pilot.groupby(["Method", "Criterion"]).ngroups
    series_lvl = pilot.groupby(["Generator", "True_k", "Persistence", "Replication"])[
        ["ARI_at_True_k", "ARI_at_Estimated_k"]].mean()
    from scipy.stats import spearmanr
    rho = spearmanr(series_lvl.ARI_at_True_k, series_lvl.ARI_at_Estimated_k)[0]

    specs = [
        ("headline oracle ARI",
         r"mean (?:Adjusted Rand Index|ARI) only to \$?(" + NUM + r")",
         pilot.ARI_at_True_k.mean(), DEC_TOL),
        ("number of synthetic series",
         r"(\d+) (?:unique )?synthetic series", float(n_series), 0.5),
        ("method-criterion evaluations",
         r"([\d,]+) method\D{0,3}criterion evaluations", float(n_series * n_pairs), 0.5),
        ("series-level Spearman rho",
         r"series-level Spearman.{0,200}?\\rho\s*=\s*(" + NUM + r")", rho, 1e-3),
        ("oracle ARI at P=0.99",
         r"reaches only ARI \$?(" + NUM + r")\$? at \$?P_?\{?ii\}?\s*=\s*0\.99",
         ded[ded.Persistence == 0.99].ARI_at_True_k.mean(), DEC_TOL),
        ("Calinski-Harabasz ceiling rate",
         r"upper candidate boundary[^)]*\)\D{0,20}in (" + NUM + r")\\%",
         (pilot[pilot.Criterion == "Calinski-Harabasz"].Estimated_k == 12).mean() * 100, PCT_TOL),
    ]

    # Wilson score intervals on the n=30 factor-grid cells. The manuscript quotes these
    # to bound the 0.0% cells and to show the headline contrasts do not overlap; each
    # bound below is recomputed from the same CSV the rate itself comes from.
    def _cell(method, criterion, k):
        s = pilot[(pilot.Method == method) & (pilot.Criterion == criterion) & (pilot.True_k == k)]
        return wilson_interval(int(s.Exact_Match.sum()), len(s))

    sil2 = _cell("K-Means", "Silhouette", 2)
    icl2 = _cell("GMM", "ICL", 2)
    zero_hi = _cell("K-Means", "Calinski-Harabasz", 2)["hi"] * 100

    sil4 = _cell("K-Means", "Silhouette", 4)
    icl4 = _cell("GMM", "ICL", 4)

    specs += [
        ("Silhouette k*=2 Wilson lower",
         r"95\\% interval of \$\[(" + NUM + r"),", sil2["lo"] * 100, PCT_TOL),
        ("Silhouette k*=2 Wilson upper",
         r"95\\% interval of \$\[" + NUM + r", (" + NUM + r")\]", sil2["hi"] * 100, PCT_TOL),
        ("Silhouette k*=4 Wilson lower",
         r"against \$\[(" + NUM + r"),", sil4["lo"] * 100, PCT_TOL),
        ("Silhouette k*=4 Wilson upper",
         r"against \$\[" + NUM + r", (" + NUM + r")\]", sil4["hi"] * 100, PCT_TOL),
        ("zero-cell Wilson upper bound",
         r"bounds the true rate at (" + NUM + r")\\%", zero_hi, PCT_TOL),
        ("GMM ICL k*=2 Wilson lower",
         r"GMM ICL falls from \$\[(" + NUM + r"),", icl2["lo"] * 100, PCT_TOL),
        ("GMM ICL k*=2 Wilson upper",
         r"GMM ICL falls from \$\[" + NUM + r", (" + NUM + r")\]", icl2["hi"] * 100, PCT_TOL),
        ("GMM ICL k*=4 Wilson lower",
         r"GMM ICL falls from \$\[[^]]*\]\$ to \$\[(" + NUM + r"),", icl4["lo"] * 100, PCT_TOL),
        ("GMM ICL k*=4 Wilson upper",
         r"GMM ICL falls from \$\[[^]]*\]\$ to \$\[" + NUM + r", (" + NUM + r")\]",
         icl4["hi"] * 100, PCT_TOL),
    ]

    k1 = pilot.dropna(subset=["Selected_k1"])
    icl = k1[(k1.Method == "GMM") & (k1.Criterion == "ICL")]
    if len(icl):
        specs.append(("GMM ICL k=1 rate",
                      r"ICL selects \$\\hat k=1\$ in (" + NUM + r")\\%",
                      icl.Selected_k1.mean() * 100, PCT_TOL))
        # "never select k=1, on any series" is an absolute claim, so it is asserted as one:
        # the count of k=1 selections outside GMM ICL must be exactly zero.
        others = k1[~((k1.Method == "GMM") & (k1.Criterion == "ICL"))]
        specs.append(("criteria other than GMM ICL never select k=1",
                      r"never select \$k=(1)\$, on any series",
                      1.0 if others.Selected_k1.sum() == 0 else -1.0, 0.5))

    # Prose that restates a table value is still a claim the artifact has to support, and
    # it is the claim most likely to go stale: re-running an experiment refreshes the table
    # body through table_bodies.py but leaves every sentence quoting it untouched. Each
    # repeat below is therefore pinned to the same CSV the table is built from.
    def gcell(method, criterion, k, col="Estimated_k"):
        s = pilot[(pilot.Method == method) & (pilot.Criterion == criterion) & (pilot.True_k == k)]
        return s[col].mean()

    def pooled(method, criterion):
        s = pilot[(pilot.Method == method) & (pilot.Criterion == criterion)]
        return s.Exact_Match.mean() * 100

    sil_k2_count = int((pilot[(pilot.Method == "K-Means") & (pilot.Criterion == "Silhouette")]
                        .Estimated_k == 2).sum())
    cell_n = len(pilot[(pilot.Method == "K-Means") & (pilot.Criterion == "Silhouette")
                       & (pilot.True_k == 2)])
    reps = pilot.Replication.nunique()
    hmm = pilot[pilot.Method == "HMM"].groupby(["Criterion", "True_k"]).Estimated_k.mean()
    sep = pilot.drop_duplicates(subset=["Generator", "True_k", "Persistence", "Replication"])

    def sepcell(gen, k, p):
        return sep[(sep.Generator == gen) & (sep.True_k == k)
                   & (sep.Persistence == p)].Realized_Vol_Ratio.mean()

    ded = pilot.drop_duplicates(
        subset=["Generator", "True_k", "Persistence", "Replication", "Method"])

    specs += [
        ("Silhouette k-hat=2 count (abstract)",
         r"selects \$\\hat k=2\$ on (\d+) of \d+ series", float(sil_k2_count), 0.5),
        ("Silhouette k-hat=2 denominator (abstract)",
         r"selects \$\\hat k=2\$ on \d+ of (\d+) series", float(n_series), 0.5),
        ("factor-grid cell size (caption)",
         r"Each cell pools (\d+) series", float(cell_n), 0.5),
        ("factor-grid cell size (uncertainty paragraph)",
         r"Each cell pools \$n=(\d+)\$ series", float(cell_n), 0.5),
        ("one-series rate in the caption",
         r"a rate of (" + NUM + r")\\% is 1/\d+", 100.0 / cell_n, PCT_TOL),
        ("one-series denominator in the caption",
         r"a rate of " + NUM + r"\\% is 1/(\d+)", float(cell_n), 0.5),
        ("replications per cell",
         r"generate (\w+) replications per cell", float(reps), 0.5),
        # This pair is quoted to one decimal, not two, so it carries the 1-dp tolerance.
        ("HMM mean k-hat, lower end",
         r"\\hat k \\approx (" + NUM + r")\$--", hmm.min(), 0.06),
        ("HMM mean k-hat, upper end",
         r"\\hat k \\approx " + NUM + r"\$--\$(" + NUM + r")", hmm.max(), 0.06),
        ("Silhouette signed bias at k*=4",
         r"E\[\\hat k - k\^\*\] = (" + NUM + r")\$ at \$k\^\*=4",
         gcell("K-Means", "Silhouette", 4, "Signed_Bias"), 0.011),
        ("GMM AIC baseline mean k-hat",
         r"GMM AIC begins at \$\\hat k = (" + NUM + r")\$", gcell("GMM", "AIC", 2), 0.011),
        ("Calinski-Harabasz mean k-hat",
         r"Calinski--Harabasz returns (" + NUM + r") at every", gcell("K-Means", "Calinski-Harabasz", 2), 0.011),
        ("Silhouette mean k-hat at k*=3",
         r"Silhouette returns (" + NUM + r") at \$k\^\*=2", gcell("K-Means", "Silhouette", 3), 0.011),
        ("Davies-Bouldin mean k-hat at k*=2",
         r"Davies--Bouldin returns (" + NUM + r") and", gcell("K-Means", "Davies-Bouldin", 2), 0.011),
        ("Davies-Bouldin mean k-hat at k*=4",
         r"Davies--Bouldin returns " + NUM + r" and (" + NUM + r")\.",
         gcell("K-Means", "Davies-Bouldin", 4), 0.011),
        ("GMM BIC pooled recovery",
         r"(" + NUM + r")\\% for BIC,", pooled("GMM", "BIC"), PCT_TOL),
        ("GMM BIC(Neff) pooled recovery",
         r"(" + NUM + r")\\% for BIC\(\$N_\{\\text\{eff\}\}\$\)", pooled("GMM", "BIC_Neff"), PCT_TOL),
        ("GMM AIC pooled recovery",
         r"and (" + NUM + r")\\% for AIC", pooled("GMM", "AIC"), PCT_TOL),
        ("MS-GARCH realized separation at P=0.90",
         r"arrives as \$(" + NUM + r")\\times\$ at \$P_\{ii\}=0\.90", sepcell("ms_garch", 4, 0.90), 0.011),
        ("MS-GARCH two-regime spread at P=0.90",
         r"differ by (" + NUM + r")\\% in realized",
         (sepcell("ms_garch", 2, 0.90) - 1) * 100, 0.6),
        ("MS-GARCH realized separation at P=0.99, k*=4",
         r"realized separation reaches \$(" + NUM + r")\\times\$", sepcell("ms_garch", 4, 0.99), 0.011),
        ("oracle ARI at P=0.90",
         r"recovery is (" + NUM + r") where mean dwell time",
         ded[ded.Persistence == 0.90].ARI_at_True_k.mean(), DEC_TOL),
        ("grand mean oracle ARI restated",
         r"grand mean of (" + NUM + r")", pilot.ARI_at_True_k.mean(), DEC_TOL),
        # Anchored on the Spearman sentence: "$n=30$ series" also occurs earlier, in the
        # uncertainty paragraph, and an unanchored pattern silently matches that one.
        ("Spearman sample size",
         r"series-level Spearman.{0,300}?\$n=(\d+)\$ series", float(n_series), 0.5),
    ]
    # "GMM BIC's apparent gain with k* (10.0, 13.3, 16.7%)" -- three values in one
    # parenthesis, so each needs its own pattern that captures a different position.
    gain = r"does not vary systematically with \$k\^\*\$ at all \("
    for pos, k in enumerate((2, 3, 4)):
        prefix = "".join(NUM + ", " for _ in range(pos))
        specs.append((f"GMM BIC recovery at k*={k} (uncertainty paragraph)",
                      gain + prefix + r"(" + NUM + r")",
                      pilot[(pilot.Method == "GMM") & (pilot.Criterion == "BIC")
                            & (pilot.True_k == k)].Exact_Match.mean() * 100, PCT_TOL))

    # Boundary pinning rests on an invariance claim, so the invariance is asserted rather
    # than eyeballed: the largest movement in mean k-hat, across the six criteria the paper
    # calls unresponsive, when the truth moves from k*=2 to k*=4.
    mean_k = pilot.pivot_table(index=["Method", "Criterion"], columns="True_k",
                               values="Estimated_k", aggfunc="mean")
    invariant = [i for i in mean_k.index if i[0] in ("K-Means", "HMM")]
    responsive = [i for i in mean_k.index if i[0] == "GMM"]
    drift = (mean_k[4] - mean_k[2]).abs()
    icl_pooled = pilot[(pilot.Method == "GMM") & (pilot.Criterion == "ICL")].Exact_Match.mean() * 100
    specs += [
        ("count of unresponsive criteria",
         r"(\w+) of the ten (?:show no response|are wholly unresponsive|do not respond)",
         float(len(invariant)), 0.5),
        ("largest drift among the unresponsive criteria",
         r"moves? their mean \$?\\hat k\$? by at most (" + NUM + r")",
         drift[invariant].max(), 5e-3),
        ("smallest drift among the responsive criteria",
         r"\(ICL\) to (" + NUM + r") \(BIC with", drift[responsive].max(), 5e-3),
        ("largest drift among the responsive criteria",
         r"by (" + NUM + r") \(ICL\) to", drift[responsive].min(), 5e-3),
        ("GMM ICL pooled recovery",
         r"exact recovery is (" + NUM + r")\\% for ICL", icl_pooled, PCT_TOL),
    ]

    # The separation sweep carries the paper's sharpest claim -- Silhouette's accuracy
    # running backwards against signal strength -- and it was previously stated in prose
    # with no table behind it. Panel (c) of the figure is MS-GARCH at P_ii=0.99, so the
    # endpoints are asserted against exactly that slice.
    try:
        sweep = load("separation_sweep.csv")
    except FileNotFoundError:
        sweep = None
    if sweep is not None:
        cell = sweep[(sweep.Generator == "ms_garch") & (sweep.Persistence == 0.99)]
        lo = cell[cell.Intended_Vol_Ratio == cell.Intended_Vol_Ratio.min()]
        hi = cell[cell.Intended_Vol_Ratio == cell.Intended_Vol_Ratio.max()]
        specs += [
            ("sweep oracle ARI, weakest separation",
             r"oracle ARI rises from (" + NUM + r") to", lo.Oracle_k_ARI.mean(), DEC_TOL),
            ("sweep oracle ARI, strongest separation",
             r"oracle ARI rises from " + NUM + r" to (" + NUM + r")",
             hi.Oracle_k_ARI.mean(), DEC_TOL),
            ("sweep Silhouette recovery, weakest separation",
             r"\\emph\{falls\}, from (" + NUM + r")\\%",
             lo.Silhouette_Exact.mean() * 100, PCT_TOL),
            ("sweep Silhouette recovery, strongest separation",
             r"\\emph\{falls\}, from " + NUM + r"\\% to (" + NUM + r")\\%",
             hi.Silhouette_Exact.mean() * 100, PCT_TOL),
        ]

    # The criterion-level control. Every claim the control licenses is asserted here,
    # because it is the control that licenses every negative result in the paper.
    try:
        cc = load("criterion_control_results.csv")
    except FileNotFoundError:
        cc = None
    if cc is not None:
        gap = cc[cc.Criterion == "Gap (uniform)"]
        non_hmm = cc[cc.Method != "HMM"]
        specs += [
            ("criterion control series",
             r"(\d+) series and 1\{?,?\}?080 evaluations",
             float(cc.groupby(["True_k", "Replication"]).ngroups), 0.5),
            ("criterion control evaluations",
             r"series and (1\{?,?\}?\d\d\d) evaluations", float(len(cc)), 0.5),
            ("Gap k=1 rate on Gaussian clusters",
             r"Gaussian series---(" + NUM + r")\\%", (gap.Estimated_k == 1).mean() * 100, PCT_TOL),
            ("criterion control non-HMM cells",
             r"The (\d+) method--criterion--\$k\^\*\$ cells outside",
             float(non_hmm.groupby(["Method", "Criterion", "True_k"]).ngroups), 0.5),
            ("criterion control non-HMM recovery",
             r"outside the HMM block average (" + NUM + r")\\%",
             non_hmm.Exact_Match.mean() * 100, PCT_TOL),
            ("criterion control non-HMM ARI",
             r"outside the HMM block average.{0,40}?mean ARI (" + NUM + r")",
             non_hmm.ARI_at_Estimated_k.mean(), DEC_TOL),
        ]

    # The feature-representation section leans on a contrast between how often Silhouette
    # answers 2 on the skewed baseline and on its de-skewed log transform, so both rates
    # and the recovery that replaces the pinning are asserted.
    try:
        fg = load("feature_grid.csv")
    except FileNotFoundError:
        fg = None
    if fg is not None:
        def frate(rep, k):
            s = fg[fg.representation == rep]
            return (s.sil_k == k).mean() * 100
        specs += [
            ("Silhouette k-hat=2 rate on the baseline feature",
             r"and in (" + NUM + r")\\% on the baseline", frate("rolling_abs_r", 2), PCT_TOL),
            ("Silhouette k-hat=2 rate on log RV",
             r"in only (" + NUM + r")\\% on \$\\log \\text\{RV\}_\{21\}\$", frate("log_rv_21", 2), PCT_TOL),
            ("Silhouette exact recovery on log RV",
             r"exact recovery on \$\\log \\text\{RV\}_\{21\}\$ is (" + NUM + r")\\%",
             frate("log_rv_21", 3), PCT_TOL),
            ("oracle ARI on log RV",
             r"oracle ARI is unmoved at (" + NUM + r")",
             fg[fg.representation == "log_rv_21"].oracle_k_ari.mean(), DEC_TOL),
            ("baseline oracle ARI in the feature grid",
             r"baseline's (" + NUM + r") by any margin",
             fg[fg.representation == "rolling_abs_r"].oracle_k_ari.mean(), DEC_TOL),
            ("feature-grid oracle ARI range, lower",
             r"stays within \$\[(" + NUM + r"), ", fg.groupby("representation").oracle_k_ari.mean().min(), 5e-4),
            ("feature-grid oracle ARI range, upper",
             r"stays within \$\[" + NUM + r", (" + NUM + r")\]",
             fg.groupby("representation").oracle_k_ari.mean().max(), 5e-4),
        ]

    # The ablation section states three main effects and one interaction. All four are the
    # point of the experiment, so all four are asserted rather than read off the table.
    try:
        ab = load("mechanism_ablation_results.csv")
    except FileNotFoundError:
        ab = None
    if ab is not None:
        geo = ab[ab.Criterion.isin(["Silhouette", "Davies-Bouldin"])]
        km = ab[(ab.Method == "K-Means") & (ab.Criterion == "Silhouette")]

        def eff(col):
            return (geo[geo[col] == 1].Exact_Match.mean()
                    - geo[geo[col] == 0].Exact_Match.mean()) * 100

        def ari(persist, skew, smooth):
            return km[(km.Persistent == persist) & (km.Skewed == skew)
                      & (km.Smoothed == smooth)].ARI_at_True_k.mean()

        specs += [
            ("ablation persistence main effect",
             r"main effect on geometric recovery is \$\+(" + NUM + r")\$ percentage points",
             eff("Persistent"), 0.06),
            ("ablation skew main effect",
             r"skew costs (" + NUM + r") percentage points", -eff("Skewed"), 0.06),
            ("ablation smoothing main effect",
             r"and smoothing (" + NUM + r")\.", -eff("Smoothed"), 0.06),
            ("ablation interaction, symmetric, iid",
             r"oracle ARI is (" + NUM + r") under i\.i\.d\.\\? states", ari(0, 0, 1), DEC_TOL),
            ("ablation interaction, symmetric, persistent",
             r"and (" + NUM + r") under persistent ones", ari(1, 0, 1), DEC_TOL),
            ("ablation interaction, skewed, iid",
             r"the pattern repeats, (" + NUM + r") against", ari(0, 1, 1), DEC_TOL),
            ("ablation interaction, skewed, persistent",
             r"the pattern repeats, " + NUM + r" against (" + NUM + r")", ari(1, 1, 1), DEC_TOL),
            ("ablation oracle ARI under skew alone",
             r"falls from 1\.0000 to (" + NUM + r") under skew alone", ari(0, 1, 0), DEC_TOL),
            ("ablation oracle ARI under smoothing alone",
             r"and to (" + NUM + r") under smoothing alone", ari(0, 0, 1), DEC_TOL),
            ("Calinski-Harabasz ceiling in every ablation condition",
             r"selects \$\\hat k=(" + NUM + r")\$, the ceiling, in every one",
             ab[ab.Criterion == "Calinski-Harabasz"].Estimated_k.mean(), 0.011),
        ]

    # The Gap statistic's k=1 rate on regime-bearing series is quoted in the abstract, in
    # the contributions, in Section 5.8 and in the conclusion. One spec covers the claim;
    # a divergence between the four statements would surface as a table mismatch.
    try:
        gref = load("gap_reference_study.csv")
    except FileNotFoundError:
        gref = None
    if gref is not None:
        uni = gref[gref.criterion == "Gap (uniform)"]
        specs.append(("Gap uniform k=1 rate on regime-bearing series",
                      r"\$\\hat k=1\$ on (" + NUM + r")\\% of",
                      uni[uni.k_star > 1].selected_k1.mean() * 100, PCT_TOL))

    # The high-frequency check reports both the raw kline count and the thinned count the
    # selection actually ran on; the second was previously undisclosed.
    try:
        bnb = load("binance_benchmark_results.csv")
    except FileNotFoundError:
        bnb = None
    if bnb is not None and len(bnb):
        specs += [
            ("Binance raw klines",
             r"([\d,]+) one-minute BTCUSDT klines",
             float(bnb.N_Raw_Klines.iloc[0]), 0.5),
            ("Binance thinned points",
             r"every tenth minute for the selection step, ([\d,]+) points",
             float(bnb.N_Subsampled.iloc[0]), 0.5),
        ]

    return specs


# --------------------------------------------------------------------------- checks

def check_tables(tex, errors, notes, perturb=None):
    for label, builder in TABLE_SPECS:
        try:
            expected, n_key, tols = builder()
        except FileNotFoundError as e:
            notes.append(f"SKIP {label}: {e}")
            continue

        if perturb == label:
            first = next(iter(expected))
            expected[first] = [(v + 1.0) if v is not None else v for v in expected[first]]

        try:
            rows = extract_table_rows(tex, label)
        except LookupError as e:
            errors.append(f"{label}: {e}")
            continue

        # Rows are matched by their normalized label, so table order is free but every
        # artifact row must appear exactly once and no invented row may appear.
        expected = {norm(k): v for k, v in expected.items()}
        typeset = {key_of(cells, n_key): cells for cells in rows}

        if len(typeset) != len(rows):
            errors.append(f"{label}: duplicate row labels in the typeset table")
        for missing in sorted(set(expected) - set(typeset)):
            errors.append(f"{label}: artifact row '{missing}' is missing from the table")
        for extra in sorted(set(typeset) - set(expected)):
            errors.append(f"{label}: table row '{extra}' has no counterpart in the artifact")

        for exp_key, want_vals in expected.items():
            cells = typeset.get(exp_key)
            if cells is None:
                continue
            got = [v for c in cells[n_key:] for v in cell_numbers(c)]
            for i, want in enumerate(want_vals):
                if want is None or (isinstance(want, float) and np.isnan(want)):
                    continue
                if i >= len(got):
                    errors.append(f"{label} [{exp_key}]: column {i+1} missing from the typeset row")
                    break
                tol = tols[i] if i < len(tols) else DEC_TOL
                if abs(got[i] - want) > tol:
                    errors.append(f"{label} [{exp_key}] col {i+1}: manuscript {got[i]:.4f} "
                                  f"vs artifact {want:.4f} (tol {tol})")

        notes.append(f"  {label}: {len(expected)} rows asserted")


def check_prose(tex, errors, notes):
    try:
        specs = prose_specs()
    except FileNotFoundError as e:
        notes.append(f"SKIP prose checks: {e}")
        return
    for name, pattern, want, tol in specs:
        m = re.search(pattern, tex, re.I)
        if not m:
            errors.append(f"prose [{name}]: no sentence in the manuscript states this value")
            continue
        # Thousands in the manuscript are typeset either as "44,638" or as LaTeX's
        # "1{,}080"; both name the same integer. Small counts are spelled out in prose,
        # as house style requires, and still have to be asserted.
        raw = re.sub(r"[,{}]", "", m.group(1)).strip().lower()
        got = float(WORD_NUMBERS[raw]) if raw in WORD_NUMBERS else float(raw)
        if abs(got - want) > tol:
            errors.append(f"prose [{name}]: manuscript {got} vs artifact {want:.4f} (tol {tol})")
        else:
            notes.append(f"  prose [{name}]: {got} matches {want:.4f}")


def check_structure(tex, errors, notes):
    for i, line in enumerate(tex.split("\n"), 1):
        if line.lstrip().startswith("%"):
            continue                      # whole-line comment
        m = re.search(r"(?<!\\)%", line)   # only the first one matters: it comments the rest
        if m is None or line[m.start():].strip() == "%":
            continue                      # no bare %, or an intentional line-join
        errors.append(f"line {i}: unescaped % truncates the sentence in the PDF: "
                      f"...{line[max(0, m.start()-40):m.start()+10]}")

    if "FIXME" in tex or "REPLACE-ME" in tex:
        errors.append("placeholder DOI/URL still present (search FIXME / REPLACE-ME)")

    # A zeroed Zenodo suffix is a placeholder that reads as a real DOI, so neither the
    # FIXME check above nor a human proofread reliably catches it.
    if re.search(r"zenodo\.0+\b", tex):
        errors.append("Zenodo DOI is still the placeholder 10.5281/zenodo.0000000 -- "
                      "mint the archive and substitute the real suffix before submitting")

    if "\\linenumbers" not in tex:
        errors.append("\\linenumbers is never called, so the review PDF carries no line numbers")

    for path in set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", tex)):
        resolved = os.path.normpath(os.path.join(REPO, "manuscript", path))
        candidates = [resolved, resolved + ".pdf", resolved + ".png"]
        if not any(os.path.exists(c) for c in candidates):
            errors.append(f"figure referenced but not on disk: {path}")

    cited = set()
    for c in re.findall(r"\\cite\{([^}]*)\}", tex):
        cited.update(x.strip() for x in c.split(","))
    defined = set(re.findall(r"\\bibitem\{([^}]*)\}", tex))
    for k in sorted(cited - defined):
        errors.append(f"citation \\cite{{{k}}} has no \\bibitem")
    for k in sorted(defined - cited):
        errors.append(f"\\bibitem{{{k}}} is never cited")

    if "\\begin{highlights}" in tex:
        errors.append("highlights block is inside the .tex; Elsevier wants the separate "
                      "highlights.txt only (duplicate sets drift apart)")

    notes.append(f"  structure: {len(cited)} citations, all resolved" if not (cited - defined)
                 else "  structure: citation problems above")


def run(perturb=None):
    if not os.path.exists(TEX):
        # The article source is not distributed with this repository -- the repository is
        # the benchmark, not the paper. Absence is therefore the normal state for anyone
        # but the authors, and it is a skip rather than a failure: a reader's pipeline
        # must not go red over a file it was never given. Drift detection still exits
        # non-zero whenever the source *is* present.
        print("=" * 90)
        print("MANUSCRIPT INTEGRITY VERIFIER -- SKIPPED")
        print("=" * 90)
        print(f"No manuscript source at {TEX}.")
        print("The article source is not distributed with this repository, so there is")
        print("nothing to assert the artifacts against. Every number the article reports")
        print("is in experiments/output/; this step re-checks the article against them.")
        print("=" * 90)
        return 0
    tex = open(TEX, encoding="utf-8").read()

    errors, notes = [], []
    print("=" * 90)
    print("MANUSCRIPT INTEGRITY VERIFIER")
    print("=" * 90)
    print(f"source: manuscript/manuscript_array.tex ({len(tex.splitlines())} lines)\n")

    print("--- tables ---")
    check_tables(tex, errors, notes, perturb=perturb)
    print("--- prose ---")
    check_prose(tex, errors, notes)
    print("--- structure ---")
    check_structure(tex, errors, notes)

    for n in notes:
        print(n)

    print("\n" + "=" * 90)
    if errors:
        print(f"VERIFICATION FAILED - {len(errors)} issue(s):\n")
        for i, e in enumerate(errors, 1):
            print(f"  {i:2d}. {e}")
        print("=" * 90)
        return 1
    print("ALL CHECKS PASSED")
    print("=" * 90)
    return 0


def self_test():
    """A verifier that cannot fail is not a verifier. Confirm it catches injected drift."""
    if not os.path.exists(TEX):
        print("SELF-TEST SKIPPED: no manuscript source in this repository, so there is")
        print("no assertion for the injected drift to break. See run() for why.")
        return 0
    print("SELF-TEST: injecting a +1.0 perturbation into tab:factor_grid's first row\n")
    code = run(perturb="tab:factor_grid")
    print()
    if code != 0:
        print("SELF-TEST PASSED: the verifier detected the injected drift.")
        return 0
    print("SELF-TEST FAILED: the verifier did not detect injected drift.")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="confirm the verifier fails when an artifact is perturbed")
    args = ap.parse_args()
    sys.exit(self_test() if args.self_test else run())
