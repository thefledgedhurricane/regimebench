import numpy as np
import pandas as pd
import pytest

from regimebench.criteria.k_selection import (
    compute_gap_curve,
    compute_gap_statistic,
    compute_information_criteria,
    compute_prediction_strength,
    select_k_one_se,
)
from regimebench.methods.clustering_wrappers import fit_hmm, fit_kmeans
from regimebench.metrics.evaluators import mean_regime_duration, wilson_interval


def _km(seed=7):
    return lambda X, k: fit_kmeans(X, k, random_state=seed)[0]


def _skewed_volatility_feature(seed=0, T=1500):
    """A right-skewed rolling volatility proxy, i.e. the feature the benchmark uses."""
    rng = np.random.RandomState(seed)
    y = rng.standard_t(5, size=T) / np.sqrt(5 / 3)
    return pd.Series(np.abs(y)).rolling(21, min_periods=1).mean().values.reshape(-1, 1)


def test_one_standard_error_rule_picks_the_first_k_that_clears_the_next():
    """The rule itself, on hand-built curves, with no bootstrap noise in the way."""
    ses = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1}

    # Monotone decreasing: the first comparison already succeeds.
    assert select_k_one_se({1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4}, ses) == 1

    # A clear peak at k=3: k=1 and k=2 are each beaten by more than one SE.
    assert select_k_one_se({1: 0.2, 2: 0.5, 3: 0.9, 4: 0.85}, ses) == 3

    # Within one SE counts as clearing it: 0.5 >= 0.55 - 0.1.
    assert select_k_one_se({1: 0.5, 2: 0.55, 3: 0.9, 4: 1.2}, ses) == 1

    # Strictly increasing beyond every SE: fall through to the last candidate.
    assert select_k_one_se({1: 0.1, 2: 0.5, 3: 0.9, 4: 1.3}, ses) == 4


def test_gap_statistic_reports_the_whole_candidate_curve():
    def one_cluster_labels(X, k):
        return np.zeros(len(X), dtype=int)

    X = np.arange(20, dtype=float).reshape(-1, 1)
    selected_k, gaps = compute_gap_statistic(
        X, one_cluster_labels, k_range=range(1, 4), n_bootstraps=5, random_state=7)

    assert set(gaps) == {1, 2, 3}
    assert selected_k in gaps


def test_gaussian_hmm_parameter_count_includes_initial_probabilities():
    X = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]])
    _, info = fit_hmm(X, k=2, random_state=7)

    # start probabilities (1) + transitions (2) + means (2) + variances (2)
    assert info["num_params"] == 7


def test_uniform_gap_reference_is_degenerate_on_a_skewed_feature():
    """Regression guard for the paper's central methodological finding.

    Under the classical uniform bounding-box reference the gap curve on a right-skewed
    volatility proxy is monotone decreasing, so the one-standard-error rule can only
    return the smallest admissible candidate. If this ever stops holding, the manuscript's
    Section 4.8 needs rewriting -- so the test asserts it rather than assuming it.
    """
    X = _skewed_volatility_feature()
    curve = compute_gap_curve(X, _km(), k_range=range(1, 6), n_bootstraps=10,
                              random_state=7, reference="uniform")
    gaps = curve["gap"]

    assert max(gaps, key=gaps.get) == 1
    assert all(gaps[k] >= gaps[k + 1] for k in range(1, 5))
    assert compute_gap_statistic(X, _km(), k_range=range(1, 6), n_bootstraps=10,
                                 random_state=7, reference="uniform")[0] == 1


def test_single_regime_references_are_not_degenerate():
    """The Gaussian and lognormal nulls remove the pinning the uniform null creates."""
    X = _skewed_volatility_feature()
    for reference in ("gaussian", "lognormal"):
        gaps = compute_gap_curve(X, _km(), k_range=range(1, 6), n_bootstraps=10,
                                 random_state=7, reference=reference)["gap"]
        assert not all(gaps[k] >= gaps[k + 1] for k in range(1, 5)), (
            f"{reference} reference produced a monotone-decreasing curve")


def test_pca_reference_matches_uniform_in_one_dimension():
    """A rotation is a no-op on univariate data; the PCA variant must not silently differ."""
    X = _skewed_volatility_feature(seed=3)
    a = compute_gap_curve(X, _km(), k_range=range(1, 4), n_bootstraps=6,
                          random_state=11, reference="uniform")["gap"]
    b = compute_gap_curve(X, _km(), k_range=range(1, 4), n_bootstraps=6,
                          random_state=11, reference="pca")["gap"]
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=0.15)


def test_unknown_reference_is_rejected():
    X = _skewed_volatility_feature(seed=1, T=200)
    with pytest.raises(ValueError, match="Unknown reference"):
        compute_gap_statistic(X, _km(), k_range=range(1, 3), n_bootstraps=3,
                              reference="not-a-null")


def test_information_criteria_penalise_effective_sample_size():
    ic = compute_information_criteria(log_likelihood=-100.0, num_params=5,
                                      n_samples=2100, effective_window=21)
    assert ic["aic"] == pytest.approx(210.0)
    assert ic["bic"] > ic["bic_neff"]          # N_eff shrinks the penalty
    assert ic["icl"] == pytest.approx(ic["bic"])  # no posteriors -> ICL falls back to BIC


def test_wilson_interval_brackets_the_rate_and_stays_in_unit_range():
    mid = wilson_interval(26, 30)
    assert mid["lo"] < mid["rate"] < mid["hi"]

    # The Wald interval leaves [0,1] at the boundary cells this grid produces; Wilson must not.
    for successes in (0, 30):
        w = wilson_interval(successes, 30)
        assert 0.0 <= w["lo"] <= w["hi"] <= 1.0

    assert np.isnan(wilson_interval(0, 0)["rate"])


def test_mean_regime_duration_matches_geometric_dwell_time():
    assert mean_regime_duration(0.90) == pytest.approx(10.0)
    assert mean_regime_duration(0.95) == pytest.approx(20.0)
    assert mean_regime_duration(0.99) == pytest.approx(100.0)


def test_prediction_strength_can_return_one():
    X = _skewed_volatility_feature(seed=5, T=800)
    k_hat, scores = compute_prediction_strength(X, k_range=range(1, 5), random_state=7)
    assert k_hat >= 1
    assert set(scores) <= {1, 2, 3, 4}


def test_factor_grid_config_keys_are_all_consumed():
    """Every key the config declares must be read; a silently ignored key is a trap."""
    import os
    import yaml

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "experiments",
                            "config", "factor_grid_pilot.yaml")
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    source = open(os.path.join(os.path.dirname(__file__), "..", "experiments",
                               "run_pilot_grid.py"), encoding="utf-8").read()

    for key in ("random_seed", "sample_size_T", "replications_per_cell", "k_candidates"):
        assert f"cfg['{key}']" in source, f"{key} declared in the config but never read"
    for key in cfg["factor_grid"]:
        assert f"'{key}'" in source, f"factor_grid.{key} declared in the config but never read"
