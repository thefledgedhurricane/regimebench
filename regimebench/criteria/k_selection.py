"""Cluster-Validity Indices & Information Criteria Module."""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
def compute_internal_indices(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Computes geometric internal validity indices for a partition (X, labels).
    
    Returns:
        dict with keys: 'silhouette', 'davies_bouldin', 'calinski_harabasz'
    """
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(X):
        return {
            'silhouette': -1.0,
            'davies_bouldin': 999.0,
            'calinski_harabasz': 0.0
        }

    try:
        sil = float(silhouette_score(X, labels))
    except Exception:
        sil = -1.0

    try:
        db = float(davies_bouldin_score(X, labels))
    except Exception:
        db = 999.0

    try:
        ch = float(calinski_harabasz_score(X, labels))
    except Exception:
        ch = 0.0

    return {
        'silhouette': sil,
        'davies_bouldin': db,
        'calinski_harabasz': ch
    }

def compute_information_criteria(
    log_likelihood: float, 
    num_params: int, 
    n_samples: int, 
    posterior_probs: Optional[np.ndarray] = None,
    effective_window: int = 21
) -> Dict[str, float]:
    """Computes AIC, BIC(N), BIC(N_eff), and ICL for model-based clustering (GMM, HMM).
    
    Formulae:
        AIC = -2 * logL + 2 * p
        BIC(N) = -2 * logL + p * log(N)
        BIC(N_eff) = -2 * logL + p * log(N / effective_window)
        ICL = BIC - 2 * sum_{i, k} z_{ik} * log(z_{ik})
    """
    aic = -2.0 * log_likelihood + 2.0 * num_params
    bic = -2.0 * log_likelihood + num_params * np.log(n_samples)

    n_eff = max(float(n_samples) / float(effective_window), 2.0)
    bic_neff = -2.0 * log_likelihood + num_params * np.log(n_eff)

    icl = bic
    if posterior_probs is not None and posterior_probs.shape[0] == n_samples:
        eps = 1e-12
        probs_clipped = np.clip(posterior_probs, eps, 1.0)
        entropy = -np.sum(probs_clipped * np.log(probs_clipped))
        icl = bic + 2.0 * entropy

    return {
        'aic': float(aic),
        'bic': float(bic),
        'bic_neff': float(bic_neff),
        'icl': float(icl)
    }

def _draw_reference(
    X: np.ndarray,
    rng: np.random.RandomState,
    reference: str
) -> np.ndarray:
    """Draws one reference sample under the chosen null model.

    'uniform'  -- Tibshirani et al. (2001) method (a): uniform over the bounding box
                  of X. Its null hypothesis is *uniformly distributed data*.
    'pca'      -- Tibshirani et al. (2001) method (b): uniform over the bounding box of
                  the principal-component rotation of X, mapped back. Identical to
                  'uniform' for univariate X.
    'gaussian' -- single-component parametric null: N(mean(X), cov(X)). Its null
                  hypothesis is *one regime*, which is the hypothesis a regime-count
                  criterion actually needs to test.
    'lognormal'-- single-component parametric null matched to a volatility proxy:
                  Gaussian fitted in log space, exponentiated.

    The choice matters decisively for skewed features. A rolling mean of |r| is
    strongly right-skewed and therefore very far from uniform, so under 'uniform' the
    k=1 gap is inflated by the marginal shape alone and the one-standard-error rule
    selects k=1 regardless of how much regime structure is present. See
    ``compute_gap_curve`` and ``experiments/run_gap_diagnostic.py``.
    """
    n_samples, n_features = X.shape

    if reference == "uniform":
        return rng.uniform(np.min(X, axis=0), np.max(X, axis=0), size=(n_samples, n_features))

    if reference == "pca":
        centre = np.mean(X, axis=0)
        Xc = X - centre
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = Xc @ Vt.T
        Z_ref = rng.uniform(np.min(Z, axis=0), np.max(Z, axis=0), size=(n_samples, n_features))
        return Z_ref @ Vt + centre

    if reference == "gaussian":
        centre = np.mean(X, axis=0)
        if n_features == 1:
            scale = np.std(X, axis=0, ddof=1)
            return centre + rng.standard_normal((n_samples, 1)) * scale
        cov = np.cov(X, rowvar=False)
        cov = np.atleast_2d(cov) + 1e-12 * np.eye(n_features)
        return rng.multivariate_normal(centre, cov, size=n_samples)

    if reference == "lognormal":
        # Single-regime null matched to a volatility proxy, which is approximately
        # lognormal: fit a Gaussian in log space and exponentiate. Falls back to the
        # Gaussian null for features that are not strictly positive.
        if np.min(X) <= 0:
            return _draw_reference(X, rng, "gaussian")
        L = np.log(X)
        centre = np.mean(L, axis=0)
        if n_features == 1:
            scale = np.std(L, axis=0, ddof=1)
            return np.exp(centre + rng.standard_normal((n_samples, 1)) * scale)
        cov = np.atleast_2d(np.cov(L, rowvar=False)) + 1e-12 * np.eye(n_features)
        return np.exp(rng.multivariate_normal(centre, cov, size=n_samples))

    raise ValueError(f"Unknown reference distribution: {reference!r}")


def compute_gap_curve(
    X: np.ndarray,
    clustering_func,
    k_range: range = range(1, 7),
    n_bootstraps: int = 10,
    random_state: Optional[int] = 42,
    reference: str = "uniform"
) -> Dict[str, Dict[int, float]]:
    """Returns the full Gap curve and its standard errors without applying any rule.

    Diagnostic entry point: exposes whether the curve is monotone decreasing (in which
    case the one-standard-error rule can only ever return the smallest candidate).
    """
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    k_values = list(k_range)
    if not k_values:
        raise ValueError("k_range must contain at least one candidate.")
    if n_bootstraps < 2:
        raise ValueError("n_bootstraps must be at least 2 for the Gap statistic.")

    rng = np.random.RandomState(random_state)
    gaps, standard_errors, log_w = {}, {}, {}

    def log_within_cluster_dispersion(data: np.ndarray, labels: np.ndarray, k: int) -> float:
        dispersion = 0.0
        for cluster in range(k):
            cluster_points = data[labels == cluster]
            if len(cluster_points) > 1:
                center = np.mean(cluster_points, axis=0)
                dispersion += np.sum((cluster_points - center) ** 2)
        return float(np.log(max(dispersion, 1e-12)))

    for k in k_values:
        labels = clustering_func(X, k)
        log_W_k = log_within_cluster_dispersion(X, labels, k)

        log_W_k_ref = []
        for _ in range(n_bootstraps):
            X_ref = _draw_reference(X, rng, reference)
            labels_ref = clustering_func(X_ref, k)
            log_W_k_ref.append(log_within_cluster_dispersion(X_ref, labels_ref, k))

        log_w[k] = log_W_k
        gaps[k] = float(np.mean(log_W_k_ref) - log_W_k)
        standard_errors[k] = float(
            np.sqrt(1.0 + 1.0 / n_bootstraps) * np.std(log_W_k_ref, ddof=1)
        )

    return {"gap": gaps, "se": standard_errors, "log_w": log_w}


def select_k_one_se(
    gaps: Dict[int, float],
    standard_errors: Dict[int, float],
    k_values: Optional[list] = None
) -> int:
    """Tibshirani's one-standard-error rule: smallest k with gap(k) >= gap(k+1) - s(k+1).

    Kept separate from the estimation so the rule can be tested against a hand-built
    curve rather than against a noisy simulation.
    """
    if k_values is None:
        k_values = sorted(gaps)
    if not k_values:
        raise ValueError("k_values must contain at least one candidate.")

    for current_k, next_k in zip(k_values, k_values[1:]):
        if gaps[current_k] >= gaps[next_k] - standard_errors[next_k]:
            return current_k
    return k_values[-1]


def compute_gap_statistic(
    X: np.ndarray,
    clustering_func,
    k_range: range = range(2, 7),
    n_bootstraps: int = 10,
    random_state: Optional[int] = 42,
    reference: str = "uniform"
) -> Tuple[int, Dict[int, float]]:
    """Computes the Gap statistic and applies Tibshirani's one-standard-error rule.

    ``reference`` selects the null model the gap is measured against; see
    ``_draw_reference``. The default is ``'uniform'``, faithful to Tibshirani et al.
    (2001) method (a) as cited in the manuscript. That default is *degenerate* on a
    right-skewed feature -- it returns the smallest admissible k unconditionally -- so
    the reference is swept as an explicit experimental factor rather than fixed; see
    ``experiments/run_gap_reference_study.py``.
    """
    curve = compute_gap_curve(
        X, clustering_func, k_range=k_range, n_bootstraps=n_bootstraps,
        random_state=random_state, reference=reference
    )
    gaps, standard_errors = curve["gap"], curve["se"]
    return select_k_one_se(gaps, standard_errors, list(k_range)), gaps


def compute_prediction_strength(
    X: np.ndarray,
    k_range: range = range(2, 7),
    threshold: float = 0.8,
    random_state: Optional[int] = 42
) -> Tuple[int, Dict[int, float]]:
    """Computes prediction strength of clustering (Tibshirani & Walther, 2005)."""
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    n_samples = X.shape[0]
    n_train = n_samples // 2
    X_train = X[:n_train]
    X_test = X[n_train:]
    
    ps_dict = {}
    
    for k in k_range:
        kmeans_train = KMeans(n_clusters=k, random_state=random_state, n_init=3)
        kmeans_train.fit(X_train)
        
        kmeans_test = KMeans(n_clusters=k, random_state=random_state, n_init=3)
        labels_test = kmeans_test.fit_predict(X_test)
        
        test_to_train_labels = kmeans_train.predict(X_test)
        
        cluster_proportions = []
        for j in range(k):
            indices = np.where(labels_test == j)[0]
            n_j = len(indices)
            
            if n_j <= 1:
                if n_j == 1:
                    cluster_proportions.append(1.0)
                continue
                
            unique_train_labels, counts = np.unique(test_to_train_labels[indices], return_counts=True)
            co_clustered_pairs = np.sum(counts * (counts - 1))
            total_pairs = n_j * (n_j - 1)
            
            cluster_proportions.append(co_clustered_pairs / total_pairs)
            
        if not cluster_proportions:
            ps_dict[k] = 0.0
        else:
            ps_dict[k] = float(np.min(cluster_proportions))
            
    best_k = 1
    for k in sorted(list(k_range), reverse=True):
        if ps_dict[k] > threshold:
            best_k = k
            break
            
    return best_k, ps_dict
