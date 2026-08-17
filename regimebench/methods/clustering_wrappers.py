"""Unified fit/predict wrappers for regime clustering models (K-Means, GMM, HMM)."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm

def fit_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fits K-Means clustering and returns (labels, model_info)."""
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return labels, {'inertia': float(model.inertia_), 'centers': model.cluster_centers_}

def fit_gmm(X: np.ndarray, k: int, random_state: int = 42) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fits Gaussian Mixture Model and returns (labels, model_info)."""
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    model = GaussianMixture(n_components=k, random_state=random_state, n_init=5)
    labels = model.fit_predict(X)
    posteriors = model.predict_proba(X)
    log_l = model.score(X) * len(X)
    n_params = k * X.shape[1] + k * X.shape[1] * (X.shape[1] + 1) / 2 + (k - 1)

    return labels, {
        'log_likelihood': float(log_l),
        'num_params': int(n_params),
        'posteriors': posteriors
    }

def fit_hmm(X: np.ndarray, k: int, random_state: int = 42) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fits Gaussian Hidden Markov Model and returns (labels, model_info)."""
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    model = hmm.GaussianHMM(n_components=k, random_state=random_state, n_iter=100)
    model.fit(X)
    labels = model.predict(X)
    posteriors = model.predict_proba(X)
    log_l = model.score(X)
    n_features = X.shape[1]
    n_params = (
        (k - 1)
        + k * (k - 1)
        + k * n_features
        + k * n_features * (n_features + 1) / 2
    )

    return labels, {
        'log_likelihood': float(log_l),
        'num_params': int(n_params),
        'posteriors': posteriors,
        'transition_matrix': model.transmat_
    }
