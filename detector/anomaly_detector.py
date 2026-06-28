"""
detector/anomaly_detector.py

Fits a multivariate Gaussian over baseline behavioral traces,
then scores new runs by Mahalanobis distance from the baseline distribution.

Also supports an Isolation Forest baseline for comparison.
"""

import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy import linalg
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import json


class MahalanobisDetector:
    """
    Models baseline behavior as a multivariate Gaussian.
    Scores runs by their Mahalanobis distance from the mean.
    High distance = anomalous = likely injected.
    """

    def __init__(self, regularization: float = 1e-4):
        self.regularization = regularization
        self.mean_ = None
        self.cov_inv_ = None
        self.threshold_ = None
        self.feature_names_ = None

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> "MahalanobisDetector":
        """
        Fit on baseline (benign) traces.
        X: (n_samples, n_features) array
        """
        self.mean_ = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        # Regularize to handle near-singular covariance
        cov += self.regularization * np.eye(cov.shape[0])
        self.cov_inv_ = linalg.inv(cov)
        self.feature_names_ = feature_names
        return self

    def score(self, x: np.ndarray) -> float:
        """Return Mahalanobis distance for a single sample."""
        diff = x - self.mean_
        return float(np.sqrt(diff @ self.cov_inv_ @ diff))

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Return Mahalanobis distances for a batch."""
        return np.array([self.score(x) for x in X])

    def calibrate_threshold(self, X_val: np.ndarray, target_fpr: float = 0.05) -> float:
        """
        Set detection threshold using held-out benign data.
        Threshold = percentile of score distribution at (1 - target_fpr).
        """
        scores = self.score_batch(X_val)
        self.threshold_ = float(np.percentile(scores, (1 - target_fpr) * 100))
        return self.threshold_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 (anomalous) or 0 (normal) for each sample."""
        if self.threshold_ is None:
            raise RuntimeError("Call calibrate_threshold() first.")
        scores = self.score_batch(X)
        return (scores > self.threshold_).astype(int)

    def feature_contributions(self, x: np.ndarray) -> dict:
        """
        Return per-feature contribution to Mahalanobis distance.
        Uses diagonal approximation for interpretability.
        """
        if self.feature_names_ is None:
            return {}
        diff = x - self.mean_
        # Diagonal contribution (ignores cross-feature correlations, but interpretable)
        diag_cov_inv = np.diag(self.cov_inv_)
        contribs = diff ** 2 * diag_cov_inv
        return dict(zip(self.feature_names_, contribs.tolist()))


class IsolationForestDetector:
    """
    Alternative detector using Isolation Forest.
    Useful as a baseline comparison.
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        X_scaled = self.scaler.transform(X)
        # sklearn returns negative scores; flip so higher = more anomalous
        return -self.model.score_samples(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        raw = self.model.predict(X_scaled)
        return (raw == -1).astype(int)  # -1 = anomaly in sklearn


def evaluate(
    detector,
    X_benign: np.ndarray,
    X_injected: np.ndarray,
    detector_name: str = "Detector"
) -> dict:
    """
    Evaluate a detector on held-out benign and injected traces.
    Returns AUROC, FPR @ 95% TPR, and threshold info.
    """
    y_true = np.concatenate([
        np.zeros(len(X_benign)),
        np.ones(len(X_injected))
    ])

    if hasattr(detector, "score_batch"):
        scores = np.concatenate([
            detector.score_batch(X_benign),
            detector.score_batch(X_injected)
        ])
    else:
        X_all = np.vstack([X_benign, X_injected])
        scores = detector.score(X_all)

    auroc = roc_auc_score(y_true, scores)

    # FPR @ 95% TPR
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    idx_95 = np.searchsorted(tpr, 0.95)
    fpr_at_95_tpr = float(fpr[min(idx_95, len(fpr) - 1)])

    return {
        "detector": detector_name,
        "auroc": round(auroc, 4),
        "fpr_at_95tpr": round(fpr_at_95_tpr, 4),
        "n_benign": len(X_benign),
        "n_injected": len(X_injected),
        "scores_benign_mean": round(float(scores[:len(X_benign)].mean()), 4),
        "scores_injected_mean": round(float(scores[len(X_benign):].mean()), 4),
        "fpr_curve": fpr.tolist(),
        "tpr_curve": tpr.tolist(),
    }


def print_results(results: dict) -> None:
    print(f"\n{'='*50}")
    print(f"  {results['detector']}")
    print(f"{'='*50}")
    print(f"  AUROC:              {results['auroc']:.4f}")
    print(f"  FPR @ 95% TPR:      {results['fpr_at_95tpr']:.4f}")
    print(f"  Benign score mean:  {results['scores_benign_mean']:.4f}")
    print(f"  Injected score mean:{results['scores_injected_mean']:.4f}")
    print(f"  N benign:           {results['n_benign']}")
    print(f"  N injected:         {results['n_injected']}")
    print(f"{'='*50}\n")
