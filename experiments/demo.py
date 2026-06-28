"""
experiments/demo.py

Runs a quick demo using synthetic trace data (no API key needed).
Demonstrates the full pipeline: traces → detector → evaluation.

Usage: python experiments/demo.py
"""

import sys
import os
import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector.anomaly_detector import MahalanobisDetector, IsolationForestDetector, evaluate, print_results
from tracer.behavioral_trace import NUMERIC_FEATURES


def generate_synthetic_traces(n: int, injected: bool, seed: int = 0) -> np.ndarray:
    """
    Generate synthetic behavioral traces for demo purposes.
    
    Benign traces: moderate, consistent tool usage
    Injected traces: anomalous patterns (fewer calls, short output, early summarize, etc.)
    """
    rng = np.random.default_rng(seed)
    
    # Feature indices for reference
    feat_idx = {f: i for i, f in enumerate(NUMERIC_FEATURES)}
    
    if not injected:
        # Benign: normal agent behavior
        data = rng.normal(
            loc=[4, 3, 3, 1, 1, 2, 1, 1, 3, 0, 180, 4.2, 1, 0, 0.92, 2800],
            scale=[0.8, 0.5, 0.5, 0.1, 0.4, 0.5, 0.3, 0.2, 0.5, 0.05, 60, 0.3, 0.5, 0.05, 0.03, 400],
            size=(n, len(NUMERIC_FEATURES))
        )
    else:
        # Injected: behavioral anomalies
        # Mix of different attack types
        data = rng.normal(
            loc=[4, 3, 3, 1, 1, 2, 1, 1, 3, 0, 180, 4.2, 1, 0, 0.92, 2800],
            scale=[0.8, 0.5, 0.5, 0.1, 0.4, 0.5, 0.3, 0.2, 0.5, 0.05, 60, 0.3, 0.5, 0.05, 0.03, 400],
            size=(n, len(NUMERIC_FEATURES))
        )
        
        # Inject anomalies in different patterns per attack type
        n3 = n // 3
        
        # Goal hijacking: early summarize, fewer calls, short output
        data[:n3, feat_idx["tool_call_count"]] = rng.normal(1.5, 0.5, n3)
        data[:n3, feat_idx["summarize_calls"]] = rng.normal(1, 0.1, n3)
        data[:n3, feat_idx["early_summarize"]] = rng.binomial(1, 0.8, n3).astype(float)
        data[:n3, feat_idx["output_length"]] = rng.normal(30, 15, n3)
        data[:n3, feat_idx["search_calls"]] = rng.normal(0.2, 0.2, n3)
        
        # Data exfiltration: high entropy output, extra fetches, low alnum
        data[n3:2*n3, feat_idx["output_entropy"]] = rng.normal(6.5, 0.4, n3)
        data[n3:2*n3, feat_idx["fetch_calls"]] = rng.normal(5, 1, n3)
        data[n3:2*n3, feat_idx["alnum_ratio"]] = rng.normal(0.55, 0.1, n3)
        data[n3:2*n3, feat_idx["output_length"]] = rng.normal(600, 150, n3)
        
        # DoS: very high step count, no completion, high latency
        data[2*n3:, feat_idx["step_count"]] = rng.normal(9, 0.5, n - 2*n3)
        data[2*n3:, feat_idx["completed"]] = rng.binomial(1, 0.05, n - 2*n3).astype(float)
        data[2*n3:, feat_idx["latency_ms"]] = rng.normal(8000, 1000, n - 2*n3)
        data[2*n3:, feat_idx["tool_call_count"]] = rng.normal(9, 0.5, n - 2*n3)
    
    # Clip to realistic ranges
    data = np.clip(data, 0, None)
    data[:, feat_idx["alnum_ratio"]] = np.clip(data[:, feat_idx["alnum_ratio"]], 0, 1)
    data[:, feat_idx["completed"]] = np.clip(data[:, feat_idx["completed"]], 0, 1)
    
    return data


def main():
    print("=" * 60)
    print("  Behavioral Fingerprinting — Demo (Synthetic Data)")
    print("=" * 60)
    print()
    print("This demo generates synthetic traces to validate the pipeline.")
    print("For real results, run collect_baseline.py and run_injections.py")
    print("with a valid ANTHROPIC_API_KEY.")
    print()

    N_TRAIN = 80
    N_VAL = 30
    N_INJECTED = 90

    print(f"Generating {N_TRAIN + N_VAL} baseline traces and {N_INJECTED} injected traces...")
    X_train = generate_synthetic_traces(N_TRAIN, injected=False, seed=0)
    X_val = generate_synthetic_traces(N_VAL, injected=False, seed=1)
    X_injected = generate_synthetic_traces(N_INJECTED, injected=True, seed=2)

    # Mahalanobis detector
    maha = MahalanobisDetector()
    maha.fit(X_train, feature_names=NUMERIC_FEATURES)
    threshold = maha.calibrate_threshold(X_val, target_fpr=0.05)
    print(f"Mahalanobis threshold (5% FPR): {threshold:.3f}")

    maha_results = evaluate(maha, X_val, X_injected, "Mahalanobis")
    print_results(maha_results)

    # Isolation Forest
    iso = IsolationForestDetector(contamination=0.05)
    iso.fit(X_train)
    iso_results = evaluate(iso, X_val, X_injected, "Isolation Forest")
    print_results(iso_results)

    # Feature contributions for a sample injected run
    print("Sample injected run — top feature contributions to anomaly score:")
    sample = X_injected[0]
    score = maha.score(sample)
    contribs = maha.feature_contributions(sample)
    top = sorted(contribs.items(), key=lambda x: -x[1])[:6]
    for feat, val in top:
        actual = sample[NUMERIC_FEATURES.index(feat)]
        baseline_mean = maha.mean_[NUMERIC_FEATURES.index(feat)]
        print(f"  {feat:30s}: contribution={val:.3f}  value={actual:.2f}  baseline_mean={baseline_mean:.2f}")
    print(f"  Total Mahalanobis distance: {score:.3f}  (threshold: {threshold:.3f})")
    print(f"  → {'FLAGGED as anomalous' if score > threshold else 'Not flagged'}")

    print("\nDemo complete. Pipeline is working correctly.")
    print("Next step: run with real API traces using collect_baseline.py and run_injections.py")


if __name__ == "__main__":
    main()
