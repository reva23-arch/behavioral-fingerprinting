"""
experiments/evaluate_detector.py

Trains and evaluates both detectors on collected traces.
Outputs metrics, per-category breakdown, and saves results for plotting.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracer.behavioral_trace import load_traces, trace_to_vector, NUMERIC_FEATURES
from detector.anomaly_detector import (
    MahalanobisDetector,
    IsolationForestDetector,
    evaluate,
    print_results,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detectors")
    parser.add_argument("--baseline", type=str, default="data/baseline.jsonl")
    parser.add_argument("--injected", type=str, default="data/injected.jsonl")
    parser.add_argument("--output", type=str, default="data/results.json")
    parser.add_argument("--val-split", type=float, default=0.3,
                        help="Fraction of baseline held out for threshold calibration")
    args = parser.parse_args()

    # Load traces
    print("Loading traces...")
    baseline_traces = load_traces(args.baseline)
    injected_traces = load_traces(args.injected)
    print(f"  Baseline: {len(baseline_traces)} traces")
    print(f"  Injected: {len(injected_traces)} traces")

    # Convert to arrays
    X_baseline = np.array([trace_to_vector(t) for t in baseline_traces])
    X_injected = np.array([trace_to_vector(t) for t in injected_traces])

    # Train/val split on baseline
    n_val = max(1, int(len(X_baseline) * args.val_split))
    X_train = X_baseline[:-n_val]
    X_val = X_baseline[-n_val:]
    print(f"  Train: {len(X_train)} | Val: {len(X_val)}")

    # --- Mahalanobis Detector ---
    print("\nFitting Mahalanobis detector...")
    maha = MahalanobisDetector()
    maha.fit(X_train, feature_names=NUMERIC_FEATURES)
    maha.calibrate_threshold(X_val, target_fpr=0.05)
    print(f"  Threshold @ 5% FPR: {maha.threshold_:.4f}")

    maha_results = evaluate(maha, X_val, X_injected, "Mahalanobis")
    print_results(maha_results)

    # --- Isolation Forest ---
    print("Fitting Isolation Forest detector...")
    iso = IsolationForestDetector(contamination=0.05)
    iso.fit(X_train)

    iso_results = evaluate(iso, X_val, X_injected, "Isolation Forest")
    print_results(iso_results)

    # --- Per-category breakdown ---
    print("Per-category breakdown (Mahalanobis):")
    categories = defaultdict(list)
    for i, trace in enumerate(injected_traces):
        cat = trace.get("payload_category", "unknown")
        categories[cat].append(i)

    category_results = {}
    for cat, idxs in categories.items():
        X_cat = X_injected[idxs]
        scores = maha.score_batch(X_cat)
        detected = (scores > maha.threshold_).sum()
        cat_result = {
            "n": len(idxs),
            "detected": int(detected),
            "detection_rate": round(float(detected / len(idxs)), 3),
            "mean_score": round(float(scores.mean()), 4),
        }
        category_results[cat] = cat_result
        print(f"  {cat:25s}: {detected}/{len(idxs)} detected "
              f"({cat_result['detection_rate']*100:.0f}%) "
              f"mean score={cat_result['mean_score']:.3f}")

    # --- Feature importance ---
    print("\nTop features by contribution (mean injected vs benign):")
    mean_benign = X_val.mean(axis=0)
    mean_injected = X_injected.mean(axis=0)
    diffs = np.abs(mean_injected - mean_benign)
    ranked = sorted(zip(NUMERIC_FEATURES, diffs), key=lambda x: -x[1])
    for feat, diff in ranked[:8]:
        b = mean_benign[NUMERIC_FEATURES.index(feat)]
        inj = mean_injected[NUMERIC_FEATURES.index(feat)]
        print(f"  {feat:30s}: benign={b:.3f}  injected={inj:.3f}  Δ={diff:.3f}")

    # Save results
    output = {
        "mahalanobis": maha_results,
        "isolation_forest": iso_results,
        "per_category": category_results,
        "feature_names": NUMERIC_FEATURES,
        "mean_benign": mean_benign.tolist(),
        "mean_injected": mean_injected.tolist(),
        "threshold": maha.threshold_,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
