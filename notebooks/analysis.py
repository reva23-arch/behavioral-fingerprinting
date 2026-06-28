"""
notebooks/analysis.py

Analysis and visualization of detector results.
Run with: python notebooks/analysis.py
Or convert to Jupyter: jupytext --to notebook notebooks/analysis.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracer.behavioral_trace import load_traces, trace_to_vector, NUMERIC_FEATURES
from detector.anomaly_detector import MahalanobisDetector

# ── Load data ──────────────────────────────────────────────────────────────
baseline_traces = load_traces("data/baseline.jsonl")
injected_traces = load_traces("data/injected.jsonl")
with open("data/results.json") as f:
    results = json.load(f)

X_baseline = np.array([trace_to_vector(t) for t in baseline_traces])
X_injected = np.array([trace_to_vector(t) for t in injected_traces])

# Refit detector for scoring
n_val = max(1, int(len(X_baseline) * 0.3))
X_train = X_baseline[:-n_val]
X_val = X_baseline[-n_val:]
maha = MahalanobisDetector()
maha.fit(X_train, feature_names=NUMERIC_FEATURES)
maha.calibrate_threshold(X_val, target_fpr=0.05)

scores_benign = maha.score_batch(X_val)
scores_injected = maha.score_batch(X_injected)

os.makedirs("figures", exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
BLUE = "#2563EB"
RED = "#DC2626"
GRAY = "#6B7280"

# ── Figure 1: Score distributions ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(scores_benign, bins=20, alpha=0.7, color=BLUE, label="Benign", density=True)
ax.hist(scores_injected, bins=20, alpha=0.7, color=RED, label="Injected", density=True)
ax.axvline(maha.threshold_, color="black", linestyle="--", linewidth=1.5,
           label=f"Threshold ({maha.threshold_:.2f})")
ax.set_xlabel("Mahalanobis Distance", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Behavioral Score Distribution: Benign vs Injected", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("figures/score_distributions.png", dpi=150)
plt.close()
print("Saved: figures/score_distributions.png")

# ── Figure 2: ROC curves ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
for name, color in [("mahalanobis", BLUE), ("isolation_forest", RED)]:
    r = results[name]
    fpr = r["fpr_curve"]
    tpr = r["tpr_curve"]
    auroc = r["auroc"]
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{r['detector']} (AUROC={auroc:.3f})")
ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves: Prompt Injection Detection", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("figures/roc_curves.png", dpi=150)
plt.close()
print("Saved: figures/roc_curves.png")

# ── Figure 3: Feature importance ───────────────────────────────────────────
mean_b = np.array(results["mean_benign"])
mean_i = np.array(results["mean_injected"])
diffs = np.abs(mean_i - mean_b)
sorted_idx = np.argsort(diffs)[::-1][:10]
feat_names = [NUMERIC_FEATURES[i] for i in sorted_idx]
feat_diffs = diffs[sorted_idx]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(feat_names[::-1], feat_diffs[::-1], color=BLUE, alpha=0.85)
ax.set_xlabel("|Δ mean| (Injected − Benign)", fontsize=12)
ax.set_title("Top Features by Mean Shift Under Injection", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=150)
plt.close()
print("Saved: figures/feature_importance.png")

# ── Figure 4: Per-category detection rates ─────────────────────────────────
cats = results["per_category"]
cat_names = list(cats.keys())
detection_rates = [cats[c]["detection_rate"] for c in cat_names]
n_per_cat = [cats[c]["n"] for c in cat_names]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(cat_names, detection_rates, color=BLUE, alpha=0.85)
ax.set_ylabel("Detection Rate", fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_title("Detection Rate by Injection Category", fontsize=13, fontweight="bold")
for bar, n in zip(bars, n_per_cat):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"n={n}", ha="center", va="bottom", fontsize=9)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("figures/detection_by_category.png", dpi=150)
plt.close()
print("Saved: figures/detection_by_category.png")

# ── Figure 5: Feature correlation heatmap (baseline) ──────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
corr = np.corrcoef(X_baseline.T)
sns.heatmap(corr, xticklabels=NUMERIC_FEATURES, yticklabels=NUMERIC_FEATURES,
            cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
            annot=False, linewidths=0.3)
ax.set_title("Behavioral Feature Correlations (Baseline)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/feature_correlations.png", dpi=150)
plt.close()
print("Saved: figures/feature_correlations.png")

print("\nAll figures saved to figures/")
print(f"\nSummary:")
print(f"  Mahalanobis AUROC: {results['mahalanobis']['auroc']:.4f}")
print(f"  Isolation Forest AUROC: {results['isolation_forest']['auroc']:.4f}")
print(f"  Mahalanobis FPR@95TPR: {results['mahalanobis']['fpr_at_95tpr']:.4f}")
