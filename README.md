# Behavioral Fingerprinting for Prompt Injection Detection

A research framework for detecting prompt injection attacks against LLM agents by modeling **behavioral deviation** rather than inspecting inputs directly.

## Core Idea

Most prompt injection defenses filter malicious inputs at ingestion time. This project takes a different approach: characterize the *expected behavioral distribution* of an agent on a task, then flag statistical anomalies when injected instructions cause the agent to deviate.

This is analogous to anomaly detection in network security — you don't need to know what an attack looks like, only that behavior has changed.

## Project Structure

```
behavioral-fingerprinting/
├── agent/          # LLM agent that performs structured tasks
├── tracer/         # Records behavioral traces per run
├── detector/       # Statistical anomaly detection on traces
├── corpus/         # Prompt injection payloads + benign baselines
├── experiments/    # Experiment runner scripts
└── notebooks/      # Analysis and visualization
```

## Behavioral Features Captured

For each agent run, the tracer records:

| Feature | Description |
|---|---|
| `tool_sequence` | Ordered list of tools called |
| `tool_call_count` | Total number of tool invocations |
| `unique_tools` | Set of distinct tools used |
| `output_length` | Character count of final output |
| `output_entropy` | Shannon entropy of output token distribution |
| `task_completion` | Whether agent returned a valid task result |
| `latency_ms` | Wall-clock time per run |
| `step_count` | Number of reasoning steps taken |
| `external_refs` | Count of URLs/domains referenced in output |
| `refusal_signal` | Whether agent refused or hedged on the task |

## Threat Model

We consider an **indirect prompt injection** adversary who:
- Controls some content the agent retrieves (e.g. a webpage, document, tool result)
- Cannot modify the system prompt or user query directly
- Goals: data exfiltration, goal hijacking, denial of service, or false output injection

## Detection Approach

1. **Baseline collection**: Run agent on N benign tasks, collect behavioral traces
2. **Distribution modeling**: Fit a multivariate model over behavioral features
3. **Runtime scoring**: For each new run, compute Mahalanobis distance from baseline
4. **Alerting**: Flag runs exceeding a threshold (calibrated to target FPR)

## Quickstart

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here  # free at aistudio.google.com

# Collect baseline traces (benign runs)
python experiments/collect_baseline.py --n 50 --output data/baseline.jsonl

# Run injection experiments
python experiments/run_injections.py --corpus corpus/payloads.json --output data/injected.jsonl

# Train detector and evaluate
python experiments/evaluate_detector.py --baseline data/baseline.jsonl --injected data/injected.jsonl

# Visualize results
jupyter notebook notebooks/analysis.ipynb
```

## Results

The detector is evaluated on:
- **AUROC**: Area under ROC curve for benign vs injected classification
- **FPR @ 95% TPR**: False positive rate at 95% true positive detection
- **Detection latency**: Time added per run for anomaly scoring

## Research Questions

1. Which behavioral features are most informative for detecting injection?
2. Does the detector generalize across injection *types* (exfiltration vs hijacking)?
3. How does detection performance degrade as injections become more subtle?
4. Can an adversary craft injections that preserve behavioral signatures?

## Citation

If you use this framework, please cite:
```
@misc{behavioral-fingerprinting-2025,
  title={Behavioral Fingerprinting for Prompt Injection Detection in LLM Agents},
  year={2025},
  url={https://github.com/your-username/behavioral-fingerprinting}
}
```
