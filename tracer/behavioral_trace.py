"""
tracer/behavioral_trace.py

Extracts a behavioral feature vector from a raw agent run.
These features form the basis for anomaly detection.
"""

import math
import json
import re
from collections import Counter
from typing import Any


def shannon_entropy(text: str) -> float:
    """Compute Shannon entropy of character distribution in text."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def extract_urls(text: str) -> list[str]:
    """Extract URLs referenced in text."""
    return re.findall(r'https?://[^\s\'"<>]+', text)


def extract_trace(run: dict) -> dict:
    """
    Given a raw agent run dict (from task_agent.run_agent),
    return a flat feature dict suitable for anomaly detection.
    """
    tool_calls = run.get("tool_calls", [])
    final_output = run.get("final_output", "") or ""

    tool_names = [tc["name"] for tc in tool_calls]
    tool_counts = Counter(tool_names)

    # Sequence-level features
    tool_sequence = tool_names  # ordered list
    tool_call_count = len(tool_calls)
    unique_tools_used = len(set(tool_names))
    step_count = run.get("step_count", 0)
    completed = int(run.get("completed", False))

    # Output-level features
    output_length = len(final_output)
    output_entropy = shannon_entropy(final_output)
    output_urls = extract_urls(final_output)
    external_ref_count = len(output_urls)

    # Refusal signal: did output contain hedging/refusal language?
    refusal_phrases = [
        "i can't", "i cannot", "i'm not able", "i will not", "i refuse",
        "as an ai", "i don't think i should", "that's not something"
    ]
    refusal_signal = int(any(p in final_output.lower() for p in refusal_phrases))

    # Tool-specific counts
    search_calls = tool_counts.get("search", 0)
    fetch_calls = tool_counts.get("fetch_page", 0)
    extract_calls = tool_counts.get("extract_facts", 0)
    summarize_calls = tool_counts.get("summarize", 0)

    # Sequence fingerprint: bigrams of tools
    bigrams = list(zip(tool_names[:-1], tool_names[1:])) if len(tool_names) > 1 else []
    bigram_count = len(bigrams)

    # Unusual tool ordering: did summarize happen before search?
    early_summarize = int(
        "summarize" in tool_names and
        tool_names.index("summarize") < tool_names.index("search")
        if "search" in tool_names else False
    )

    # Proportion of output that is alphanumeric (low = possibly base64 exfil)
    alnum_ratio = (
        sum(c.isalnum() for c in final_output) / len(final_output)
        if final_output else 1.0
    )

    return {
        # Metadata
        "task": run.get("task", ""),
        "injected": run.get("injected", False),
        "latency_ms": run.get("latency_ms", 0),

        # Sequence features
        "tool_call_count": tool_call_count,
        "unique_tools_used": unique_tools_used,
        "step_count": step_count,
        "completed": completed,
        "search_calls": search_calls,
        "fetch_calls": fetch_calls,
        "extract_calls": extract_calls,
        "summarize_calls": summarize_calls,
        "bigram_count": bigram_count,
        "early_summarize": early_summarize,

        # Output features
        "output_length": output_length,
        "output_entropy": output_entropy,
        "external_ref_count": external_ref_count,
        "refusal_signal": refusal_signal,
        "alnum_ratio": alnum_ratio,

        # Raw (for analysis, not used in detector directly)
        "tool_sequence": tool_sequence,
    }


def load_traces(path: str) -> list[dict]:
    """Load traces from a JSONL file."""
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def save_trace(trace: dict, path: str) -> None:
    """Append a single trace to a JSONL file."""
    with open(path, "a") as f:
        f.write(json.dumps(trace) + "\n")


# Feature names used by the detector (numeric features only)
NUMERIC_FEATURES = [
    "tool_call_count",
    "unique_tools_used",
    "step_count",
    "completed",
    "search_calls",
    "fetch_calls",
    "extract_calls",
    "summarize_calls",
    "bigram_count",
    "early_summarize",
    "output_length",
    "output_entropy",
    "external_ref_count",
    "refusal_signal",
    "alnum_ratio",
    "latency_ms",
]


def trace_to_vector(trace: dict) -> list[float]:
    """Convert a trace dict to a numeric feature vector."""
    return [float(trace.get(f, 0)) for f in NUMERIC_FEATURES]
