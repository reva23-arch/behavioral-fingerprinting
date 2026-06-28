"""
experiments/collect_baseline.py

Runs the agent on benign tasks to collect baseline behavioral traces.
"""

import argparse
import json
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.task_agent import run_agent
from tracer.behavioral_trace import extract_trace, save_trace


def main():
    parser = argparse.ArgumentParser(description="Collect baseline behavioral traces")
    parser.add_argument("--n", type=int, default=20, help="Number of runs per task")
    parser.add_argument("--output", type=str, default="data/baseline.jsonl")
    parser.add_argument("--tasks-file", type=str, default="corpus/payloads.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.tasks_file) as f:
        corpus = json.load(f)

    tasks = corpus["benign_tasks"]

    # If n > len(tasks), cycle through tasks
    task_list = []
    while len(task_list) < args.n:
        task_list.extend(tasks)
    task_list = task_list[:args.n]

    print(f"Collecting {args.n} baseline traces → {args.output}")

    for i, task in enumerate(tqdm(task_list, desc="Baseline runs")):
        try:
            run = run_agent(task, injected_content=None)
            trace = extract_trace(run)
            save_trace(trace, args.output)
        except Exception as e:
            print(f"\nRun {i} failed: {e}")
            continue

    print(f"\nDone. Traces saved to {args.output}")


if __name__ == "__main__":
    main()
