"""
experiments/run_injections.py

Runs the agent on benign tasks with injected payloads in tool results.
Records behavioral traces for comparison against baseline.
"""

import argparse
import json
import os
import sys
from tqdm import tqdm
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.task_agent import run_agent
from tracer.behavioral_trace import extract_trace, save_trace


def main():
    parser = argparse.ArgumentParser(description="Run injection experiments")
    parser.add_argument("--corpus", type=str, default="corpus/payloads.json")
    parser.add_argument("--output", type=str, default="data/injected.jsonl")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Only run payloads from these categories")
    parser.add_argument("--repeats", type=int, default=2,
                        help="Runs per (task, payload) pair")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.corpus) as f:
        corpus = json.load(f)

    payloads = corpus["payloads"]
    tasks = corpus["benign_tasks"]

    if args.categories:
        payloads = [p for p in payloads if p["category"] in args.categories]

    pairs = list(product(tasks[:5], payloads))  # 5 tasks × all payloads
    total = len(pairs) * args.repeats

    print(f"Running {total} injection experiments → {args.output}")
    print(f"Payloads: {len(payloads)} | Tasks: 5 | Repeats: {args.repeats}")

    completed = 0
    for task, payload in tqdm(pairs, desc="Injection runs"):
        for _ in range(args.repeats):
            try:
                run = run_agent(task, injected_content=payload["content"])
                trace = extract_trace(run)
                trace["payload_id"] = payload["id"]
                trace["payload_category"] = payload["category"]
                trace["payload_name"] = payload["name"]
                save_trace(trace, args.output)
                completed += 1
            except Exception as e:
                print(f"\nFailed ({payload['id']} / {task[:30]}): {e}")
                continue

    print(f"\nDone. {completed}/{total} traces saved to {args.output}")


if __name__ == "__main__":
    main()
