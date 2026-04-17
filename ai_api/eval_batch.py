"""
Batch evaluation script for the /evaluation endpoint.

Usage:
    python eval_batch.py                          # defaults
    python eval_batch.py --dataset my_cases.json
    python eval_batch.py --url http://localhost:9000 --collection my_coll
    python eval_batch.py --token <JWT>            # when SKIP_AUTH=false

Output: CSV file with one row per case, saved to eval_results_<timestamp>.csv
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import requests

DEFAULT_URL = "http://localhost:9000"
DEFAULT_DATASET = Path(__file__).parent / "eval_dataset.json"


def parse_args():
    p = argparse.ArgumentParser(description="Batch evaluation of the RAG /evaluation endpoint")
    p.add_argument("--url", default=DEFAULT_URL, help="Base URL of the AI API (default: %(default)s)")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to JSON dataset file (default: %(default)s)")
    p.add_argument("--output", default=None, help="Output CSV path (default: eval_results_<timestamp>.csv)")
    p.add_argument("--token", default=None, help="Bearer JWT token (not needed when SKIP_AUTH=true)")
    p.add_argument("--username", default=None, help="Username / tenant")
    p.add_argument("--collection", default=None, help="Qdrant collection name")
    p.add_argument("--model", default=None, help="LLM model override")
    return p.parse_args()


def call_evaluation(base_url, case, *, token=None, username=None, collection=None, model=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    data = {"question": case["question"]}
    if case.get("reference"):
        data["reference"] = case["reference"]
    if username:
        data["username"] = username
    if collection:
        data["collection"] = collection
    if model:
        data["model"] = model

    resp = requests.post(f"{base_url}/evaluation", data=data, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()


def main():
    args = parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset file not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    with open(dataset_path, encoding="utf-8") as f:
        cases = json.load(f)

    if not cases:
        print("[ERROR] Dataset is empty.", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    print(f"Dataset   : {dataset_path} ({len(cases)} cases)")
    print(f"Endpoint  : {args.url}/evaluation")
    print(f"Output    : {output_path}")
    print()

    fieldnames = [
        "case_id",
        "question",
        "reference",
        "response",
        "n_chunks",
        "retrieved_sources",
        "retrieved_contexts",
        "error",
    ]

    results = []
    ok, failed = 0, 0

    for i, case in enumerate(cases, start=1):
        question = case.get("question", "").strip()
        if not question:
            print(f"  [{i}/{len(cases)}] SKIP — missing question")
            continue

        print(f"  [{i}/{len(cases)}] {question[:80]}...", end=" ", flush=True)
        try:
            result = call_evaluation(
                args.url,
                case,
                token=args.token,
                username=args.username,
                collection=args.collection,
                model=args.model,
            )
            contexts = result.get("retrieved_contexts", [])
            sources = result.get("retrieved_sources", [])
            row = {
                "case_id": i,
                "question": question,
                "reference": case.get("reference", ""),
                "response": result.get("response", ""),
                "n_chunks": len(contexts),
                "retrieved_sources": " | ".join(sources),
                "retrieved_contexts": " ||| ".join(contexts),
                "error": "",
            }
            print(f"OK  ({len(contexts)} chunks)")
            ok += 1
        except Exception as exc:
            row = {
                "case_id": i,
                "question": question,
                "reference": case.get("reference", ""),
                "response": "",
                "n_chunks": 0,
                "retrieved_sources": "",
                "retrieved_contexts": "",
                "error": str(exc),
            }
            print(f"FAIL — {exc}")
            failed += 1

        results.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Done: {ok} OK, {failed} failed — results saved to {output_path}")


if __name__ == "__main__":
    main()
