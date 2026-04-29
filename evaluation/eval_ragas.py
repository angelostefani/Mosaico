"""
RAGAS evaluation script for Mosaico.

Reads a CSV produced by eval_batch.py and computes RAGAS metrics using a local
Ollama LLM and local HuggingFace embeddings (no OpenAI required).

Usage:
    python eval_ragas.py --input risultati01.csv --judge-model qwen3:8b
    python eval_ragas.py --input risultati01.csv --metrics context_recall,answer_correctness --batch-size 5 --verbose
    python eval_ragas.py --input risultati01.csv --output ragas_full.csv

Expected runtime with gemma4:e4b generator + qwen3:8b judge: ~1 hour for 100 samples x 5 metrics on H100.
Start with --metrics context_recall,answer_correctness for a faster first run.

Caveats for small local LLMs (1B params):
- Expect 15-30% NaN values where the LLM-judge fails to return valid JSON.
- faithfulness scores are biased upward (same model generates and judges).
- RAGAS prompts are in English; Italian corpus may reduce context_recall quality.
- Use scores for relative comparison between configurations, not as absolute quality.
"""

import argparse
import csv
import math
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

CONTEXT_SEPARATOR = " ||| "

ALL_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]


def _ollama_base_url(raw_url: str) -> str:
    """Strip /api/generate suffix so ChatOllama can append its own /api/chat path."""
    return raw_url.split("/api/")[0] if "/api/" in raw_url else raw_url.rstrip("/")


def build_llm(base_url: str, model: str, timeout: int, max_tokens: int = 8192, no_think: bool = False):
    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}\nRun: pip install ragas langchain-openai", file=sys.stderr)
        sys.exit(1)

    # extra_body must live in model_kwargs so langchain-openai merges it into every
    # request body — including instructor-patched calls ragas uses for JSON extraction.
    model_kwargs = {}
    if no_think:
        model_kwargs["extra_body"] = {"think": False}

    chat_model = ChatOpenAI(
        model=model,
        base_url=f"{base_url}/v1",
        api_key="ollama",
        max_tokens=max_tokens,
        timeout=timeout,
        model_kwargs=model_kwargs,
    )
    return LangchainLLMWrapper(chat_model)


def build_embeddings(model_name: str):
    try:
        from ragas.embeddings import HuggingFaceEmbeddings, BaseRagasEmbeddings
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}\nRun: pip install ragas sentence-transformers", file=sys.stderr)
        sys.exit(1)

    _inner = HuggingFaceEmbeddings(model=model_name)

    # ragas.embeddings.HuggingFaceEmbeddings uses the newer BaseRagasEmbedding interface
    # (embed_text / embed_texts), but some metrics use the Langchain-compatible
    # BaseRagasEmbeddings interface (embed_query / embed_documents / embed_text).
    # BaseRagasEmbeddings.embed_text delegates to embed_texts which requires run_config;
    # override all four paths to delegate directly to _inner and avoid that dependency.
    class _Adapter(BaseRagasEmbeddings):
        def embed_query(self, text: str):
            return _inner.embed_text(text)

        def embed_documents(self, texts):
            return _inner.embed_texts(texts)

        async def aembed_query(self, text: str):
            return await _inner.aembed_text(text)

        async def aembed_documents(self, texts):
            return await _inner.aembed_texts(texts)

        async def embed_text(self, text: str, is_async: bool = True):
            return await _inner.aembed_text(text)

        async def embed_texts(self, texts, is_async: bool = True):
            return await _inner.aembed_texts(texts)

    return _Adapter()


def load_csv_as_samples(path: str):
    """Parse eval_batch.py CSV into RAGAS SingleTurnSample list.

    Returns (samples, skipped) where skipped is a list of raw row dicts
    that could not be converted (error field set or no contexts).
    """
    try:
        from ragas import SingleTurnSample
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}\nRun: pip install ragas", file=sys.stderr)
        sys.exit(1)

    required_cols = {"case_id", "question", "response", "retrieved_contexts"}

    samples = []
    skipped = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not required_cols.issubset(set(reader.fieldnames or [])):
            missing = required_cols - set(reader.fieldnames or [])
            print(f"[ERROR] CSV missing columns: {missing}", file=sys.stderr)
            sys.exit(1)

        for row in reader:
            if row.get("error", "").strip():
                row["_skip_reason"] = f"RAG error: {row['error']}"
                skipped.append(row)
                continue

            raw_contexts = row.get("retrieved_contexts", "").strip()
            if not raw_contexts:
                row["_skip_reason"] = "empty retrieved_contexts"
                skipped.append(row)
                continue

            contexts = [c.strip() for c in raw_contexts.split(CONTEXT_SEPARATOR) if c.strip()]
            if not contexts:
                row["_skip_reason"] = "no contexts after split"
                skipped.append(row)
                continue

            reference = row.get("reference", "").strip() or None

            sample = SingleTurnSample(
                user_input=row["question"],
                response=row["response"],
                reference=reference,
                retrieved_contexts=contexts,
            )
            sample._case_id = row["case_id"]
            sample._n_chunks = row.get("n_chunks", "")
            samples.append(sample)

    return samples, skipped


def load_already_evaluated(output_path: str) -> tuple[set[str], list[dict]]:
    """Se il file di output esiste, ritorna (case_id già valutati, righe esistenti)."""
    if not Path(output_path).exists():
        return set(), []
    evaluated_ids: set[str] = set()
    existing_rows: list[dict] = []
    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("case_id", "").strip()
            if cid:
                evaluated_ids.add(cid)
                existing_rows.append(dict(row))
    return evaluated_ids, existing_rows


def pick_metrics(metric_names: list[str], llm, embeddings, strictness: int = 1):
    try:
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._context_precision import ContextPrecision
        from ragas.metrics._context_recall import ContextRecall
        from ragas.metrics._answer_correctness import AnswerCorrectness
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}\nRun: pip install ragas", file=sys.stderr)
        sys.exit(1)

    registry = {
        "faithfulness": lambda: Faithfulness(llm=llm),
        "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=strictness),
        "context_precision": lambda: ContextPrecision(llm=llm),
        "context_recall": lambda: ContextRecall(llm=llm),
        "answer_correctness": lambda: AnswerCorrectness(llm=llm, embeddings=embeddings),
    }

    metrics = []
    for name in metric_names:
        if name not in registry:
            print(f"[WARN] Unknown metric '{name}', skipping.", file=sys.stderr)
            continue
        metrics.append(registry[name]())

    if not metrics:
        print("[ERROR] No valid metrics selected.", file=sys.stderr)
        sys.exit(1)

    return metrics


def evaluate_in_batches(samples, metrics, batch_size: int, verbose: bool, timeout: int) -> list[dict]:
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.run_config import RunConfig
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}\nRun: pip install ragas", file=sys.stderr)
        sys.exit(1)

    # max_workers=1 prevents concurrent requests to a single Ollama instance,
    # which would cause queue buildup and cascading timeouts.
    # log_tenacity=True surfaces retry attempts to help diagnose judge LLM failures.
    run_config = RunConfig(timeout=timeout, max_workers=1, log_tenacity=True)

    results = []
    total = len(samples)
    num_batches = math.ceil(total / batch_size)

    try:
        for b in range(num_batches):
            batch = samples[b * batch_size : (b + 1) * batch_size]
            print(f"  Batch {b + 1}/{num_batches} ({len(batch)} samples)...", flush=True)

            dataset = EvaluationDataset(samples=batch)

            try:
                result_ds = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)
                df = result_ds.to_pandas()
            except Exception as exc:
                print(f"  [WARN] Batch {b + 1} failed: {exc} — recording NaN for all samples.")
                for s in batch:
                    row = {
                        "case_id": getattr(s, "_case_id", ""),
                        "question": s.user_input,
                        "reference": s.reference or "",
                        "response": s.response,
                        "n_chunks": getattr(s, "_n_chunks", ""),
                        "error_ragas": str(exc),
                    }
                    for m in metrics:
                        row[m.name] = float("nan")
                    row["judge_warning"] = "batch_error"
                    results.append(row)
                continue

            for i, s in enumerate(batch):
                row = {
                    "case_id": getattr(s, "_case_id", ""),
                    "question": s.user_input,
                    "reference": s.reference or "",
                    "response": s.response,
                    "n_chunks": getattr(s, "_n_chunks", ""),
                    "error_ragas": "",
                }
                for m in metrics:
                    col = m.name
                    row[col] = df.iloc[i].get(col, float("nan")) if i < len(df) else float("nan")

                cp = row.get("context_precision", float("nan"))
                cr = row.get("context_recall", float("nan"))
                ff = row.get("faithfulness", 0.0) or 0.0
                row["judge_warning"] = (
                    "suspicious_zero"
                    if (cp == 0.0 and cr == 0.0 and ff > 0.3)
                    else ""
                )
                results.append(row)

                if verbose:
                    scores = {m.name: f"{row[m.name]:.3f}" if not math.isnan(row[m.name]) else "NaN" for m in metrics}
                    print(f"    [{row['case_id']}] {s.user_input[:60]}... → {scores}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted — saving partial results.")

    return results


def write_output(results: list[dict], skipped: list[dict], metric_names: list[str], output_path: str, input_path: str, existing_rows: list[dict] | None = None):
    metric_cols = [m for m in ALL_METRICS if m in metric_names]
    fieldnames = ["case_id", "question", "reference", "response", "n_chunks"] + metric_cols + ["error_ragas", "judge_warning"]

    for row in skipped:
        entry = {f: "" for f in fieldnames}
        entry["case_id"] = row.get("case_id", "")
        entry["question"] = row.get("question", "")
        entry["reference"] = row.get("reference", "")
        entry["response"] = row.get("response", "")
        entry["n_chunks"] = row.get("n_chunks", "")
        entry["error_ragas"] = row.get("_skip_reason", "skipped")
        results.append(entry)

    if existing_rows:
        results = list(existing_rows) + results

    results.sort(key=lambda r: int(r["case_id"]) if str(r["case_id"]).isdigit() else 0)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    n_resumed = len(existing_rows) if existing_rows else 0
    n_new = len(results) - len(skipped) - n_resumed
    print(f"\nResults saved to {output_path}")
    resumed_str = f" | Resumed: {n_resumed}" if n_resumed else ""
    print(f"Input: {input_path} | Evaluated: {n_new}{resumed_str} | Skipped: {len(skipped)}")

    # Summary statistics
    import statistics

    print("\nMetric summary:")
    for col in metric_cols:
        vals = [r[col] for r in results if isinstance(r.get(col), float) and not math.isnan(r[col])]
        nan_count = sum(1 for r in results if isinstance(r.get(col), float) and math.isnan(r[col]))
        if vals:
            print(f"  {col:<25} mean={statistics.mean(vals):.4f}  std={statistics.pstdev(vals):.4f}  NaN={nan_count}")
        else:
            print(f"  {col:<25} all NaN ({nan_count} samples)")

    flagged = sum(1 for r in results if r.get("judge_warning") == "suspicious_zero")
    if flagged:
        print(f"\n  [WARN] {flagged} row(s) flagged 'suspicious_zero' (context_precision=0 & context_recall=0 & faithfulness>0.3 — likely judge LLM failure)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute RAGAS metrics on eval_batch.py CSV output using local Ollama + HuggingFace embeddings."
    )
    p.add_argument("--input", required=True, help="CSV file produced by eval_batch.py")
    p.add_argument("--output", default=None, help="Output CSV path (default: ragas_<timestamp>.csv)")
    p.add_argument("--ollama-url", default=None, help="Override OLLAMA_URL env var")
    p.add_argument("--model", default=None, help="Override OLLAMA_MODEL env var")
    p.add_argument("--judge-model", default=None,
        help="Separate Ollama model for RAGAS judging (default: same as --model)")
    p.add_argument("--no-think", action="store_true",
        help="Disable thinking mode for the judge model (qwen3 and similar; passes think=false)")
    p.add_argument("--embedding-model", default=None, help="HuggingFace embedding model (default: all-MiniLM-L6-v2)")
    p.add_argument(
        "--metrics",
        default=",".join(ALL_METRICS),
        help=f"Comma-separated metrics to compute (default: all). Choices: {', '.join(ALL_METRICS)}",
    )
    p.add_argument("--batch-size", type=int, default=10, help="Samples per evaluate() call (default: 10)")
    p.add_argument("--timeout", type=int, default=600, help="Seconds per LLM call (default: 600)")
    p.add_argument("--max-tokens", type=int, default=8192, help="Max output tokens per LLM call (default: 8192; increase for thinking models)")
    p.add_argument("--strictness", type=int, default=1,
        help="Generations requested per sample for AnswerRelevancy (default: 1; ragas default: 3 — causes warnings with Ollama)")
    p.add_argument("--verbose", action="store_true", help="Print per-sample scores")
    return p.parse_args()


def main():
    load_dotenv(Path(__file__).parent.parent / ".env")

    args = parse_args()

    input_path = args.input
    if not Path(input_path).exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or f"ragas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    raw_url = args.ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    model = args.model or os.environ.get("OLLAMA_MODEL", "gemma3:1b")
    judge_model = args.judge_model or model
    emb_model = args.embedding_model or os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    base_url = _ollama_base_url(raw_url)
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]

    print(f"Input     : {input_path}")
    print(f"Output    : {output_path}")
    print(f"Ollama    : {base_url}  model={model}  judge={judge_model}")
    print(f"Embeddings: {emb_model}")
    print(f"Metrics   : {', '.join(metric_names)}")
    print(f"Batch size: {args.batch_size}")
    print()

    print(f"Max tokens: {args.max_tokens}")
    print("Loading LLM and embeddings...")
    llm = build_llm(base_url, model, args.timeout, args.max_tokens)
    judge_llm = build_llm(base_url, judge_model, args.timeout, args.max_tokens, no_think=args.no_think) if args.judge_model else llm
    embeddings = build_embeddings(emb_model)

    print("Parsing CSV...")
    samples, skipped = load_csv_as_samples(input_path)
    print(f"  {len(samples)} samples to evaluate, {len(skipped)} skipped")

    evaluated_ids, existing_rows = load_already_evaluated(output_path)
    if evaluated_ids:
        print(f"  Resume: {len(evaluated_ids)} case(s) già in output — skipped.")
        samples = [s for s in samples if getattr(s, "_case_id", "") not in evaluated_ids]
        print(f"  {len(samples)} campioni rimanenti.")
    else:
        existing_rows = []

    if not samples and not existing_rows:
        print("[ERROR] No valid samples found in CSV.", file=sys.stderr)
        sys.exit(1)

    if not samples:
        print("  Tutti i campioni già valutati — nessun nuovo lavoro da fare.")
        write_output([], skipped, metric_names, output_path, input_path, existing_rows=existing_rows)
        return

    metrics = pick_metrics(metric_names, judge_llm, embeddings, strictness=args.strictness)
    print(f"  Metrics instantiated: {[m.name for m in metrics]}  (strictness={args.strictness})")
    print()

    results = evaluate_in_batches(samples, metrics, args.batch_size, args.verbose, args.timeout)

    write_output(results, skipped, [m.name for m in metrics], output_path, input_path, existing_rows=existing_rows)


if __name__ == "__main__":
    main()
