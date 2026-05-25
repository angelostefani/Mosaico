"""
Dataset generation script for Mosaico evaluation.

Generates Q&A pairs from the knowledge-base corpus using the Anthropic API
and saves them as a JSON file compatible with eval_batch.py.

Methodology (as described in the paper):
  - 60 pairs with claude-sonnet-4-6
  - 60 pairs with claude-haiku-4-5-20251001
  Total: 120 generated pairs → manual review → 20 validation + 50 test

Usage:
    # English corpus (CER-EN collection)
    python generate_dataset.py --lang en --output datasets/eval_dataset_EN_CER_raw.json

    # Italian corpus (CER-IT collection)
    python generate_dataset.py --lang it --output datasets/eval_dataset_IT_CER_raw.json

    # Dry run (3 pairs per model, no API calls — for testing)
    python generate_dataset.py --lang en --dry-run

Requirements:
    pip install anthropic pdfplumber python-dotenv
    ANTHROPIC_API_KEY must be set in the environment or in ../.env
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import pdfplumber
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = [
    ("claude-sonnet-4-6",           60),
    ("claude-haiku-4-5-20251001",   60),
]

CORPUS_DIRS = {
    "en": Path(__file__).parent.parent / "docs" / "Knowledge_base" / "ENG",
    "it": Path(__file__).parent.parent / "docs" / "Knowledge_base" / "ITA",
}

# Characters per context chunk sent to the model
CHUNK_CHARS = 6000
# Overlap between consecutive chunks (avoids cutting mid-paragraph)
CHUNK_OVERLAP = 500
# How many Q&A pairs to request per API call
PAIRS_PER_CALL = 5
# Seconds to wait between API calls (rate-limit buffer)
CALL_DELAY = 1.0

SYSTEM_PROMPT_EN = """You are an expert in EU cybersecurity and critical infrastructure law.
Your task is to generate high-quality question–answer pairs for evaluating a
Retrieval-Augmented Generation (RAG) system.

Rules:
1. Each question must be self-contained (answerable without external context).
2. Each answer must be strictly grounded in the provided text — no hallucination.
3. Questions must require synthesis, cross-referencing, or interpretation —
   not simple fact retrieval (avoid "what year was X published?").
4. Use precise legal and technical terminology from the source text.
5. Vary the question types: definition, obligation, comparison, procedure,
   threshold, deadline, exception, cross-reference between CER and NIS2.

Return a JSON array with exactly {n} objects, each with keys "question" and "reference".
Return only the JSON array — no markdown fences, no commentary."""

SYSTEM_PROMPT_IT = """Sei un esperto di diritto europeo in materia di sicurezza informatica e
infrastrutture critiche. Il tuo compito è generare coppie domanda–risposta di alta qualità
per valutare un sistema RAG (Retrieval-Augmented Generation).

Regole:
1. Ogni domanda deve essere autocontenuta (rispondibile senza contesto esterno).
2. Ogni risposta deve essere strettamente fondata sul testo fornito — nessuna allucinazione.
3. Le domande devono richiedere sintesi, riferimenti incrociati o interpretazione —
   non semplice recupero di fatti (evitare "in che anno è stata pubblicata X?").
4. Usa terminologia legale e tecnica precisa tratta dal testo sorgente.
5. Varia i tipi di domanda: definizione, obbligo, confronto, procedura, soglia,
   scadenza, eccezione, riferimento incrociato tra CER e NIS2.

Restituisci un array JSON con esattamente {n} oggetti, ciascuno con le chiavi
"question" e "reference".
Restituisci solo l'array JSON — nessun fence markdown, nessun commento."""


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(path: Path) -> str:
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


def extract_text_from_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_corpus(lang: str) -> list[dict]:
    """Return list of {name, text} dicts for every file in the corpus dir."""
    corpus_dir = CORPUS_DIRS[lang]
    if not corpus_dir.exists():
        sys.exit(f"Corpus directory not found: {corpus_dir}")

    docs = []
    for p in sorted(corpus_dir.iterdir()):
        if p.suffix.lower() == ".pdf":
            print(f"  Loading PDF: {p.name}")
            text = extract_text_from_pdf(p)
        elif p.suffix.lower() in (".txt", ".md"):
            print(f"  Loading TXT: {p.name}")
            text = extract_text_from_txt(p)
        else:
            continue
        if text.strip():
            docs.append({"name": p.name, "text": text})

    if not docs:
        sys.exit(f"No documents found in {corpus_dir}")
    print(f"  Loaded {len(docs)} document(s), "
          f"{sum(len(d['text']) for d in docs):,} total characters\n")
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_context_chunks(docs: list[dict]) -> list[dict]:
    """Flatten all documents into a list of {doc_name, chunk} dicts."""
    all_chunks = []
    for doc in docs:
        for chunk in chunk_text(doc["text"], CHUNK_CHARS, CHUNK_OVERLAP):
            all_chunks.append({"doc_name": doc["name"], "chunk": chunk})
    return all_chunks


# ---------------------------------------------------------------------------
# Q&A generation
# ---------------------------------------------------------------------------

def build_user_prompt(context: str, n: int, lang: str) -> str:
    intro = (
        f"Generate {n} question–answer pairs based on the following text excerpt.\n\n"
        if lang == "en"
        else f"Genera {n} coppie domanda–risposta basate sul seguente estratto testuale.\n\n"
    )
    return intro + "---\n" + context + "\n---"


def parse_pairs(raw: str) -> list[dict]:
    """Extract JSON array from model output, tolerating markdown fences."""
    raw = raw.strip()
    # Strip optional ```json ... ``` fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find a JSON array anywhere in the string
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    valid = []
    for item in data:
        if isinstance(item, dict) and "question" in item and "reference" in item:
            valid.append({
                "question":  str(item["question"]).strip(),
                "reference": str(item["reference"]).strip(),
            })
    return valid


def generate_pairs_for_model(
    client,
    model_id: str,
    target: int,
    chunks: list[dict],
    lang: str,
    dry_run: bool,
) -> list[dict]:
    system = (SYSTEM_PROMPT_EN if lang == "en" else SYSTEM_PROMPT_IT).format(n=PAIRS_PER_CALL)
    pairs: list[dict] = []
    attempts = 0
    max_attempts = target * 4  # allow retries

    # Shuffle chunks so we sample the whole corpus
    chunk_pool = chunks.copy()
    random.shuffle(chunk_pool)
    chunk_idx = 0

    while len(pairs) < target and attempts < max_attempts:
        n_needed = min(PAIRS_PER_CALL, target - len(pairs))

        if dry_run:
            # Return dummy pairs without calling the API
            for i in range(n_needed):
                pairs.append({
                    "question":  f"[DRY-RUN {model_id}] Question {len(pairs) + 1}",
                    "reference": f"[DRY-RUN] Reference answer {len(pairs) + 1}",
                    "_model": model_id,
                })
            attempts += 1
            continue

        # Pick a context chunk (cycle through pool)
        ctx = chunk_pool[chunk_idx % len(chunk_pool)]["chunk"]
        chunk_idx += 1
        attempts += 1

        user_prompt = build_user_prompt(ctx, n_needed, lang)

        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text
            new_pairs = parse_pairs(raw)
            for p in new_pairs:
                p["_model"] = model_id
            pairs.extend(new_pairs)
            print(f"    [{model_id}] {len(pairs)}/{target} pairs collected "
                  f"(+{len(new_pairs)} this call, attempt {attempts})")
        except Exception as e:
            print(f"    [{model_id}] API error (attempt {attempts}): {e}")

        time.sleep(CALL_DELAY)

    if len(pairs) < target:
        print(f"  WARNING: only {len(pairs)}/{target} pairs collected for {model_id}")
    return pairs[:target]


def deduplicate(pairs: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for p in pairs:
        key = p["question"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Mosaico evaluation dataset")
    parser.add_argument("--lang", choices=["en", "it"], required=True,
                        help="Corpus language (en or it)")
    parser.add_argument("--output", type=Path,
                        default=None,
                        help="Output JSON file (default: datasets/eval_dataset_<LANG>_raw.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate dummy pairs without calling the API (for testing)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for chunk shuffling")
    args = parser.parse_args()

    random.seed(args.seed)

    # Resolve output path
    if args.output is None:
        out_dir = Path(__file__).parent / "datasets"
        out_dir.mkdir(exist_ok=True)
        args.output = out_dir / f"eval_dataset_{'CER' if args.lang == 'en' else 'IT_CER'}_raw.json"

    # Load API key
    load_dotenv(Path(__file__).parent.parent / ".env")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        sys.exit("ANTHROPIC_API_KEY not set. Add it to .env or export it.")

    print(f"=== Dataset generation — lang={args.lang}, output={args.output} ===\n")

    # Load corpus
    print("Loading corpus...")
    docs = load_corpus(args.lang)
    chunks = build_context_chunks(docs)
    print(f"  {len(chunks)} context chunks available\n")

    # Import Anthropic (only needed at call time so dry-run works without the package)
    if not args.dry_run:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            sys.exit("anthropic package not found. Run: pip install anthropic")
    else:
        client = None
        print("  [DRY-RUN mode — no API calls]\n")

    # Generate pairs for each model
    all_pairs: list[dict] = []
    for model_id, target in MODELS:
        print(f"Generating {target} pairs with {model_id}...")
        pairs = generate_pairs_for_model(
            client, model_id, target, chunks, args.lang, args.dry_run
        )
        all_pairs.extend(pairs)
        print(f"  Done: {len(pairs)} pairs\n")

    # Deduplicate
    before = len(all_pairs)
    all_pairs = deduplicate(all_pairs)
    print(f"Deduplication: {before} → {len(all_pairs)} unique pairs\n")

    # Shuffle
    random.shuffle(all_pairs)

    # Save full output with metadata (includes _model field)
    args.output.write_text(
        json.dumps(all_pairs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(all_pairs)} pairs to {args.output}")

    # Also save a clean version (no _model field) ready for eval_batch.py
    clean_path = args.output.with_name(args.output.stem.replace("_raw", "") + "_clean.json")
    clean_pairs = [{"question": p["question"], "reference": p["reference"]} for p in all_pairs]
    clean_path.write_text(
        json.dumps(clean_pairs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved clean version (no metadata) to {clean_path}")
    print("\nNext steps:")
    print("  1. Manually review the raw file and remove pairs failing any of the 4 criteria:")
    print("     (i) self-containedness  (ii) faithfulness  (iii) non-triviality  (iv) linguistic correctness")
    print("  2. From the accepted pairs, take 20 as validation set and 100 as test set.")
    print("  3. Use the test set JSON as input to eval_batch.py.")


if __name__ == "__main__":
    main()
