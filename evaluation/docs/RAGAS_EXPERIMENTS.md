# Esperimenti RAGAS — piano completo per il paper

**Generator:** `gemma4:26b` &nbsp;|&nbsp; **Judge:** `qwen3:8b` (con `--no-think`)  
**Dataset EN:** `datasets/eval_dataset_EN_CER_test.json` (50 items, collection `CER-EN`)  
**Dataset IT:** `datasets/eval_dataset_IT_CER_test.json` (50 items, collection `CER-IT`)  
**Server API:** `http://localhost:9000`

---

## Stato avanzamento (aggiornato 2026-05-29)

| File risultato | Batch | RAGAS | Note |
|---|---|---|---|
| `batch_EN_baseline` | ✅ 50/50 | ✅ 50/50 | completo |
| `batch_EN_no_mmr` | ✅ 50/50 | ⏳ 10/50 | RAGAS da riprendere |
| `batch_EN_naive` | ❌ | ❌ | da fare |
| `batch_EN_no_mvs` | ❌ | ❌ | da fare |
| `batch_EN_hybrid_only` | ❌ | ❌ | da fare |
| `batch_EN_no_stitch` | ❌ | ❌ | da fare |
| `batch_EN_no_rerank` | ❌ | ❌ | da fare |
| `batch_EN_chunk_600` | ❌ | ❌ | richiede re-index |
| `batch_EN_chunk_1500` | ❌ | ❌ | richiede re-index |
| `batch_EN_chunk_no_overlap` | ❌ | ❌ | richiede re-index |
| `batch_EN_k20_l3` | ❌ | ❌ | da fare |
| `batch_EN_k150_l15` | ❌ | ❌ | da fare |
| `batch_EN_mmr_0.0` | ❌ | ❌ | da fare |
| `batch_EN_mmr_0.5` | ❌ | ❌ | da fare |
| `batch_EN_mmr_1.0` | ❌ | ❌ | da fare |
| `batch_EN_ce_k10` | ❌ | ❌ | da fare |
| `batch_EN_ce_k20` | ❌ | ❌ | da fare |
| `batch_EN_ce_k50` | ❌ | ❌ | da fare |
| `batch_IT_naive` | ❌ | ❌ | da fare |
| `batch_IT_baseline` | ❌ | ❌ | da fare |

**Prossima run:** terminare `ragas_EN_no_mmr` (resume automatico da riga 10), poi `batch_EN_naive`.

---

## Panoramica tabelle

| Tabella paper | Descrizione | Re-index? | Metriche |
|---|---|---|---|
| `tab:ablation` | Feature flag ablation (6 config) | sì | Faith, CtxPrec, CtxRec (+AnsRel, AnsCorr per baseline e Naive) |
| `tab:chunking` | Chunk size/overlap (4 config) | sì | Faith, CtxPrec, CtxRec |
| `tab:retrieval_k` | Pool size K/L (3 config) | no | Faith, CtxPrec, CtxRec, latency |
| `tab:mmr` | MMR lambda sweep (4 valori) | no | Faith, CtxPrec, CtxRec, diversity |
| `tab:cross_k` | Cross-encoder pool K' (4 valori) | no | Faith, CtxPrec, CtxRec, CE latency |
| `tab:reranking` | Reranking strategy (3 config) | no | tutte e 5 |
| `tab:italian` | Cross-lingual IT (2 config) | no | Faith, CtxPrec, CtxRec |

**Stima totale:** ~14 run × 50 Q&A × 3–5 metriche RAGAS ≈ 4–6 ore su H100.

---

## Ordine di esecuzione consigliato

```
Fase 0 — smoke test (10 min):
  1 config × 5 item  →  verifica che i punteggi siano sensati e non tutti NaN

Fase 1 — baseline (30 min):
  Full pipeline × 50 item  →  popola la riga baseline in tab:ablation e tab:reranking

Fase 2 — ablation + chunking (2–3h):
  5 ablation config + 3 chunk variant (richiedono re-index)

Fase 3 — sweep senza re-index (1–2h):
  tab:retrieval_k, tab:mmr, tab:cross_k, tab:reranking (partial), tab:italian
```

---

## Tab:ablation — Feature flag ablation

### Impostazioni `.env` per config

| Config (nome file output) | MVS | RERANK | CE | MMR | STITCH |
|---|---|---|---|---|---|
| `batch_EN_naive` | false | false | false | false | false |
| `batch_EN_baseline` | true | true | true | true | true |
| `batch_EN_no_mvs` | **false** | true | true | true | true |
| `batch_EN_hybrid_only` | true | true | **false** | true | true |
| `batch_EN_no_mmr` | true | true | true | **false** | true |
| `batch_EN_no_stitch` | true | true | true | true | **false** |

MVS = `ENABLE_MULTI_VECTOR_SEARCH`, RERANK = `ENABLE_RERANK`, CE = `ENABLE_CROSS_ENCODER_RERANK`

### Flusso per ogni config

```powershell
# 1. Modifica .env con i flag della riga, poi riavvia lo stack
docker compose down; docker compose up -d --build

# 2. Verifica che i flag siano recepiti
curl http://localhost:9000/healthz | python -m json.tool | findstr "enable_"

# 3. Raccoglie risposte (eval_batch)
python .\eval_batch.py `
  --url http://localhost:9000 `
  --dataset .\datasets\eval_dataset_EN_CER_test.json `
  --username enea --collection CER-EN `
  --output ".\results\batch_EN_<config>.csv"

# 4. Calcola metriche RAGAS (3 metriche per le righe ablation)
python .\eval_ragas.py `
  --input ".\results\batch_EN_<config>.csv" `
  --output ".\results\ragas_EN_<config>.csv" `
  --ollama-url "http://192.168.118.218:11434/api/generate" `
  --judge-model qwen3:8b --no-think `
  --metrics faithfulness,context_precision,context_recall `
  --batch-size 5 --timeout 900 --verbose --exclude-suspicious
```

Per **baseline** e **Naive RAG** aggiungere tutte e 5 le metriche:
```powershell
  --metrics faithfulness,context_precision,context_recall,answer_relevancy,answer_correctness `
  --batch-size 5 --timeout 900 --verbose --exclude-suspicious
```

---

## Tab:chunking — Chunk size/overlap

Ogni config richiede re-indicizzazione: aggiorna `CHUNK_SIZE` e `CHUNK_OVERLAP` in `.env`, riavvia, ri-carica i documenti, poi lancia.

| Config | `CHUNK_SIZE` | `CHUNK_OVERLAP` | Output |
|---|---|---|---|
| `batch_EN_chunk_600` | 600 | 120 | `batch_EN_chunk_600.csv` |
| `batch_EN_baseline` | 1000 | 200 | (già fatto in tab:ablation) |
| `batch_EN_chunk_1500` | 1500 | 300 | `batch_EN_chunk_1500.csv` |
| `batch_EN_chunk_no_overlap` | 1000 | 0 | `batch_EN_chunk_no_overlap.csv` |

```powershell
# Metriche: solo le 3 standard
python .\eval_ragas.py `
  --metrics faithfulness,context_precision,context_recall ... --exclude-suspicious
```

---

## Tab:retrieval_k — Pool size K/L

Nessun re-index. Cambia `CHAT_CANDIDATES` e `CHAT_RESULT_LIMIT` in `.env`.

| Config | `CHAT_CANDIDATES` (K) | `CHAT_RESULT_LIMIT` (L) | Output |
|---|---|---|---|
| `batch_EN_k20_l3` | 20 | 3 | `batch_EN_k20_l3.csv` |
| `batch_EN_baseline` | 100 | 10 | (già fatto) |
| `batch_EN_k150_l15` | 150 | 15 | `batch_EN_k150_l15.csv` |

La latency di retrieval viene già registrata da `eval_batch.py` nella colonna `retrieval_latency_ms`.

---

## Tab:mmr — MMR lambda sweep

Nessun re-index. Cambia `MMR_LAMBDA` in `.env` (no riavvio necessario se l'app legge live, altrimenti `make up`).

| Lambda | Output |
|---|---|
| 0.0 | `batch_EN_mmr_0.0.csv` |
| 0.5 | `batch_EN_mmr_0.5.csv` |
| 0.65 (baseline) | (già fatto) |
| 1.0 | `batch_EN_mmr_1.0.csv` |

```powershell
  --metrics faithfulness,context_precision,context_recall
```

---

## Tab:cross_k — Cross-encoder pool K'

Nessun re-index. Cambia `CROSS_ENCODER_TOP_K` in `.env`.

| K' | Output |
|---|---|
| 10 | `batch_EN_ce_k10.csv` |
| 20 | `batch_EN_ce_k20.csv` |
| 30 (baseline) | (già fatto) |
| 50 | `batch_EN_ce_k50.csv` |

La latency del cross-encoder viene già registrata in `ce_latency_ms` da `eval_batch.py`.

---

## Tab:reranking — Reranking strategy

Nessun re-index. Tre config:

| Strategia | `ENABLE_RERANK` | `ENABLE_CROSS_ENCODER_RERANK` | Output |
|---|---|---|---|
| No reranking | false | false | `batch_EN_no_rerank.csv` |
| Hybrid only | true | false | `batch_EN_hybrid_only.csv` (già fatto in tab:ablation) |
| Cascade (baseline) | true | true | `batch_EN_baseline.csv` (già fatto) |

Tutte e 5 le metriche:
```powershell
  --metrics faithfulness,context_precision,context_recall,answer_relevancy,answer_correctness
```

---

## Tab:italian — Cross-lingual

Usa la collection `CER-IT`. Nessuna modifica al `.env` rispetto al baseline.

```powershell
# Naive RAG (disabilita tutti i flag)
python .\eval_batch.py `
  --url http://localhost:9000 `
  --dataset .\datasets\eval_dataset_IT_CER_test.json `
  --username enea --collection CER-IT `
  --output ".\results\batch_IT_naive.csv"

# Full pipeline (ripristina tutti i flag a true)
python .\eval_batch.py `
  --url http://localhost:9000 `
  --dataset .\datasets\eval_dataset_IT_CER_test.json `
  --username enea --collection CER-IT `
  --output ".\results\batch_IT_baseline.csv"
```

```powershell
python .\eval_ragas.py `
  --input ".\results\batch_IT_<config>.csv" `
  --output ".\results\ragas_IT_<config>.csv" `
  --ollama-url "http://192.168.118.218:11434/api/generate" `
  --judge-model qwen3:8b --no-think `
  --metrics faithfulness,context_precision,context_recall `
  --batch-size 5 --timeout 900 --verbose --exclude-suspicious
```

---

## Variabili `.env` di riferimento (baseline)

```dotenv
OLLAMA_MODEL=gemma4:26b
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
QDRANT_SCORE_THRESHOLD=0.20
CHAT_CANDIDATES=100
CHAT_RESULT_LIMIT=10
CHAT_CONTEXT_CHAR_BUDGET=10000
CHAT_HISTORY_CHAR_BUDGET=0
MMR_LAMBDA=0.65
CROSS_ENCODER_TOP_K=30
ENABLE_RERANK=true
ENABLE_MMR=true
ENABLE_STITCH=true
ENABLE_MULTI_VECTOR_SEARCH=true
ENABLE_CROSS_ENCODER_RERANK=true
```
