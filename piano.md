# Piano: Miglioramenti qualità RAG pipeline

## Context

Analisi esterna (Codex/GPT) ha identificato 7 difetti nella pipeline RAG di Mosaico. Il codice è stato verificato e i finding sono confermati. Questo piano li prioritizza e descrive la fix esatta per ciascuno, partendo dai più impattanti.

---

## Finding verificati e piano di fix

### [HIGH-1] Stitching distrugge il ranking di rilevanza
**File:** `ai_api/app.py:1743`

**Problema:** `stitch_chunks()` riordina i chunk per `(source_file, page_number, chunk_index)` *prima* di consumare il budget. Il vettore di rilevanza (score) viene ignorato, e chunk con score basso ma nome file alfabeticamente anteriore finiscono nel prompt al posto di chunk più rilevanti.

**Fix:** Separare l'ordinamento dal budget-consumption. Il sort lessicografico serve solo per *trovare* run consecutive — dopo la stitching, le run risultanti vanno **riordinate per max score del run** prima di consumare il budget.

```python
# Dopo stitching, ogni run porta con sé il max score dei suoi chunk
stitched_with_score = [(text, meta, max_score_of_run), ...]
stitched_with_score.sort(key=lambda x: x[2], reverse=True)
# poi consuma budget in questo ordine
```

Il sort interno (per trovare contigui) rimane, ma l'output finale deve preservare il ranking di rilevanza.

---

### [HIGH-2] Fallback embedding silenzioso e arbitrario
**File:** `ai_api/app.py:193`

**Problema:** Se `SentenceTransformer` non carica, si usa un hash deterministico a 32 dimensioni senza relazione semantica. Il sistema continua a funzionare apparentemente. Recall e ranking diventano arbitrari.

**Fix:** Al posto del fallback silenzioso, **sollevare un errore esplicito** all'avvio che blocchi il servizio (o almeno che restituisca HTTP 503 su ogni richiesta di chat/upload finché il modello non è disponibile). Il fallback può rimanere come ultima risorsa per casi di test, ma deve essere **opt-in esplicito** via env var (`ALLOW_EMBEDDING_FALLBACK=true`).

Aggiornare `/healthz` per riportare `embedding_model: "fallback"` come stato degradato (non healthy).

---

### [MEDIUM-1] ENABLE_RERANK e ENABLE_MMR sono accoppiati in modo errato
**File:** `ai_api/app.py:2027`, `ai_api/app.py:2289`

**Problema:** `ENABLE_RERANK=true + ENABLE_MMR=false` → rerank non viene eseguito affatto. La logica attuale è:
```python
reranked = rerank_results(...) if ENABLE_MMR else pool
```
Quindi MMR *governa* anche il rerank, contro la documentazione.

**Fix:** Separare le due feature:
```python
if ENABLE_RERANK:
    reranked = rerank_results(question, pool, top_k=min(pool_k, len(pool)))
else:
    reranked = pool

if ENABLE_MMR:
    reranked = apply_mmr(reranked, ...)
```
Aggiornare `ai_api/docs/documentazione_progetto.md` per riflettere il comportamento corretto.

---

### [MEDIUM-2] Query expansion naive (solo token grezzo)
**File:** `ai_api/app.py:1630`

**Problema:** `_select_query_expansions()` prende semplicemente i primi N token ≥ 3 caratteri senza pesatura né filtro stopword. Su domande tecniche/verbose produce expansion rumorose.

**Fix minima (senza NLP pesante):** Filtrare le stopword italiane/inglesi più comuni da una lista statica embedded (~100 parole), e preferire token che appaiono meno frequentemente nella domanda (proxy di informatività). Non richiedere dipendenze esterne.

```python
STOPWORDS_IT_EN = {"che", "del", "della", "come", "when", "what", "the", "and", ...}
tokens = [t for t in _tokenize(question) if t not in STOPWORDS_IT_EN]
```

---

### [MEDIUM-3] Deduplication O(n²) e collasso di chunk simili ma distinti
**File:** `ai_api/app.py:1861`, `ai_api/app.py:2173`

**Problema:** `SequenceMatcher` a 95% su tutti i chunk è O(n²) e può collassare chunk distinti in documenti ripetitivi (regolamenti, template).

**Fix:** Usare un hash veloce (es. primi+ultimi 64 caratteri del chunk normalizzato) per dedup esatta *prima* di SequenceMatcher, e alzare leggermente la soglia a 0.97 per ridurre i falsi positivi su documenti strutturati. Le due versioni (`_dedup` e `_dedup_stream`) vanno unificate in una sola funzione.

---

### [LOW] Chunking solo character-based
**File:** `ai_api/app.py:751`

Il chunking è semplice ma ragionevole per ora. Il boundary-detection su `.!?\n` è già presente. **Non è prioritario** — rimandare a una fase successiva quando si valuterà il supporto strutturato (heading-aware, table-aware).

---

### [LOW] Grounding senza verifica strutturata
**File:** `ai_api/app.py:1263`

Il prompt è già ben costruito. Citazioni strutturate e verifica di coverage richiedono un refactor significativo del loop LLM. **Non prioritario** ora.

---

### [LOW] Copertura test RAG
**File:** `ai_api/tests/test_app.py`

Aggiungere almeno: test di `stitch_chunks()` con score preservation, test del fallback embedding che verifica HTTP 503, test del comportamento ENABLE_RERANK/ENABLE_MMR separati.

---

## File critici da modificare

| File | Sezioni coinvolte |
|------|-------------------|
| `ai_api/app.py` | L.175-207 (fallback), L.1630-1646 (expansion), L.1732-1770 (stitch), L.1861/2173 (dedup), L.2027/2289 (rerank+mmr) |
| `ai_api/tests/test_app.py` | Nuovi test per stitch, fallback, rerank |
| `ai_api/docs/documentazione_progetto.md` | Sezioni ENABLE_RERANK, ENABLE_MMR, fallback embedding |

## Ordine di implementazione consigliato

1. **[HIGH-1] Fix stitching** — massimo impatto sulla qualità, cambiamento localizzato
2. **[HIGH-2] Fix fallback embedding** — sicurezza del sistema, blocca degradi silenziosi
3. **[MEDIUM-1] Fix rerank/MMR** — corregge comportamento contro-documentato
4. **[MEDIUM-2] Stopword expansion** — miglioramento incrementale, basso rischio
5. **[MEDIUM-3] Dedup unificata** — cleanup + piccolo fix soglia
6. **[LOW] Test aggiuntivi** — copertura per i fix precedenti

## Verifica

- `pytest -q ai_api/tests/test_app.py` deve passare dopo ogni modifica
- Test manuale: caricare un documento tecnico ripetitivo (es. regolamento), fare query con termini specifici, verificare che i chunk nel prompt siano ordinati per rilevanza
- Verificare `/healthz` che riporti stato degradato quando embedding fallisce
- Testare `ENABLE_RERANK=true ENABLE_MMR=false` e verificare che il rerank venga eseguito
