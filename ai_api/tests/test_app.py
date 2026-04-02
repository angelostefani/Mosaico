import os
import json
import tempfile
import asyncio
import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook

os.environ.setdefault("SKIP_AUTH", "true")
os.environ.setdefault("ALLOW_EMBEDDING_FALLBACK", "true")
os.environ.setdefault("PYTEST_CURRENT_TEST", "bootstrap")
os.environ.setdefault("DB_ENGINE", "sqlite")
os.environ.setdefault("UPLOAD_DB_PATH", os.path.join(os.path.dirname(__file__), "test_uploads.sqlite3"))

import app

client = TestClient(app.app)


@pytest.fixture(autouse=True)
def clean_conversations():
    # Pulisce le tabelle delle conversazioni prima e dopo ogni test
    with app.get_db_session() as session:
        session.query(app.ConversationMessage).delete()
        session.query(app.Conversation).delete()
    yield
    with app.get_db_session() as session:
        session.query(app.ConversationMessage).delete()
        session.query(app.Conversation).delete()


def test_chunk_text_simple():
    text = "abcdefghij"
    chunks = app.chunk_text(text, chunk_size=3, chunk_overlap=0)
    assert chunks == ["abc", "def", "ghi", "j"]


def test_extract_text_txt(tmp_path):
    # Crea un file .txt e verifica l'estrazione
    p = tmp_path / "test.txt"
    p.write_text("hello world", encoding="utf-8")
    result = app.extract_text(str(p))
    assert result == "hello world"


def test_extract_text_excel(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Foglio1"
    ws.append(["name", "age"])
    ws.append(["Alice", 30])
    ws.append(["Bob", None])
    file_path = tmp_path / "test.xlsx"
    wb.save(file_path)

    text, meta = app.extract_text(str(file_path), return_metadata=True)
    assert "Alice" in text
    assert "age=30" in text
    assert meta["source_ext"] == ".xlsx"
    assert meta["pages"]
    assert any(page.get("sheet_name") == "Foglio1" for page in meta["pages"])


def test_extract_text_unsupported(tmp_path):
    # Estrazione da formato non supportato solleva HTTPException 400
    p = tmp_path / "test.xyz"
    p.write_text("data")
    with pytest.raises(app.HTTPException) as excinfo:
        app.extract_text(str(p))
    assert excinfo.value.status_code == 400


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "I-NEST_API is running!"}


test_process_called = False

@pytest.fixture(autouse=True)
def override_process(monkeypatch):
    # Override di process_document per non fare chiamate esterne
    def dummy_process(path, username, collection, upload_id=None, original_filename=None):
        global test_process_called
        test_process_called = True
        return {
            "num_chunks": 1,
            "num_points_inserted": 1,
            "collection_used": "test",
            "upload_id": upload_id or "dummy-id",
            "stored_filename": os.path.basename(path),
            "original_filename": original_filename,
        }
    monkeypatch.setattr(app, "process_document", dummy_process)
    yield


def test_upload_endpoint(tmp_path):
    # Crea un file di testo e invia POST /upload
    global test_process_called
    test_process_called = False
    p = tmp_path / "file.txt"
    p.write_text("hello", encoding="utf-8")
    with open(p, "rb") as f:
        response = client.post(
            "/upload",
            files={"file": ("file.txt", f, "text/plain")}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "File caricato e processato con successo"
    assert data["processing_info"]["collection_used"] == "test"
    assert "upload_id" in data
    assert data["processing_info"]["upload_id"] == data["upload_id"]
    assert data["processing_info"]["original_filename"] == "file.txt"
    assert data["processing_info"]["stored_filename"].endswith(".txt")
    assert test_process_called


def test_upload_endpoint_excel(tmp_path):
    global test_process_called
    test_process_called = False

    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["col1", "col2"])
    ws.append(["val1", "val2"])
    excel_path = tmp_path / "file.xlsx"
    wb.save(excel_path)

    with open(excel_path, "rb") as f:
        response = client.post(
            "/upload",
            files={"file": ("file.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "File caricato e processato con successo"
    assert data["processing_info"]["stored_filename"].endswith(".xlsx")
    assert test_process_called

@ pytest.fixture(autouse=False)
def dummy_ask(monkeypatch):
    # Override di ask per restituire risposta fissa
    async def fake_ask(prompt, context, conversation_history=None, model_override=None, collection_scope=None):
        return "dummy answer"
    monkeypatch.setattr(app, "ask", fake_ask)
    yield

@pytest.fixture(autouse=False)
def override_search(monkeypatch):
    # Override di init_qdrant_client per restituire client dummy
    class DummyClient:
        def collection_exists(self, coll):
            return True
        def search(self, collection_name, query_vector, limit, with_payload):
            class R:
                payload = {"text_chunk": "chunk"}
            return [R()]
    monkeypatch.setattr(app, "init_qdrant_client", lambda: DummyClient())
    yield


def test_chat_endpoint(dummy_ask, override_search):
    response = client.post(
        "/chat",
        data={"question": "Test?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "dummy answer"
    assert data["conversation_id"]


def test_chat_endpoint_with_history(monkeypatch, override_search):
    captured = {}

    async def fake_ask(prompt, context, conversation_history=None, model_override=None, collection_scope=None):
        captured["history"] = conversation_history
        return "with history"

    monkeypatch.setattr(app, "ask", fake_ask)

    history_payload = json.dumps([
        {"role": "user", "content": "ciao"},
        {"role": "assistant", "content": "salve"}
    ])

    response = client.post(
        "/chat",
        data={"question": "Test?", "conversation_history": history_payload}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "with history"
    assert data["conversation_id"]
    assert captured["history"] is not None
    assert "user: ciao" in captured["history"]
    assert "assistant: salve" in captured["history"]


def test_chat_endpoint_with_model_override(monkeypatch, override_search):
    captured = {}

    async def fake_ask(prompt, context, conversation_history=None, model_override=None, collection_scope=None):
        captured["model_override"] = model_override
        return "with model"

    monkeypatch.setattr(app, "ask", fake_ask)

    response = client.post(
        "/chat",
        data={"question": "Test?", "model": "my-model"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "with model"
    assert data["conversation_id"]
    assert captured["model_override"] == "my-model"


def test_chat_continues_conversation(monkeypatch, override_search):
    histories = []

    async def fake_ask(prompt, context, conversation_history=None, model_override=None, collection_scope=None):
        histories.append(conversation_history)
        return f"reply-{len(histories)}"

    monkeypatch.setattr(app, "ask", fake_ask)

    first = client.post("/chat", data={"question": "Hello?"})
    assert first.status_code == 200
    conv_id = first.json()["conversation_id"]

    second = client.post("/chat", data={"question": "Second?", "conversation_id": conv_id})
    assert second.status_code == 200
    assert len(histories) == 2
    assert histories[0] is None or histories[0] == ""
    assert "reply-1" in histories[1]

    detail = client.get(f"/conversations/{conv_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["message_count"] == 4  # user+assistant per ogni chiamata
    assert body["messages"][0]["content"] == "Hello?"
    assert body["messages"][1]["content"] == "reply-1"
    assert body["messages"][2]["content"] == "Second?"
    assert body["messages"][3]["content"] == "reply-2"


def test_recent_conversations_endpoint(dummy_ask, override_search):
    created_ids = []
    for i in range(3):
        resp = client.post("/chat", data={"question": f"Q{i}?"})
        assert resp.status_code == 200
        created_ids.append(resp.json()["conversation_id"])

    listing = client.get("/conversations")
    assert listing.status_code == 200
    data = listing.json()
    assert data["count"] >= 3
    returned_ids = [c["conversation_id"] for c in data["conversations"]]
    for cid in created_ids:
        assert cid in returned_ids


def test_ask_function(monkeypatch):
    captured = {"models": []}

    class DummyStreamResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            for line in ['{"response":"hello"}', ""]:
                yield line

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            captured["prompt"] = kwargs["json"]["prompt"]
            captured["models"].append(kwargs["json"]["model"])
            return DummyStreamResponse()

    def dummy_post(*args, **kwargs):  # pragma: no cover - compat guard
        captured["prompt"] = kwargs["json"]["prompt"]
        captured["models"].append(kwargs["json"]["model"])
        raise AssertionError("requests.post non dovrebbe essere usato da ask()")

    monkeypatch.setattr(app.requests, "post", dummy_post)
    monkeypatch.setattr(app.httpx, "AsyncClient", DummyAsyncClient)

    answer = asyncio.run(app.ask("prompt", "context", "user: hello"))
    assert answer == "hello"
    assert "user: hello" in captured["prompt"]
    assert captured["models"][-1] == app.OLLAMA_MODEL

    answer_override = asyncio.run(app.ask("prompt", "context", model_override="custom-model"))
    assert answer_override == "hello"
    assert captured["models"][-1] == "custom-model"

    answer_scope = asyncio.run(app.ask("prompt", "context", collection_scope="Sei un esperto in cinema e film"))
    assert answer_scope == "hello"
    assert "Ambito di pertinenza della collection" in captured["prompt"]
    assert "Sei un esperto in cinema e film" in captured["prompt"]


def test_collection_config_endpoints():
    # create / update
    upsert_resp = client.put(
        "/collection/config",
        json={"username": "user1", "collection": "docs", "scope_prompt": "Sei un esperto in Infrastrutture Critiche"},
    )
    assert upsert_resp.status_code == 200
    upsert_data = upsert_resp.json()
    assert upsert_data["collection"] == "user1_docs"
    assert upsert_data["config"]["scope_prompt"] == "Sei un esperto in Infrastrutture Critiche"

    # read
    read_resp = client.get("/collection/config", params={"username": "user1", "collection": "docs"})
    assert read_resp.status_code == 200
    read_data = read_resp.json()
    assert read_data["collection"] == "user1_docs"
    assert read_data["config"]["scope_prompt"] == "Sei un esperto in Infrastrutture Critiche"

    # delete
    del_resp = client.delete("/collection/config", params={"username": "user1", "collection": "docs"})
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # read after delete
    read_after_del = client.get("/collection/config", params={"username": "user1", "collection": "docs"})
    assert read_after_del.status_code == 200
    assert read_after_del.json()["config"] is None


def test_chat_passes_collection_scope(monkeypatch, override_search):
    captured = {}

    async def fake_ask(prompt, context, conversation_history=None, model_override=None, collection_scope=None):
        captured["collection_scope"] = collection_scope
        return "with scope"

    monkeypatch.setattr(app, "ask", fake_ask)

    put_resp = client.put(
        "/collection/config",
        json={"username": "scopeuser", "collection": "knowledge", "scope_prompt": "Sei un esperto in cinema e film"},
    )
    assert put_resp.status_code == 200

    response = client.post(
        "/chat",
        data={"question": "Test?", "username": "scopeuser", "collection": "knowledge"},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "with scope"
    assert captured["collection_scope"] == "Sei un esperto in cinema e film"


def test_healthz_endpoint(monkeypatch, tmp_path):
    # Mock QdrantClient.get_collections
    class DummyQdrant:
        def __init__(self, *args, **kwargs):
            pass
        def get_collections(self):
            return {"collections": []}

    monkeypatch.setattr(app, "QdrantClient", DummyQdrant)

    # Mock requests.get for Ollama tags
    class DummyResp:
        status_code = 200
    monkeypatch.setattr(app.requests, "get", lambda *args, **kwargs: DummyResp())

    # Ensure upload dir is writable (override to tmp_path to be safe)
    monkeypatch.setattr(app, "UPLOAD_DIR", str(tmp_path))

    # Short-circuit DB healthcheck to avoid reliance on real DB
    monkeypatch.setattr(
        app,
        "_db_healthcheck",
        lambda: {
            "ok": True,
            "error": None,
            "latency_ms": 1,
            "engine": "sqlite",
            "database": "test",
            "host": None,
            "port": None,
        },
    )
    monkeypatch.setattr(
        app,
        "get_embedding_status",
        lambda: {
            "ok": True,
            "mode": "ready",
            "fallback_enabled": False,
            "detail": None,
            "model": "test-model",
        },
    )

    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["qdrant"]["ok"] is True
    assert data["ollama"]["ok"] is True
    assert data["storage"]["writable"] is True
    assert data["database"]["ok"] is True
    assert data["embedding"]["ok"] is True


def test_list_ollama_models(monkeypatch):
    class DummyResp:
        status_code = 200
        def json(self):
            return {
                "models": [
                    {
                        "name": "gemma:latest",
                        "size": 123,
                        "modified_at": "2024-01-01T00:00:00Z",
                        "digest": "abc",
                        "details": {"parameter_size": "1B", "quantization_level": "Q4_0"},
                    }
                ]
            }

    captured = {}

    def fake_get(url, *args, **kwargs):
        captured["url"] = url
        return DummyResp()

    monkeypatch.setattr(app.requests, "get", fake_get)

    resp = client.get("/ollama/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["models"][0]["name"] == "gemma:latest"
    assert captured["url"].endswith("/api/tags")


def test_collections_endpoint(monkeypatch):
    # Mock QdrantClient.get_collections for /collections
    class DummyQdrant:
        def __init__(self, *args, **kwargs):
            pass
        def get_collections(self):
            return {"collections": [{"name": "a"}, {"name": "b"}]}

    monkeypatch.setattr(app, "QdrantClient", DummyQdrant)

    resp = client.get("/collections")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert data["collections"] == ["a", "b"]


def test_collections_endpoint_with_username(monkeypatch):
    class DummyQdrant:
        def __init__(self, *args, **kwargs):
            pass
        def get_collections(self):
            return {
                "collections": [
                    {"name": "user1_docs"},
                    {"name": "user1"},
                    {"name": "user2"},
                    {"name": "user1_reports"},
                    {"name": "shared"},
                ]
            }

    monkeypatch.setattr(app, "QdrantClient", DummyQdrant)

    resp = client.get("/collections", params={"username": "user1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 3
    assert data["collections"] == ["user1_docs", "user1", "user1_reports"]


def test_stitch_chunks_preserves_best_rank_under_budget():
    class Result:
        def __init__(self, text, score, source, page, chunk):
            self.score = score
            self.payload = {
                "text_chunk": text,
                "source_file": source,
                "page_number": page,
                "chunk_index": chunk,
            }

    results = [
        Result("Top result block.", 0.99, "zeta.pdf", 1, 0),
        Result("Lower ranked first chunk.", 0.20, "alpha.pdf", 1, 0),
        Result("Lower ranked second chunk.", 0.19, "alpha.pdf", 1, 1),
    ]

    stitched = app.stitch_chunks(results, char_budget=40, max_run_len=3)
    assert stitched
    assert "Top result block." in stitched[0][0]


def test_dedup_results_keeps_similar_but_distinct_chunks():
    class Result:
        def __init__(self, text):
            self.payload = {"text_chunk": text}
            self.score = 0.9

    rows = [
        Result("Articolo 12 - limite 10 MW per il sito A."),
        Result("Articolo 12 - limite 12 MW per il sito A."),
    ]

    deduped = app._dedup_results(rows, sim_threshold=0.97)
    assert len(deduped) == 2


def test_chat_returns_503_when_embedding_model_unavailable(monkeypatch):
    def fail_embedding():
        raise app.HTTPException(status_code=503, detail="embedding down")

    monkeypatch.setattr(app, "get_embedding_model", fail_embedding)

    response = client.post("/chat", data={"question": "Test?"})
    assert response.status_code == 503
    assert response.json()["detail"] == "embedding down"


def test_healthz_reports_embedding_degraded(monkeypatch, tmp_path):
    class DummyQdrant:
        def __init__(self, *args, **kwargs):
            pass

        def get_collections(self):
            return {"collections": []}

    class DummyResp:
        status_code = 200

    monkeypatch.setattr(app, "QdrantClient", DummyQdrant)
    monkeypatch.setattr(app.requests, "get", lambda *args, **kwargs: DummyResp())
    monkeypatch.setattr(app, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(
        app,
        "_db_healthcheck",
        lambda: {
            "ok": True,
            "error": None,
            "latency_ms": 1,
            "engine": "sqlite",
            "database": "test",
            "host": None,
            "port": None,
        },
    )
    monkeypatch.setattr(
        app,
        "get_embedding_status",
        lambda: {
            "ok": False,
            "mode": "fallback",
            "fallback_enabled": True,
            "detail": "Fallback deterministico attivo.",
            "model": "test-model",
        },
    )

    resp = client.get("/healthz")
    assert resp.status_code == 503
    assert resp.json()["status"] == "degraded"
    assert resp.json()["embedding"]["mode"] == "fallback"


def test_chat_executes_rerank_even_when_mmr_disabled(monkeypatch, override_search):
    captured = {"rerank": 0, "mmr": 0}

    async def fake_ask(prompt, context, conversation_history=None, model_override=None, collection_scope=None):
        return "ok"

    def fake_rerank(question, results):
        captured["rerank"] += 1
        return [(0.9, results[0])]

    def fake_apply_mmr(scored_results, top_k):
        captured["mmr"] += 1
        return [r for _score, r in scored_results[:top_k]]

    monkeypatch.setattr(app, "ask", fake_ask)
    monkeypatch.setattr(app, "rerank_results", fake_rerank)
    monkeypatch.setattr(app, "apply_mmr", fake_apply_mmr)
    monkeypatch.setattr(app, "ENABLE_RERANK", True)
    monkeypatch.setattr(app, "ENABLE_MMR", False)

    response = client.post("/chat", data={"question": "Test?"})
    assert response.status_code == 200
    assert captured["rerank"] == 1
    assert captured["mmr"] == 0
