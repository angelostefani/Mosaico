import os
import json
import tempfile
import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook

import app

client = TestClient(app.app)


@pytest.fixture(autouse=True)
def clean_conversations():
    # Pulisce le tabelle delle conversazioni prima e dopo ogni test
    with app.get_upload_db_conn() as conn:
        conn.execute("DELETE FROM conversation_messages")
        conn.execute("DELETE FROM conversations")
        conn.commit()
    yield
    with app.get_upload_db_conn() as conn:
        conn.execute("DELETE FROM conversation_messages")
        conn.execute("DELETE FROM conversations")
        conn.commit()


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
    def fake_ask(prompt, context, conversation_history=None, model_override=None):
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

    def fake_ask(prompt, context, conversation_history=None, model_override=None):
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

    def fake_ask(prompt, context, conversation_history=None, model_override=None):
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

    def fake_ask(prompt, context, conversation_history=None, model_override=None):
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
    # Simula la risposta streaming di requests.post
    class DummyResponse:
        status_code = 200
        def __init__(self):
            self._lines = [b'{"response":"hello"}', b'']
        def iter_lines(self):
            return iter(self._lines)
        def raise_for_status(self):
            return None
        def close(self):
            pass

    captured = {"models": []}

    def dummy_post(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        captured["models"].append(kwargs["json"]["model"])
        return DummyResponse()

    monkeypatch.setattr(app.requests, "post", dummy_post)

    answer = app.ask("prompt", "context", "user: hello")
    assert answer == "hello"
    assert "user: hello" in captured["prompt"]
    assert captured["models"][-1] == app.OLLAMA_MODEL

    answer_override = app.ask("prompt", "context", model_override="custom-model")
    assert answer_override == "hello"
    assert captured["models"][-1] == "custom-model"


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

    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["qdrant"]["ok"] is True
    assert data["ollama"]["ok"] is True
    assert data["storage"]["writable"] is True
    assert data["database"]["ok"] is True


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
