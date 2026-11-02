from fastapi.testclient import TestClient

from app.main import app, get_service
from app.rag import RagService
from app.store import VectorStore
from tests.fakes import FakeLLM


def make_client():
    service = RagService(VectorStore(), FakeLLM(), chunk_size=400, overlap=50, top_k=2)
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def test_health():
    assert make_client().get("/health").json() == {"status": "ok"}


def test_upload_list_ask_delete_flow():
    client = make_client()
    files = {"file": ("notes.md", b"I built a Spring Boot API with JWT login and a FastAPI evaluation platform.", "text/markdown")}
    upload = client.post("/api/documents", files=files)
    assert upload.status_code == 201
    assert upload.json() == {"document": "notes.md", "chunks": 1}

    assert client.get("/api/documents").json() == [{"name": "notes.md", "chunks": 1}]

    answer = client.post("/api/ask", json={"question": "What APIs did I build?"})
    assert answer.status_code == 200
    assert answer.json()["answer"] == "FAKE ANSWER"
    assert answer.json()["sources"][0]["document"] == "notes.md"

    assert client.delete("/api/documents/notes.md").status_code == 204
    assert client.delete("/api/documents/notes.md").status_code == 404


def test_unsupported_file_type_is_rejected():
    client = make_client()
    response = client.post("/api/documents", files={"file": ("photo.png", b"data", "image/png")})
    assert response.status_code == 400


def test_short_question_is_rejected():
    client = make_client()
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 422
