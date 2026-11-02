from app.rag import EMPTY_MESSAGE, RagService
from app.store import VectorStore
from tests.fakes import FakeLLM


def make_service():
    llm = FakeLLM()
    return RagService(VectorStore(), llm, chunk_size=400, overlap=50, top_k=2), llm


def test_ask_on_empty_store_does_not_call_the_model():
    service, llm = make_service()
    result = service.ask("What is in my documents?")
    assert result == {"answer": EMPTY_MESSAGE, "sources": []}
    assert llm.chat_calls == 0


def test_retrieval_finds_the_relevant_document():
    service, _ = make_service()
    service.ingest("cooking.md", "To make pasta, boil water, add salt, and cook the spaghetti for ten minutes.")
    service.ingest("coding.md", "FastAPI is a Python framework for building REST APIs with automatic validation.")
    result = service.ask("How do I cook spaghetti pasta?", top_k=1)
    assert result["answer"] == "FAKE ANSWER"
    assert result["sources"][0]["document"] == "cooking.md"


def test_ingest_rejects_empty_text():
    service, _ = make_service()
    try:
        service.ingest("empty.txt", "   ")
        assert False, "expected ValueError"
    except ValueError:
        pass
