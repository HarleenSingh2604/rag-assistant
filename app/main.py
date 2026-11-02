"""FastAPI app exposing the RAG assistant."""
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import config
from app.llm import LLMError, OllamaClient
from app.loaders import extract_text
from app.rag import RagService
from app.store import VectorStore

app = FastAPI(title="RAG Assistant", description="Ask questions about your own documents using a local LLM.")

_service: RagService | None = None
INDEX_HTML = Path(__file__).resolve().parent.parent / "static" / "index.html"


def get_service() -> RagService:
    global _service
    if _service is None:
        _service = RagService(
            store=VectorStore(config.DATA_DIR / "store.json"),
            llm=OllamaClient(config.OLLAMA_URL, config.EMBED_MODEL, config.CHAT_MODEL),
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
            top_k=config.TOP_K,
        )
    return _service


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/documents", status_code=201)
def upload_document(file: UploadFile = File(...), service: RagService = Depends(get_service)):
    data = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is too large")
    name = Path(file.filename or "document").name
    try:
        text = extract_text(name, data)
        chunks = service.ingest(name, text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"document": name, "chunks": chunks}


@app.get("/api/documents")
def list_documents(service: RagService = Depends(get_service)):
    return service.list_documents()


@app.delete("/api/documents/{name}", status_code=204)
def delete_document(name: str, service: RagService = Depends(get_service)):
    if not service.delete(name):
        raise HTTPException(status_code=404, detail="Document not found")


@app.post("/api/ask")
def ask(body: AskRequest, service: RagService = Depends(get_service)):
    try:
        return service.ask(body.question, body.top_k)
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
