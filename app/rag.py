"""The RAG pipeline: ingest documents, retrieve relevant chunks, ask the model using only those chunks."""
from app.chunking import split_text
from app.store import VectorStore

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "If the answer is not in the context, say \"I couldn't find that in the uploaded documents.\" "
    "Do not make things up. Cite the passages you used like [1] or [2]. Keep answers concise."
)

EMPTY_MESSAGE = "No documents have been uploaded yet. Upload a .txt, .md or .pdf file first."


class RagService:
    def __init__(self, store: VectorStore, llm, chunk_size: int = 800, overlap: int = 150, top_k: int = 4):
        self.store = store
        self.llm = llm
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k

    def ingest(self, name: str, text: str) -> int:
        chunks = split_text(text, self.chunk_size, self.overlap)
        if not chunks:
            raise ValueError("No readable text found in this file")
        vectors = self.llm.embed(chunks)
        self.store.add_document(name, chunks, vectors)
        return len(chunks)

    def list_documents(self) -> list[dict]:
        return self.store.documents()

    def delete(self, name: str) -> bool:
        return self.store.delete_document(name)

    def ask(self, question: str, top_k: int | None = None) -> dict:
        if self.store.is_empty():
            return {"answer": EMPTY_MESSAGE, "sources": []}
        k = top_k or self.top_k
        query_vector = self.llm.embed([question])[0]
        hits = self.store.search(query_vector, k)
        context = "\n\n".join(f"[{i}] (from {h['document']})\n{h['text']}" for i, h in enumerate(hits, start=1))
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        answer = self.llm.chat(SYSTEM_PROMPT, user_prompt)
        return {"answer": answer, "sources": hits}
