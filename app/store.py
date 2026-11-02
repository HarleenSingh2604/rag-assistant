"""A small vector store: embeddings kept in memory, searched with cosine similarity, saved to JSON."""
import json
import os
import threading
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, path: Path | str | None = None):
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._items: list[dict] = []
        if self._path and self._path.exists():
            self._items = json.loads(self._path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._items), encoding="utf-8")
        os.replace(tmp, self._path)

    def add_document(self, name: str, chunks: list[str], vectors: list[list[float]]) -> None:
        """Store a document's chunks. Uploading the same name again replaces the old version."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        with self._lock:
            self._items = [i for i in self._items if i["document"] != name]
            for idx, (text, vec) in enumerate(zip(chunks, vectors)):
                self._items.append({"document": name, "chunk_index": idx, "text": text, "vector": list(vec)})
            self._save()

    def delete_document(self, name: str) -> bool:
        with self._lock:
            before = len(self._items)
            self._items = [i for i in self._items if i["document"] != name]
            changed = len(self._items) != before
            if changed:
                self._save()
            return changed

    def documents(self) -> list[dict]:
        counts: dict[str, int] = {}
        with self._lock:
            for item in self._items:
                counts[item["document"]] = counts.get(item["document"], 0) + 1
        return [{"name": n, "chunks": c} for n, c in sorted(counts.items())]

    def is_empty(self) -> bool:
        return not self._items

    def search(self, query_vector: list[float], k: int = 4) -> list[dict]:
        """Return the k chunks most similar to the query (cosine similarity, highest first)."""
        with self._lock:
            items = list(self._items)
        if not items:
            return []
        matrix = np.array([i["vector"] for i in items], dtype=float)
        query = np.array(query_vector, dtype=float)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        norms[norms == 0] = 1e-12
        scores = matrix @ query / norms
        top = np.argsort(-scores)[:k]
        return [
            {
                "document": items[i]["document"],
                "chunk_index": items[i]["chunk_index"],
                "text": items[i]["text"],
                "score": round(float(scores[i]), 4),
            }
            for i in top
        ]
