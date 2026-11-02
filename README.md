# RAG Assistant

Ask questions about your own documents (resume, job notes, study material) and get answers grounded in those files. It runs completely free and offline using a local model through [Ollama](https://ollama.com).

## How it works
```
Upload (.txt .md .pdf) -> split into chunks -> embed each chunk -> store vectors
Question -> embed -> cosine-similarity search -> top matching chunks -> local LLM answers using only those chunks
```
Every answer shows the source passages it was based on, so you can check it.

## Features
- Upload and manage documents (`.txt`, `.md`, `.pdf`) through a simple web page or the REST API
- Paragraph-aware chunking with overlap for long text
- Embeddings and answers from local models (no API key, no cost)
- Own vector store: cosine similarity search with NumPy, saved to disk
- Prompt that tells the model to answer only from the retrieved context and say so when it can't find the answer
- Source citations with similarity scores
- Automated tests that run without Ollama (a fake model is used), Docker, GitHub Actions CI, Dependabot

## Tech stack
Python 3.12, FastAPI, NumPy, pypdf, httpx, Ollama (`nomic-embed-text` for embeddings, `llama3.2:3b` for answers), pytest, Docker, GitHub Actions

## Run locally
1. Install Ollama from https://ollama.com, then pull the models:
   ```
   ollama pull nomic-embed-text
   ollama pull llama3.2:3b
   ```
2. Install and start the app:
   ```
   python -m venv .venv
   .venv\Scripts\activate        # Mac/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```
3. Open http://localhost:8000 to upload a file and ask questions. API docs are at http://localhost:8000/docs.

Try it with `sample_docs/about_me_sample.md`.

## Run the tests
```
pytest
```

## Run with Docker
```
docker build -t rag-assistant .
docker run -p 8000:8000 -e OLLAMA_URL=http://host.docker.internal:11434 -v ragdata:/data rag-assistant
```

## API
| Method | Path | Description |
|---|---|---|
| POST | /api/documents | Upload and index a file |
| GET | /api/documents | List indexed documents |
| DELETE | /api/documents/{name} | Remove a document |
| POST | /api/ask | Ask a question: `{"question": "...", "top_k": 4}` |

## Configuration (environment variables)
`OLLAMA_URL`, `EMBED_MODEL`, `CHAT_MODEL`, `DATA_DIR`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`

## Possible improvements
Swap the JSON vector store for a vector database (Chroma or pgvector), add streaming answers, re-ranking, and document-level filters.
