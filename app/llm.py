"""Client for a local Ollama server (free, runs on your own machine)."""
import httpx


class LLMError(Exception):
    """Raised when the language model server cannot be reached or returns an error."""


class OllamaClient:
    def __init__(self, base_url: str, embed_model: str, chat_model: str, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        try:
            response = httpx.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
                        raise LLMError(
                f"Could not get a response from Ollama at {self.base_url} "
                f"({type(exc).__name__}: {exc}). Is it running, and have you pulled the models?"
            ) from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post("/api/embed", {"model": self.embed_model, "input": texts})
        return data["embeddings"]

    def chat(self, system: str, user: str) -> str:
        data = self._post(
            "/api/chat",
            {
                "model": self.chat_model,
                "stream": False,
                "options": {"temperature": 0.1},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        return data["message"]["content"].strip()
