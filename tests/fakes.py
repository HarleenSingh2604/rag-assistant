"""A fake language model so tests run instantly with no Ollama installed."""
import re
import zlib

DIM = 256


class FakeLLM:
    def __init__(self):
        self.chat_calls = 0
        self.embed_calls = 0

    def embed(self, texts):
        self.embed_calls += 1
        vectors = []
        for text in texts:
            vec = [0.0] * DIM
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[zlib.crc32(word.encode()) % DIM] += 1.0
            vectors.append(vec)
        return vectors

    def chat(self, system, user):
        self.chat_calls += 1
        return "FAKE ANSWER"
