from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from langchain_core.embeddings import Embeddings


class LocalHashEmbeddings(Embeddings):
    """Small dependency-free embedding used when the chat provider is Groq.

    Groq provides chat completions, not embeddings. Hashing keeps PDF retrieval
    local and avoids requiring an OpenAI embedding key.
    """

    def __init__(self, dimensions: int = 512) -> None:
        if dimensions < 2:
            raise ValueError("dimensions must be at least 2")
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"\w+", text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimensions
            vector[index] += 1.0

        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector] if norm else vector

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
