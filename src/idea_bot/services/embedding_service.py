import hashlib
import pickle
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

from idea_bot.config import AppConfig


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class DeterministicEmbeddings(Embeddings):
    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]


class LangChainEmbeddingProvider(EmbeddingProvider):
    def __init__(self, client: Embeddings) -> None:
        self.client = client

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed_query(text)


class FaissIndexStore:
    """Encapsulates FAISS persistence (single responsibility)."""

    def __init__(self, config: AppConfig, embeddings: Embeddings) -> None:
        self._path = config.faiss_path
        self._embeddings = embeddings

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            with open(self._path, "rb") as f:
                return pickle.load(f)
        return {"index": None, "docstore": {}, "mapping": {}}

    def _save(self, data: dict[str, Any]) -> None:
        with open(self._path, "wb") as f:
            pickle.dump(data, f)

    def add_text(self, text: str, metadata: dict[str, Any]) -> None:
        data = self._load()
        if data["index"] is None:
            store = FAISS.from_texts([text], self._embeddings, metadatas=[metadata])
        else:
            store = FAISS(
                embedding_function=self._embeddings,
                index=data["index"],
                docstore=InMemoryDocstore(data["docstore"]),
                index_to_docstore_id=data["mapping"],
            )
            store.add_texts([text], metadatas=[metadata])
        self._save(
            {
                "index": store.index,
                "docstore": store.docstore._dict,
                "mapping": store.index_to_docstore_id,
            }
        )


def build_embedding_services(config: AppConfig) -> tuple[EmbeddingProvider, Embeddings]:
    if config.azure_endpoint and config.azure_api_key and config.azure_embedding_deployment:
        azure_embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            azure_deployment=config.azure_embedding_deployment,
        )
        return LangChainEmbeddingProvider(azure_embeddings), azure_embeddings

    fallback = DeterministicEmbeddings()
    return LangChainEmbeddingProvider(fallback), fallback
