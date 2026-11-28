from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class FaissVectorStore:
    """Handles BGE embeddings and exact cosine-similarity FAISS indexing/search."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(model_name, device=self.device)
        self.index: faiss.IndexFlatIP | None = None

    def encode(
        self, texts: Sequence[str], batch_size: int = 64, show_progress: bool = False
    ) -> np.ndarray:
        return self.embedder.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        ).astype(np.float32)

    def build_and_save(self, texts: Sequence[str], index_path: Path) -> int:
        embeddings = self.encode(texts, show_progress=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        return self.index.ntotal

    def load(self, index_path: Path) -> None:
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        self.index = faiss.read_index(str(index_path))

    def search(self, queries: Sequence[str], top_k: int = 3) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Load or build a FAISS index before searching.")
        q_vecs = self.encode(queries, show_progress=False)
        return self.index.search(q_vecs, top_k)
