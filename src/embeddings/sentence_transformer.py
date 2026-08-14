"""
Sentence Transformer embedding model.
"""

from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from sentence_transformers import SentenceTransformer
import numpy as np

class SentenceEmbeddingModel:
    _model = None

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        if SentenceEmbeddingModel._model is None:
            SentenceEmbeddingModel._model = SentenceTransformer(model_name)
        self.model = SentenceEmbeddingModel._model

    def embed_query(self, text):
        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def encode(self, texts, batch_size=64):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )


    # ---------------------------
    # NEW
    # ---------------------------

    def embed_documents(
        self,
        texts,
    ) -> np.ndarray:
        """
        Embed a list of documents.
        """
        return self.encode(texts)

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Embed a single query.
        """
        return self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    # ---------------------------

    def save(
        self,
        embeddings: np.ndarray,
        path: str,
    ):

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            embeddings,
            path,
        )

    @staticmethod
    def load(path: str):

        return joblib.load(path)