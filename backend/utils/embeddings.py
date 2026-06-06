"""BGE embedding utility via sentence-transformers."""
import logging
import numpy as np
from typing import Union

logger = logging.getLogger("nora.embeddings")

_model = None


def get_embed_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from backend.config import EMBED_MODEL, HF_TOKEN
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL, token=HF_TOKEN or None)
        logger.info("Embedding model loaded.")
    return _model


def embed(texts: Union[str, list[str]]) -> np.ndarray:
    model = get_embed_model()
    single = isinstance(texts, str)
    if single:
        texts = [texts]
    prefixed = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
    vecs = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return vecs[0] if single else vecs


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def batch_cosine_sim(query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    return corpus_vecs @ query_vec
