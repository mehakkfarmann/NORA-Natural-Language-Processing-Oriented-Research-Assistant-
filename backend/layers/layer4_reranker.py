"""NORA — Layer 4: Evidence-Grounded Embedding Reranker"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from backend.config_production import (
    DEFAULT_CITATION_SCORE,
    DEFAULT_DOMAIN_ALIGN_SCORE,
    DOMAIN_RELAX_STEP,
    ENABLE_BEST_EFFORT_MODE,
    BEST_EFFORT_MAX_PAPERS,
    MIN_DOMAIN_ALIGN_THRESHOLD,
    MIN_RELEVANCE_THRESHOLD,
    RELEVANCE_RELAX_STEP,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankerConfig:
    preferred_model: str = "fallback"
    model_cache_dir: str = "data/models"
    max_seq_length: int = 512
    batch_size: int = 16
    embedding_weight: float = 0.60
    domain_align_weight: float = 0.25
    citation_weight: float = 0.15
    min_relevance_score: float = 0.40
    min_domain_align: float = 0.30
    top_k_refine: int = 50

    def validate_weights(self) -> bool:
        total = self.embedding_weight + self.domain_align_weight + self.citation_weight
        return abs(total - 1.0) < 0.01


@dataclass
class RerankedPaper:
    paper: Any
    embedding_score: float = 0.0
    domain_align_score: float = 0.0
    citation_score: float = 0.0
    composite_score: float = 0.0
    cross_encoder_score: Optional[float] = None
    passed_relevance_threshold: bool = False
    passed_domain_threshold: bool = False

    @property
    def is_valid_for_downstream(self) -> bool:
        return self.passed_relevance_threshold and self.passed_domain_threshold

    def to_dict(self) -> Dict[str, Any]:
        if isinstance(self.paper, dict):
            pid = self.paper.get("paper_id", "unknown")
            title = self.paper.get("title", "unknown")
        else:
            pid = getattr(self.paper, "paper_id", "unknown")
            title = getattr(self.paper, "title", "unknown")
        return {
            "paper_id": pid,
            "title": title,
            "composite_score": self.composite_score,
            "embedding_score": self.embedding_score,
            "domain_align_score": self.domain_align_score,
            "citation_score": self.citation_score,
            "is_valid": self.is_valid_for_downstream,
        }


class EmbeddingModelLoader:
    _loaded_models: Dict[str, Any] = {}

    @classmethod
    def load(cls, config: RerankerConfig):
        model_key = config.preferred_model
        if model_key in cls._loaded_models:
            return cls._loaded_models[model_key]
        model = cls._try_load(config.preferred_model, config)
        if model is None:
            for fallback in ["SciBERT", "fallback"]:
                if fallback == config.preferred_model:
                    continue
                model = cls._try_load(fallback, config)
                if model:
                    logger.info("[Layer4] Using fallback model: %s", fallback)
                    break
        if model is None:
            model = cls._load_miniLM(config)
        cls._loaded_models[model_key] = model
        return model

    @staticmethod
    def _try_load(name: str, config: RerankerConfig):
        try:
            from sentence_transformers import SentenceTransformer
            if name == "BGE-large":
                m = SentenceTransformer("BAAI/bge-large-en-v1.5", cache_folder=config.model_cache_dir, device="cpu")
                m.max_seq_length = config.max_seq_length
                logger.info("[Layer4] Loaded BGE-large")
                return m
            elif name == "SciBERT":
                m = SentenceTransformer("allenai/scibert_scivocab_uncased", cache_folder=config.model_cache_dir, device="cpu")
                m.max_seq_length = config.max_seq_length
                logger.info("[Layer4] Loaded SciBERT")
                return m
        except Exception:
            return None
        return None

    @staticmethod
    def _load_miniLM(config: RerankerConfig):
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=config.model_cache_dir, device="cpu")
        m.max_seq_length = config.max_seq_length
        logger.info("[Layer4] Loaded fallback: all-MiniLM-L6-v2")
        return m


def rerank_papers(
    papers: List[Any],
    query: str,
    domain_context: Optional[Dict] = None,
    config: Optional[RerankerConfig] = None,
    embed_model=None,
) -> List[RerankedPaper]:
    """
    Rerank papers by semantic relevance + domain alignment + citation score.

    embed_model: Pass the already-loaded BGE from the orchestrator.
                 When provided, skips EmbeddingModelLoader entirely.
                 When None, loads its own model (SciBERT fallback chain).
    """
    if not papers:
        return []

    config = config or RerankerConfig()

    if embed_model is not None:
        model = embed_model
        logger.info("[Layer4] Using injected BGE model")
    else:
        logger.warning("[Layer4] No model injected — loading via EmbeddingModelLoader")
        model = EmbeddingModelLoader.load(config)

    query_text = query.strip()

    paper_texts = []
    for p in papers:
        if hasattr(p, "title"):
            title = getattr(p, "title", "")
            abstract = getattr(p, "abstract", "")
        elif isinstance(p, dict):
            title = p.get("title", "")
            abstract = p.get("abstract", p.get("summary", ""))
        else:
            title, abstract = "", ""
        paper_texts.append(f"{title}. {abstract[:500]}")

    try:
        query_emb = model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)[0]
        paper_embs = model.encode(paper_texts, batch_size=config.batch_size, normalize_embeddings=True, show_progress_bar=False)
    except Exception as exc:
        logger.error("[Layer4] Embedding failed: %s", exc)
        return [RerankedPaper(paper=p, composite_score=0.0) for p in papers]

    embedding_scores = [float(np.dot(query_emb, pe)) for pe in paper_embs]

    citation_counts = []
    for p in papers:
        if hasattr(p, "citations"):
            citation_counts.append(getattr(p, "citations", 0))
        elif isinstance(p, dict):
            citation_counts.append(p.get("citations", 0))
        else:
            citation_counts.append(0)
    max_cit = max(citation_counts) if citation_counts else 0
    citation_scores = [
        min(1.0, math.log10(c + 1) / math.log10(max_cit + 1)) if max_cit > 0 else DEFAULT_CITATION_SCORE
        for c in citation_counts
    ]

    domain_anchors = []
    if domain_context:
        domain_anchors = [a.lower() for a in domain_context.get("domain_anchors", [])]

    reranked: List[RerankedPaper] = []
    for i, paper in enumerate(papers):
        if hasattr(paper, "domain_align_score"):
            dom_score = getattr(paper, "domain_align_score", 0.0)
        elif isinstance(paper, dict):
            dom_score = paper.get("domain_align_score", DEFAULT_DOMAIN_ALIGN_SCORE)
        else:
            dom_score = DEFAULT_DOMAIN_ALIGN_SCORE

        if dom_score == 0.0 and domain_anchors:
            if hasattr(paper, "title"):
                text = f"{paper.title} {paper.abstract}".lower()
            elif isinstance(paper, dict):
                text = f"{paper.get('title','')} {paper.get('abstract','')}".lower()
            else:
                text = ""
            matches = sum(1 for a in domain_anchors if a in text)
            dom_score = min(1.0, matches * 0.25)

        composite = (
            embedding_scores[i] * config.embedding_weight
            + dom_score * config.domain_align_weight
            + citation_scores[i] * config.citation_weight
        )

        reranked.append(RerankedPaper(
            paper=paper,
            embedding_score=embedding_scores[i],
            domain_align_score=dom_score,
            citation_score=citation_scores[i],
            composite_score=composite,
        ))

    reranked.sort(key=lambda r: r.composite_score, reverse=True)

    rel_thresh = config.min_relevance_score
    dom_thresh = config.min_domain_align
    valid: List[RerankedPaper] = []

    for attempt in range(4):
        valid = [r for r in reranked if r.composite_score >= rel_thresh and r.domain_align_score >= dom_thresh]
        if valid:
            break
        if rel_thresh <= MIN_RELEVANCE_THRESHOLD and dom_thresh <= MIN_DOMAIN_ALIGN_THRESHOLD:
            break
        rel_thresh = max(MIN_RELEVANCE_THRESHOLD, rel_thresh - RELEVANCE_RELAX_STEP)
        dom_thresh = max(MIN_DOMAIN_ALIGN_THRESHOLD, dom_thresh - DOMAIN_RELAX_STEP)
        logger.info("[Layer4] Relaxing thresholds: relevance>=%.2f, domain>=%.2f", rel_thresh, dom_thresh)

    for r in valid:
        r.passed_relevance_threshold = True
        r.passed_domain_threshold = True

    if not valid and ENABLE_BEST_EFFORT_MODE and reranked:
        logger.warning("[Layer4] Best-effort: returning top %d below threshold", BEST_EFFORT_MAX_PAPERS)
        valid = reranked[:BEST_EFFORT_MAX_PAPERS]
        for r in valid:
            r.passed_relevance_threshold = False
            r.passed_domain_threshold = False

    logger.info(
        "[Layer4] Reranked %d -> %d valid | top score: %.3f",
        len(papers), len(valid),
        valid[0].composite_score if valid else 0.0,
    )
    return valid


def compute_gap_confidence(
    reranked_papers: List[RerankedPaper],
    gap_evidence_papers: List[str],
    config: Optional[RerankerConfig] = None,
) -> float:
    if not gap_evidence_papers:
        return 0.0
    score_map: Dict[str, float] = {}
    for r in reranked_papers:
        if isinstance(r.paper, dict):
            pid = r.paper.get("paper_id", "")
        else:
            pid = getattr(r.paper, "paper_id", "")
        if pid:
            score_map[pid] = r.composite_score
    supporting = [score_map[pid] for pid in gap_evidence_papers if pid in score_map]
    if not supporting:
        return 0.0
    mean_score = sum(supporting) / len(supporting)
    evidence_boost = min(1.0, 0.1 * math.log(len(supporting) + 1))
    return min(1.0, max(0.0, mean_score * (1 + evidence_boost)))


__all__ = [
    "RerankerConfig",
    "RerankedPaper",
    "EmbeddingModelLoader",
    "rerank_papers",
    "compute_gap_confidence",
]
