"""
backend/orchestrator.py  —  NORA Pipeline Orchestrator

Integrates:
  Layer 0a — layer0_domain.py        (deterministic intent + domain detection)
  Layer 0b — layer0_query.py         (Layer0QueryProcessor: strict query + variants)
  Layer 1  — layer1_fetcher.py       (IntentAwareRouter, LLMRelevanceFilter, enrichment)
  Layer 3  — gap_extractor.py        (GapExtractor / extract_gaps)
  Layer 4  — layer4_idea_generator.py (run_layer4 / IdeaGenerator)

PIPELINE FLOW (run_full_pipeline)
  Layer 0a: extract_domain_context(raw_query)
     → domain, subdomain, arxiv_category, negative_constraints, openalex_concepts …
  Layer 0b: Layer0QueryProcessor.process(raw_query)
     → strict_query, query_variants, retrieval_filters, domain_confidence, narrowing_mode
  Merge 0a + 0b into domain_context

  Layer 1: fetch_papers(smart_query, domain_context, …)
     → Papers (arXiv + S2 + OpenAlex, filtered, enriched, LLM-precision-gated)

  Layer 2.5: llm_relevance_filter  (optional, if available)
  Layer 2:   SemanticPaperRanker
  Layer 2.75: synthesize_results
  Layer 4-reranker: rerank_papers

  Layer 3: extract_gaps(papers, groq, model, domain_context)
     → Gaps (validated evidence quotes + refined gap_statements)

  Layer 4: run_layer4(gaps, query, domain, groq, embed_model, paper_evidences)
     → Ideas (3-stage proposals, domain-locked, novelty-scored)

  Persist: save_run_result + update_run_status
"""

from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path

_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

import json
import logging
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from backend.config import BGE_MODEL, GROQ_MODEL, MAX_PAPERS_FOR_GAP_EXTRACTION
from backend.config_production import ENABLE_BEST_EFFORT_MODE, BEST_EFFORT_MAX_PAPERS

from backend.database import init_db, save_run_result, update_run_status

from backend.layers.layer0_domain import extract_domain_context

from backend.layers.layer0_query import (
    Layer0QueryProcessor,
    ProcessedQuery,
)

from backend.layers.layer1_fetcher import (
    Paper,
    PaperCache,
    ConfidenceConfig,
    _fetch_with_fallback,
    fetch_papers,
    _extract_sections_from_text,
    _dedup as _l1_dedup,
)

from backend.layers.layer2_5_filter import llm_relevance_filter
from backend.layers.layer2_ranker import SemanticPaperRanker
from backend.layers.layer2_75_synthesizer import synthesize_results

from backend.layers.gap_extractor import extract_gaps as run_layer3

from backend.layers.layer4_idea_generator import run_layer4
from backend.layers.layer4_reranker import RerankerConfig, rerank_papers

from backend.utils.gap_dedup import deduplicate_gaps
from backend.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


# MODEL CACHE  (singleton per process — avoids reloading on every request)

_model_cache: Dict[str, Any] = {"bge": None, "groq": None}
_model_lock = threading.Lock()


def get_cached_models(device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with _model_lock:
        if _model_cache["bge"] is None:
            logger.info("[Pipeline] Loading BGE embed model: %s", BGE_MODEL)
            _model_cache["bge"] = SentenceTransformer(
                BGE_MODEL, device=device, cache_folder="data/models"
            )
        if _model_cache["groq"] is None:
            _model_cache["groq"] = LLMClient()
    return _model_cache["bge"], _model_cache["groq"]


# LAYER 0 — ALIGNED DOMAIN + QUERY EXTRACTION

def _run_layer0(raw_query: str, groq: LLMClient) -> Tuple[str, Dict]:
    """
    Run Layer 0a (domain detection) then Layer 0b (query processing) and
    merge their outputs into a single domain_context dict.

    Returns
    -------
    smart_query  : str   — strict retrieval string from Layer 0b
    domain_context : Dict — merged fields consumed by all downstream layers
    """
    domain_context: Dict = extract_domain_context(raw_query, groq, GROQ_MODEL)
    domain_context["original_query"] = raw_query   # MUST be set before Layer 3 runs

    qp = Layer0QueryProcessor()
    q_res: ProcessedQuery = qp.process(raw_query)

    # Merge Layer 0b fields into domain_context.
    # to_dict() produces ALL keys Layer 1 + Layer 4 expect:
    #   strict_query, query_variants, retrieval_filters,
    #   arxiv_category, s2_fields, openalex_concepts,
    #   negative_constraints, domain_confidence, narrowing_mode, …
    # Layer 0a fields (intent_domain, intent_subdomain, etc.) were already set
    # above; Layer 0b values WIN for overlapping keys (they are more specific).
    domain_context.update(q_res.to_dict())

    # Keep original_query intact — to_dict() overwrites it with raw_query anyway
    # but be explicit so nothing silently clobbers user intent.
    domain_context["original_query"] = raw_query

    smart_query = q_res.strict_query
    return smart_query, domain_context


# HELPERS

def _preflight_llm_check(groq_client: LLMClient, model_name: str) -> bool:
    try:
        r = groq_client.generate(
            prompt="OK", model=model_name, max_tokens=5, temperature=0.1
        )
        return bool(r and len(r.strip()) > 0)
    except Exception:
        return False


def _to_serializable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(i) for i in obj]
    if isinstance(obj, Paper) and hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return _to_serializable(obj.model_dump())
    return str(obj)


def _prepare_final_output(
    raw_query: str,
    smart_query: str,
    papers: List[Any],
    gaps: List[Dict],
    ideas: List[Dict],
    domain_context: Dict,
    status: str = "completed",
) -> Dict[str, Any]:
    return {
        "query":                 raw_query,
        "smart_query":           smart_query,
        "status":                status,
        "papers_count":          len(papers),
        "papers":                [_to_serializable(p) for p in papers],
        "gaps":                  [_to_serializable(g) for g in gaps],
        "ideas":                 [_to_serializable(i) for i in ideas],
        "completed_at":          time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "capabilities_disabled": domain_context.get("capabilities_disabled", []),
    }


# GAP CONFIDENCE

def _compute_gap_confidence(gap: Dict, cfg: ConfidenceConfig) -> float:
    # Gaps from faithful_extraction carry their own pre-scored confidence
    if gap.get("gap_source") == "faithful_extraction":
        return gap.get("extraction_confidence", 0.75)

    evidence_count = max(1, gap.get("evidence_count", 1))
    cross_support  = gap.get("cross_paper_support", 1)
    citations_raw  = gap.get("supporting_citations", 0)
    domain_align   = gap.get("domain_alignment", 0.70)
    methods        = gap.get("distinct_methods", [])
    contradictions = len(gap.get("contradictions", []))
    is_grounded    = gap.get("is_grounded", False)

    evidence_agreement    = min(1.0, cross_support / evidence_count)
    citation_quality      = min(1.0, citations_raw / 50.0) if citations_raw > 0 else 0.40
    methodology_diversity = (
        min(1.0, len(set(methods)) / 3.0) if methods
        else min(1.0, evidence_count / 3.0)
    )
    extraction_consistency = domain_align
    contradiction_penalty  = max(0.0, 1.0 - contradictions * 0.25)

    confidence = (
        0.30 * evidence_agreement
        + 0.25 * citation_quality
        + 0.20 * methodology_diversity
        + 0.15 * extraction_consistency
        + 0.10 * contradiction_penalty
    )
    if is_grounded:
        confidence = min(1.0, confidence + 0.08)

    return round(min(1.0, confidence), 3)


# PROVENANCE ENFORCEMENT

def _enforce_provenance(gaps: List[Dict]) -> List[Dict]:
    for gap in gaps:
        gap.setdefault("supporting_papers",  [])
        gap.setdefault("supporting_quotes",  [])
        gap.setdefault("contradictions",     [])
        gap.setdefault("missing_evidence",   [])
        gap.setdefault("confidence",         gap.get("extraction_confidence", 0.0))
        # Propagate extraction_confidence so IdeaService filters never see 0
        if "extraction_confidence" not in gap:
            gap["extraction_confidence"] = gap.get("confidence", 0.75)
    return gaps


# BATCHED EMBEDDING UTILS

def _batch_encode(
    texts: List[str], BGE: SentenceTransformer, batch_size: int = 32
) -> np.ndarray:
    return BGE.encode(texts, normalize_embeddings=True, batch_size=batch_size)


def _align_gaps_to_query(
    gaps: List[Dict], query: str, BGE: SentenceTransformer
) -> List[Dict]:
    if not gaps:
        return gaps
    gap_texts = [g.get("gap_description", "") for g in gaps]
    vecs      = _batch_encode(gap_texts + [query], BGE)
    query_vec = vecs[-1]
    for i, gap in enumerate(gaps):
        gap["domain_alignment"] = round(
            float(np.clip(np.dot(vecs[i], query_vec), 0.0, 1.0)), 3
        )
    return gaps


def _align_gaps_to_focus(
    gaps: List[Dict], focus: str, BGE: SentenceTransformer
) -> List[Dict]:
    if not gaps or not focus:
        return gaps
    gap_texts = [g.get("gap_description", "") for g in gaps]
    vecs      = _batch_encode(gap_texts + [focus], BGE)
    focus_vec = vecs[-1]
    for i, gap in enumerate(gaps):
        fa = round(float(np.dot(vecs[i], focus_vec)), 3)
        gap["focus_alignment"] = fa
        sig = gap.get("research_significance", 0.7)
        if fa >= 0.55:
            gap["research_significance"] = min(1.0, sig + 0.20)
        elif fa >= 0.40:
            gap["research_significance"] = min(1.0, sig + 0.10)
        else:
            gap["research_significance"] = max(0.0, sig - 0.30)
    return gaps


def _dedup_ideas(
    ideas: List[Dict], BGE: SentenceTransformer, threshold: float = 0.85
) -> List[Dict]:
    if len(ideas) <= 1:
        return ideas
    sigs = [
        f"{i['title']}|{i.get('primary_method', '')}|{i.get('gap_description', '')[:80]}"
        for i in ideas
    ]
    embs    = _batch_encode(sigs, BGE)
    kept:   List[Dict]       = [ideas[0]]
    kept_e: List[np.ndarray] = [embs[0]]
    for idea, emb in zip(ideas[1:], embs[1:]):
        if not any(float(np.dot(emb, ke)) > threshold for ke in kept_e):
            kept.append(idea)
            kept_e.append(emb)
    return kept


# PAPER EVIDENCES RECONSTRUCTION

def _reconstruct_paper_evidences(
    gaps: List[Dict],
    papers: List[Any],
) -> List[Dict]:
    """
    Build the paper_evidences list that Layer 4 uses as paper_context.

    For PDF uploads, extracts methodology/results sections for richer context.
    For S2/OA/arXiv papers, uses abstract (with full_text fallback).
    """
    def _pget(p, key, default=""):
        if isinstance(p, dict):
            return p.get(key, default)
        return getattr(p, key, default)

    paper_map: Dict[int, Dict] = {}
    for i, p in enumerate(papers):
        pidx     = i + 1
        _source   = _pget(p, "source", "")
        _sections = _pget(p, "sections", None) or {}

        if _source == "pdf_upload" and _sections:
            parts = []
            for key in ("introduction", "methodology", "results", "discussion"):
                sec_text = (_sections.get(key) or "").strip()
                if sec_text:
                    parts.append(f"[{key.upper()}]\n{sec_text[:1200]}")
            rich_ctx     = "\n\n".join(parts)[:3600]
            raw_abstract = rich_ctx if rich_ctx else (_pget(p, "abstract", "") or "")[:1200]
        else:
            raw_abstract = (
                (_pget(p, "abstract", "") or "").strip()
                or (_pget(p, "full_text", "") or "").strip()[:500]
                or "No abstract available."
            )[:1200]

        paper_map[pidx] = {
            "paper_index":       pidx,
            "paper_id":          _pget(p, "paper_id", f"paper_{i}"),
            "title":             _pget(p, "title", ""),
            "raw_abstract":      raw_abstract,
            "evidence":          [],
            "evidence_richness": 0.0,
        }

    seen: Dict[int, set] = {pidx: set() for pidx in paper_map}
    for gap in gaps:
        for ev in gap.get("grounding_evidence", []):
            pidx = ev.get("paper_index")
            if pidx not in paper_map:
                continue
            fp = ev.get("text", "")[:60]
            if fp in seen[pidx]:
                continue
            seen[pidx].add(fp)
            paper_map[pidx]["evidence"].append({
                "type":       ev.get("type", "unknown"),
                "text":       ev.get("text", ""),
                "confidence": ev.get("confidence", 0.5),
            })

    for entry in paper_map.values():
        evs = entry["evidence"]
        if evs:
            avg_conf       = sum(e["confidence"] for e in evs) / len(evs)
            type_diversity = len({e["type"] for e in evs}) / 8.0
            entry["evidence_richness"] = round(
                min(1.0, avg_conf * (1 + type_diversity)), 3
            )

    return list(paper_map.values())


# PDF PAPER BUILDER

def _build_paper_from_pdf(pdf_text: str, filename: str) -> Paper:
    """
    Build a single Paper object from raw PDF text.

    Extracts sections first, builds a rich paper context from
    abstract + introduction + methodology (for context) plus
    limitations / future work / discussion / conclusion (for gaps).
    Attaches sections dict so Layer 1 enrichment is skipped.
    """
    import re as _re
    pdf_text = _re.sub(r"  +", " ", pdf_text)

    sections = _extract_sections_from_text(pdf_text)

    if sections:
        context_parts = [
            sections[key]
            for key in ("abstract", "introduction", "methodology")
            if key in sections
        ]
        gap_parts = [
            sections[key]
            for key in ("limitations", "future_work", "discussion", "conclusion", "results")
            if key in sections
        ]
        context = " ".join(context_parts)[:3000] if context_parts else ""
        gaps_text = " ".join(gap_parts)[:3000] if gap_parts else ""
        abstract = (context + "\n\n" + gaps_text).strip()[:4500] if context and gaps_text else (context or gaps_text or pdf_text[:3000])
    else:
        abstract = pdf_text[:3000]

    clean_name = Path(filename).stem.replace("_", " ").replace("-", " ")
    # Smarter title extraction: scan first 30 lines for a plausible title.
    # A paper title is typically 15-120 chars and does NOT end with a period
    # (i.e. it's not a complete sentence from the abstract/introduction).
    # Also skip titles that are very short (page numbers, headers) or match
    # known author/affiliation patterns.
    pdf_title = clean_name
    for raw_line in pdf_text.split("\n")[:30]:
        line = raw_line.strip()
        if not line:
            continue
        # Skip if it ends with a period (it's a sentence, likely body text)
        if line.rstrip(":.").endswith("."):
            continue
        # Skip single numbers (page numbers), very short or very long
        if line.isdigit() or len(line) < 15 or len(line) > 120:
            continue
        pdf_title = line[:150]
        break

    return Paper(
        paper_id           = f"pdf:{filename}",
        title              = pdf_title,
        abstract           = abstract,
        source             = "pdf_upload",
        year               = 2024,
        s2_id              = f"pdf:{filename}",
        domain_align_score = 0.95,
        relevance_score    = 0.95,
        quality_score      = 0.80,
        sections           = sections if sections else None,
        full_text          = pdf_text[:20_000],
    )


# SERVICE LAYER  (stateless dataclasses — no shared state between runs)

@dataclass
class RetrievalService:
    """
    Layer 1 retrieval.
    Passes the full domain_context (populated by both Layer 0a + 0b) to
    fetch_papers() so the IntentAwareRouter receives query_variants,
    arxiv_category, s2_fields, and openalex_concepts.
    """
    def run(
        self,
        smart_query: str,
        domain_context: Dict,
        BGE: SentenceTransformer,
        *,
        precision_mode: bool = False,
        research_focus: Optional[str] = None,
    ) -> List[Paper]:
        return _fetch_with_fallback(
            query           = smart_query,
            max_results     = 20,
            domain_context  = domain_context,
            embed_model     = BGE,
            precision_mode  = precision_mode,
            research_focus  = research_focus,
        )


@dataclass
class RankingService:
    """
    Layers 2.5 → 2 → 2.75 → reranker.
    Returns (reranked_papers, aspects, consensus_gaps).
    """
    def run(
        self,
        papers: List[Paper],
        smart_query: str,
        raw_query: str,
        domain_context: Dict,
        BGE: SentenceTransformer,
        groq: LLMClient,
        target_category: Optional[str],
        precision_mode: bool,
        research_focus: Optional[str] = None,
    ) -> Tuple[List[Any], List[str], List[Dict]]:

        filtered, aspects = llm_relevance_filter(
            papers, smart_query, groq, BGE,
            domain_context  = domain_context,
            research_focus  = research_focus,
        )
        if not filtered:
            logger.warning("[Ranking] Layer 2.5 empty — using raw[:5]")
            filtered = papers[:5]

        ranker = SemanticPaperRanker(BGE, target_category, precision_mode)
        ranked = ranker.rank_papers(smart_query, filtered, aspects=aspects)

        threshold = 0.55 if precision_mode else 0.35
        final, consensus_gaps, _ = synthesize_results(
            ranked, raw_query, smart_query, BGE,
            groq_client       = groq,
            gaps              = [],
            relevance_threshold = threshold,
        )
        if not final:
            logger.warning("[Ranking] Layer 2.75 empty — using ranked")
            final = ranked or papers[:5]

        cfg = RerankerConfig(
            preferred_model     = "fallback",
            min_relevance_score = 0.50 if precision_mode else 0.35,
            min_domain_align    = 0.25,
            top_k_refine        = 0,
        )
        reranked_results = rerank_papers(
            papers         = final,
            query          = smart_query,
            domain_context = domain_context,
            config         = cfg,
            embed_model    = BGE,
        )

        reranked_papers: List[Any] = []
        for rp in reranked_results:
            paper = rp.paper
            if hasattr(paper, "domain_align_score"):
                paper.domain_align_score = rp.composite_score
            elif isinstance(paper, dict):
                paper["domain_align_score"] = rp.composite_score
            reranked_papers.append(paper)

        if not reranked_papers:
            logger.warning("[Ranking] Reranker returned 0 — best-effort fallback")
            reranked_papers = list(final[:BEST_EFFORT_MAX_PAPERS])

        return reranked_papers, aspects, consensus_gaps


@dataclass
class GapService:
    """
    Layer 3 — gap extraction + confidence scoring + dedup.

    domain_context["original_query"] MUST be set before calling run()
    so GapExtractor builds its user_query correctly.  _run_layer0() sets
    it before merging Layer 0b output, so this invariant is always satisfied
    when run_full_pipeline is the caller.
    """
    def run(
        self,
        papers: List[Any],
        smart_query: str,
        domain_context: Dict,
        BGE: SentenceTransformer,
        groq: LLMClient,
        research_focus: Optional[str],
        precision_mode: bool,
        seeded_consensus_gaps: Optional[List[Dict]] = None,
    ) -> Tuple[List[Dict], List[Dict]]:

        print(f"\n{'='*60}")
        print(f"[GAP SERVICE] Starting with {len(papers)} papers")
        print(f"  original_query = '{domain_context.get('original_query','')[:80]}'")
        print(f"{'='*60}")

        all_gaps = run_layer3(
            papers         = papers[:MAX_PAPERS_FOR_GAP_EXTRACTION],
            groq_client    = groq,
            model          = GROQ_MODEL,
            domain_context = domain_context,
            embed_model    = BGE,
        )
        print(f"[GAP SERVICE] Layer 3 produced: {len(all_gaps)} gaps")

        # Merge consensus gaps from Layer 2.75 (cross-paper themes)
        if seeded_consensus_gaps:
            for cg in seeded_consensus_gaps:
                cg.setdefault("gap_type",              cg.get("gap_type", "field_gap"))
                cg.setdefault("evidence_quote",        cg.get("evidence_quote", ""))
                cg.setdefault("section",               "consensus")
                cg.setdefault("source_paper_id",       "consensus_synthesis")
                cg.setdefault("supporting_paper_ids",  [])
                cg.setdefault("evidence_count",        3)
                cg.setdefault("cross_paper_support",   3)
                cg.setdefault("is_grounded",           False)
                cg.setdefault("extraction_confidence", cg.get("confidence", 0.75))
                cg.setdefault("research_significance", 0.80)
                cg.setdefault("gap_quality",           "consensus")
                cg.setdefault("is_fallback",           False)
                cg.setdefault("gap_source",            "consensus_synthesis")
                cg.setdefault("grounding_evidence",    [])
                cg.setdefault("supporting_paper_indices", [])
                cg.setdefault("supporting_citations",  0)
                cg.setdefault("domain_alignment",      0.75)
                cg.setdefault("contradictions",        [])
                cg.setdefault("distinct_methods",      [])
            all_gaps = seeded_consensus_gaps + all_gaps
            print(
                f"[GAP SERVICE] Merged {len(seeded_consensus_gaps)} consensus gap(s) "
                f"from Layer 2.75 → total {len(all_gaps)}"
            )

        paper_evidences = _reconstruct_paper_evidences(all_gaps, papers)
        all_gaps = _enforce_provenance(all_gaps)
        all_gaps = _align_gaps_to_query(all_gaps, smart_query, BGE)

        deduped = deduplicate_gaps(all_gaps, BGE, similarity_threshold=0.88)
        print(f"[GAP SERVICE] After dedup (threshold=0.88): {len(deduped)} gaps")

        cfg = ConfidenceConfig()
        for gap in deduped:
            gap["extraction_confidence"] = _compute_gap_confidence(gap, cfg)

        min_align = 0.25
        min_conf  = 0.30

        print(f"[GAP SERVICE] Applying filter: min_align={min_align} min_conf={min_conf}")
        for g in deduped:
            tag = "✓" if g.get("is_grounded") else "✗"
            print(
                f"  [{tag}] align={g.get('domain_alignment',0):.3f} "
                f"conf={g.get('extraction_confidence',0):.3f} "
                f"ev={g.get('evidence_count',0)} | "
                f"{g.get('gap_description','')[:70]}"
            )

        final_gaps = [
            g for g in deduped
            if g.get("domain_alignment", 0.0) >= min_align
            and g.get("extraction_confidence", 0.0) >= min_conf
        ]
        print(f"[GAP SERVICE] After threshold filter: {len(final_gaps)} gaps survived")

        fallback_count = sum(1 for g in final_gaps if g.get("is_fallback"))
        if fallback_count:
            print(f"[GAP SERVICE] Dropping {fallback_count} fallback gap(s)")
            final_gaps = [g for g in final_gaps if not g.get("is_fallback")]

        if not final_gaps:
            print("[GAP SERVICE] No evidence-grounded gaps.")
            print(f"{'='*60}\n")
            return [], paper_evidences

        if research_focus:
            final_gaps = _align_gaps_to_focus(final_gaps, research_focus, BGE)
            before = len(final_gaps)
            final_gaps = [g for g in final_gaps if g.get("focus_alignment", 1.0) >= 0.35]
            print(
                f"[GAP SERVICE] Focus filter: {before} → {len(final_gaps)} "
                f"(focus='{research_focus[:50]}')"
            )

        final_gaps.sort(
            key=lambda g: (
                (1 if g.get("is_grounded") else 0) * 0.10
                + g.get("research_significance", 0) * 0.60
                + g.get("extraction_confidence", 0) * 0.30
            ),
            reverse=True,
        )

        print(f"[GAP SERVICE] Final gaps to Layer 4: {len(final_gaps)}")
        print(f"{'='*60}\n")
        return final_gaps, paper_evidences


@dataclass
class IdeaService:
    """
    Layer 4 — 3-stage research proposal generation.

    KEY ALIGNMENT: the `domain` argument to run_layer4 is now taken from
    domain_context["intent_domain"] (set by Layer 0a/0b) rather than the
    target_category arxiv code.  intent_domain is human-readable
    (e.g. "natural_language_processing") and is what the Layer 4 prompt uses
    for its DOMAIN LOCK section.  target_category (e.g. "cs.CL") is still
    passed through as a fallback.
    """
    def run(
        self,
        final_gaps: List[Dict],
        raw_query: str,
        aspects: List[str],
        target_category: Optional[str],
        BGE: SentenceTransformer,
        groq: LLMClient,
        research_focus: Optional[str],
        domain_context: Optional[Dict] = None,
        paper_evidences: Optional[List[Dict]] = None,
    ) -> List[Dict]:

        # Prefer intent_domain from Layer 0 for domain-locking in proposals.
        # Fall back to target_category (arxiv code) if intent_domain is absent.
        dc = domain_context or {}
        idea_domain = (
            dc.get("intent_domain")
            or dc.get("domain")
            or target_category
            or "Computer Science"
        )

        # Confidence gate: accept gaps with extraction_confidence OR legacy confidence key
        strong_gaps = [
            g for g in final_gaps
            if g.get("extraction_confidence", g.get("confidence", 0.75)) >= 0.30
        ]
        print(f"[IDEA SERVICE] strong_gaps (conf >= 0.30): {len(strong_gaps)}/{len(final_gaps)}")

        if not strong_gaps:
            print("[IDEA SERVICE] No strong gaps — returning empty")
            return []

        l4_query = (
            f"{raw_query} — focus: {research_focus}" if research_focus else raw_query
        )

        ideas = run_layer4(
            gaps            = strong_gaps,
            query           = l4_query,
            aspects         = aspects,
            domain          = idea_domain,
            groq            = groq,
            embed_model     = BGE,
            mock_mode       = False,
            paper_evidences = paper_evidences,
        )

        ideas = _dedup_ideas(ideas, BGE, threshold=0.85)
        ideas = [
            i for i in ideas
            if max(
                i.get("novelty_score", 0),
                i.get("proposal_strength", 0),
            ) >= 0.30
        ]

        print(f"[IDEA SERVICE] Final ideas after dedup+filter: {len(ideas)}")
        return ideas


@dataclass
class PersistenceService:
    def finalize(
        self,
        run_id: str,
        raw_query: str,
        smart_query: str,
        reranked_papers: List[Any],
        final_gaps: List[Dict],
        ideas: List[Dict],
        domain_context: Dict,
    ) -> None:
        final_output = _prepare_final_output(
            raw_query, smart_query, reranked_papers, final_gaps, ideas,
            domain_context, "completed",
        )
        save_run_result(run_id, final_gaps, ideas, final_output)
        update_run_status(
            run_id, "completed", "100",
            result_json=json.dumps(final_output),
        )

        grounded   = sum(1 for g in final_gaps if g.get("is_grounded"))
        ungrounded = len(final_gaps) - grounded
        logger.info(
            "[Pipeline] COMPLETE | papers=%d | gaps=%d (grounded=%d ungrounded=%d) | ideas=%d",
            len(reranked_papers), len(final_gaps), grounded, ungrounded, len(ideas),
        )


# PUBLIC PIPELINE ENTRY POINTS

def run_full_pipeline(
    run_id: str,
    raw_query: str,
    research_focus: Optional[str] = None,
    embed_model: Optional[SentenceTransformer] = None,
    groq_client: Optional[LLMClient] = None,
    precision_mode: bool = False,
    target_category_override: Optional[str] = None,
) -> None:
    try:
        _execute_pipeline(
            run_id, raw_query, embed_model, groq_client,
            precision_mode, target_category_override, research_focus,
        )
    except Exception as exc:
        logger.error("[Pipeline] CRASHED: %s\n%s", exc, traceback.format_exc())
        update_run_status(
            run_id, "failed", "0", error=f"{type(exc).__name__}: {str(exc)}"
        )


def run_custom_pipeline(
    run_id: str,
    raw_papers: list,
    query: str,
    research_focus: Optional[str] = None,
) -> None:
    try:
        _execute_custom_pipeline(run_id, raw_papers, query, research_focus)
    except Exception as exc:
        logger.error("[CustomPipeline] CRASHED: %s\n%s", exc, traceback.format_exc())
        update_run_status(
            run_id, "failed", "0", error=f"{type(exc).__name__}: {str(exc)}"
        )


def run_pdf_pipeline(
    run_id: str,
    pdf_text: str,
    filename: str,
    research_focus: Optional[str] = None,
) -> None:
    try:
        _execute_pdf_pipeline(run_id, pdf_text, filename, research_focus)
    except Exception as exc:
        logger.error("[PDFPipeline] CRASHED: %s\n%s", exc, traceback.format_exc())
        update_run_status(
            run_id, "failed", "0", error=f"{type(exc).__name__}: {str(exc)}"
        )


# FULL PIPELINE  (query → retrieval → ranking → gaps → ideas)

def _execute_pipeline(
    run_id: str,
    raw_query: str,
    embed_model: Optional[SentenceTransformer],
    groq_client: Optional[LLMClient],
    precision_mode: bool,
    target_category_override: Optional[str],
    research_focus: Optional[str] = None,
) -> None:
    init_db()
    BGE, groq = (
        (embed_model, groq_client)
        if (embed_model and groq_client)
        else get_cached_models()
    )

    if not _preflight_llm_check(groq, GROQ_MODEL):
        update_run_status(run_id, "failed", "10", error="LLM API unavailable")
        return

    update_run_status(run_id, "running", "10")
    logger.info("[Pipeline] Starting | query='%s'", raw_query[:60])

    # Both sub-layers run here; domain_context receives ALL their merged output.
    smart_query, domain_context = _run_layer0(raw_query, groq)

    target_category = (
        target_category_override
        or domain_context.get("arxiv_category")
    )

    update_run_status(
        run_id, "running", "25",
        result_json=json.dumps({
            "layer":        "0",
            "smart_query":  smart_query,
            "domain":       domain_context.get("intent_domain", ""),
            "subdomain":    domain_context.get("intent_subdomain", ""),
            "category":     target_category or "",
            "confidence":   domain_context.get("domain_confidence", 0),
            "narrowing":    domain_context.get("narrowing_mode", False),
        }),
    )
    logger.info(
        "[Pipeline] RAW='%s' | SMART='%s' | domain=%s | cat=%s | conf=%.2f",
        raw_query[:60], smart_query[:60],
        domain_context.get("intent_domain", "?"),
        target_category or "?",
        domain_context.get("domain_confidence", 0),
    )


    retrieval = RetrievalService()
    papers = retrieval.run(
        smart_query, domain_context, BGE,
        precision_mode  = precision_mode,
        research_focus  = research_focus,
    )

    _MIN_PAPERS_FOR_GAP = 4

    if len(papers) < _MIN_PAPERS_FOR_GAP:
        logger.info("[Pipeline] Only %d papers — attempting retrieval expansion", len(papers))
        # Broad fallback: strip all constraints and retry on raw query
        broad_ctx = {
            "original_query":       raw_query,
            "negative_constraints": [],
            "arxiv_category":       "",
            "s2_fields":            [],
            "openalex_concepts":    [],
            "query_variants":       [raw_query],
            "intent_domain":        domain_context.get("intent_domain", ""),
        }
        extra_papers = retrieval.run(
            raw_query, broad_ctx, BGE,
            precision_mode = False,
            research_focus = None,
        )
        papers = _l1_dedup(papers, extra_papers, max_results=20)
        logger.info("[Pipeline] After expansion: %d papers total", len(papers))

    if not papers:
        logger.warning("[Pipeline] Zero papers — saving empty result")
        final_output = _prepare_final_output(
            raw_query, smart_query, [], [], [], domain_context, "no_evidence"
        )
        final_output["message"] = (
            "No papers found. Try broadening your query or checking your network connection."
        )
        save_run_result(run_id, [], [], final_output)
        update_run_status(run_id, "completed", "100", result_json=json.dumps(final_output))
        return

    if len(papers) < 3:
        logger.warning("[Pipeline] Only %d papers after expansion — insufficient", len(papers))
        final_output = _prepare_final_output(
            raw_query, smart_query, papers, [], [], domain_context, "insufficient_evidence"
        )
        final_output["message"] = (
            f"Only {len(papers)} paper(s) found — not enough to extract reliable gaps. "
            "Try broadening the query, or upload a PDF directly using the PDF Analysis tab."
        )
        save_run_result(run_id, [], [], final_output)
        update_run_status(run_id, "completed", "100", result_json=json.dumps(final_output))
        return

    logger.info(
        "[Pipeline] Layer 1 | total=%d | arXiv=%d | S2=%d | OA=%d",
        len(papers),
        sum(1 for p in papers if p.source == "arxiv"),
        sum(1 for p in papers if p.source == "semantic_scholar"),
        sum(1 for p in papers if p.source == "openalex"),
    )

    ranking = RankingService()
    reranked_papers, aspects, consensus_gaps = ranking.run(
        papers, smart_query, raw_query, domain_context,
        BGE, groq, target_category, precision_mode,
        research_focus = research_focus,
    )
    logger.info("[Pipeline] Layer 2.75 consensus gaps: %d", len(consensus_gaps))

    update_run_status(run_id, "running", "75")

    gap_svc = GapService()
    final_gaps, paper_evidences = gap_svc.run(
        reranked_papers, smart_query, domain_context,
        BGE, groq, research_focus, precision_mode,
        seeded_consensus_gaps = consensus_gaps,
    )

    update_run_status(run_id, "running", "85")

    if not final_gaps:
        logger.warning("[Pipeline] No grounded gaps — insufficient evidence for idea generation")
        final_output = _prepare_final_output(
            raw_query, smart_query, reranked_papers, [], [], domain_context,
            "insufficient_evidence",
        )
        final_output["message"] = (
            "Papers were found but lacked sufficient author-stated evidence "
            "(limitations, future work) to extract reliable gaps. "
            "Try uploading a PDF directly using the PDF Analysis tab — "
            "full-text papers produce much better results."
        )
        persist = PersistenceService()
        persist.finalize(
            run_id, raw_query, smart_query, reranked_papers, [], [], domain_context
        )
        update_run_status(run_id, "completed", "100", result_json=json.dumps(final_output))
        return

    idea_svc = IdeaService()
    ideas = idea_svc.run(
        final_gaps, raw_query, aspects, target_category,
        BGE, groq, research_focus,
        domain_context  = domain_context,      # carries intent_domain for domain-locking
        paper_evidences = paper_evidences,
    )

    persist = PersistenceService()
    persist.finalize(
        run_id, raw_query, smart_query, reranked_papers, final_gaps, ideas, domain_context
    )


# CUSTOM PIPELINE  (user-supplied paper list, no retrieval)

def _execute_custom_pipeline(
    run_id: str,
    raw_papers: list,
    query: str,
    research_focus: Optional[str] = None,
) -> None:
    init_db()
    BGE, groq = get_cached_models()
    update_run_status(run_id, "running", "10")

    papers: List[Paper] = []
    for i, rp in enumerate(raw_papers):
        title    = (rp.get("title") or "").strip()
        abstract = (rp.get("abstract") or "").strip()
        if not title:
            continue
        papers.append(Paper(
            paper_id           = f"custom_{i}",
            title              = title,
            abstract           = abstract,
            source             = "user_provided",
            year               = int(rp.get("year") or 2024),
            s2_id              = f"custom_{i}",
            domain_align_score = 0.90,
            relevance_score    = 0.90,
        ))

    if not papers:
        final_output = {
            "query":        query, "status": "no_evidence",
            "papers":       [], "gaps": [], "ideas": [],
            "papers_count": 0,
            "message":      "No valid papers — check title and abstract fields.",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "capabilities_disabled": [],
        }
        save_run_result(run_id, [], [], final_output)
        update_run_status(run_id, "completed", "100", result_json=json.dumps(final_output))
        return

    update_run_status(run_id, "running", "30")

    # Run Layer 0 for domain detection (used by Layer 4 domain-locking)
    # Custom pipeline skips retrieval, but domain context still needed.
    smart_query, domain_context = _run_layer0(query, groq)
    domain_context["arxiv_category"] = (
        domain_context.get("arxiv_category") or "cs.SE"
    )

    update_run_status(run_id, "running", "40")

    all_gaps = run_layer3(
        papers         = papers,
        groq_client    = groq,
        model          = GROQ_MODEL,
        domain_context = domain_context,
        embed_model    = BGE,
    )
    all_gaps = _enforce_provenance(all_gaps)
    all_gaps = _align_gaps_to_query(all_gaps, query, BGE)

    deduped = deduplicate_gaps(all_gaps, BGE, similarity_threshold=0.88)
    cfg = ConfidenceConfig()
    for gap in deduped:
        gap["extraction_confidence"] = _compute_gap_confidence(gap, cfg)

    final_gaps = [
        g for g in deduped
        if g.get("domain_alignment", 0) >= 0.25
        and g.get("extraction_confidence", 0) >= 0.30
    ]

    if research_focus:
        final_gaps = _align_gaps_to_focus(final_gaps, research_focus, BGE)
        final_gaps = [g for g in final_gaps if g.get("focus_alignment", 1.0) >= 0.35]

    paper_evidences = _reconstruct_paper_evidences(final_gaps, papers)
    update_run_status(run_id, "running", "70")

    idea_svc = IdeaService()
    ideas = idea_svc.run(
        final_gaps, query, [], None, BGE, groq, research_focus,
        domain_context  = domain_context,
        paper_evidences = paper_evidences,
    )

    final_output = {
        "query":        query,
        "smart_query":  smart_query,
        "status":       "completed",
        "mode":         "custom",
        "papers_count": len(papers),
        "papers":       [p.to_dict() for p in papers],
        "gaps":         final_gaps,
        "ideas":        ideas,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "capabilities_disabled": [],
    }
    save_run_result(run_id, final_gaps, ideas, final_output)
    update_run_status(run_id, "completed", "100", result_json=json.dumps(final_output))


# PDF PIPELINE  (single uploaded PDF, no retrieval or ranking)

def _execute_pdf_pipeline(
    run_id: str,
    pdf_text: str,
    filename: str,
    research_focus: Optional[str] = None,
) -> None:
    """
    Gap extraction + idea generation from a single user-uploaded PDF.

    Flow:
      Layer 0  → domain detection for alignment scoring + Layer 4 domain-lock
      (no Layer 1 retrieval — PDF is the sole source)
      Layer 3  → gap extraction from PDF sections
      Layer 4  → idea generation from extracted gaps
    """
    init_db()
    BGE, groq = get_cached_models()

    update_run_status(run_id, "running", "15")
    logger.info("[PDFPipeline] Starting | file='%s' | chars=%d", filename, len(pdf_text))

    paper = _build_paper_from_pdf(pdf_text, filename)
    papers = [paper]

    sections_found = list((paper.sections or {}).keys())
    logger.info(
        "[PDFPipeline] Sections extracted: %s",
        sections_found if sections_found else "none — using full text slice",
    )
    if not sections_found:
        logger.warning(
            "[PDFPipeline] No sections found in '%s' (%d chars). "
            "Header patterns did not match — evidence extractor will use abstract-only mode.",
            filename, len(pdf_text),
        )

    update_run_status(run_id, "running", "30")

    # Derive a base query from the paper title (or filename as fallback)
    base_query = paper.title if paper.title and paper.title != Path(filename).stem else Path(filename).stem.replace("_", " ").replace("-", " ")
    query_for_alignment = (
        f"{base_query} — {research_focus}" if research_focus else base_query
    )

    # Run Layer 0 so Layer 4 gets a proper intent_domain for domain-locking
    smart_query, domain_context = _run_layer0(query_for_alignment, groq)
    domain_context["original_query"] = query_for_alignment

    all_gaps = run_layer3(
        papers         = papers,
        groq_client    = groq,
        model          = GROQ_MODEL,
        domain_context = domain_context,
        embed_model    = BGE,
    )
    logger.info("[PDFPipeline] Layer 3 → %d raw gaps", len(all_gaps))

    update_run_status(run_id, "running", "55")

    all_gaps = _enforce_provenance(all_gaps)
    all_gaps = _align_gaps_to_query(all_gaps, query_for_alignment, BGE)

    deduped = deduplicate_gaps(all_gaps, BGE, similarity_threshold=0.88)
    cfg = ConfidenceConfig()
    for gap in deduped:
        gap["extraction_confidence"] = _compute_gap_confidence(gap, cfg)

    # Softer thresholds for single-PDF mode (no cross-paper support available)
    final_gaps = [
        g for g in deduped
        if g.get("domain_alignment", 0.0)     >= 0.20
        and g.get("extraction_confidence", 0.0) >= 0.25
        and not g.get("is_fallback")
    ]

    if research_focus and final_gaps:
        final_gaps = _align_gaps_to_focus(final_gaps, research_focus, BGE)
        final_gaps = [g for g in final_gaps if g.get("focus_alignment", 1.0) >= 0.30]

    final_gaps.sort(
        key=lambda g: (
            (1 if g.get("is_grounded") else 0) * 0.10
            + g.get("research_significance", 0) * 0.60
            + g.get("extraction_confidence", 0) * 0.30
        ),
        reverse=True,
    )

    logger.info("[PDFPipeline] After filtering → %d gaps", len(final_gaps))
    update_run_status(run_id, "running", "70")

    paper_evidences = _reconstruct_paper_evidences(final_gaps, papers)

    if not final_gaps:
        logger.warning("[PDFPipeline] No gaps extracted from PDF")
        final_output = {
            "query":        query_for_alignment,
            "smart_query":  smart_query,
            "status":       "insufficient_evidence",
            "mode":         "pdf",
            "papers_count": 1,
            "papers":       [_to_serializable(paper)],
            "gaps":         [],
            "ideas":        [],
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "capabilities_disabled": [],
            "message": (
                "No research gaps could be extracted from this PDF. "
                "This usually means the paper lacks explicit limitations or future work "
                "sections, or the PDF is scanned/image-based with limited extractable text."
            ),
        }
        save_run_result(run_id, [], [], final_output)
        update_run_status(run_id, "completed", "100", result_json=json.dumps(final_output))
        return

    idea_svc = IdeaService()
    ideas = idea_svc.run(
        final_gaps, query_for_alignment, [], None, BGE, groq, research_focus,
        domain_context  = domain_context,
        paper_evidences = paper_evidences,
    )

    update_run_status(run_id, "running", "90")

    final_output = {
        "query":        query_for_alignment,
        "smart_query":  smart_query,
        "status":       "completed",
        "mode":         "pdf",
        "papers_count": 1,
        "papers":       [_to_serializable(paper)],
        "gaps":         [_to_serializable(g) for g in final_gaps],
        "ideas":        [_to_serializable(i) for i in ideas],
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "capabilities_disabled": [],
    }
    save_run_result(run_id, final_gaps, ideas, final_output)
    update_run_status(run_id, "completed", "100", result_json=json.dumps(final_output))

    logger.info(
        "[PDFPipeline] COMPLETE | file='%s' | gaps=%d | ideas=%d",
        filename, len(final_gaps), len(ideas),
    )


# EXPORTS

__all__ = [
    "run_full_pipeline",
    "run_custom_pipeline",
    "run_pdf_pipeline",
    "get_cached_models",
    "RetrievalService",
    "RankingService",
    "GapService",
    "IdeaService",
    "PersistenceService",
    "_to_serializable",
    "_prepare_final_output",
]