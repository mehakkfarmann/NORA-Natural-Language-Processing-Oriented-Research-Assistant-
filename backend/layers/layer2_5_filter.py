"""NORA — Layer 2.5: Deterministic Relevance Filter"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.layers.layer1_fetcher import Paper
from backend.config_production import (
    INITIAL_RELEVANCE_THRESHOLD,
    MIN_RELEVANCE_THRESHOLD,
    RELEVANCE_RELAX_STEP,
    ENABLE_BEST_EFFORT_MODE,
    BEST_EFFORT_MAX_PAPERS,
    LOG_FILTERING_DECISIONS,
)

logger = logging.getLogger(__name__)

MIN_PAPERS_OUT = 3

# Composite scoring weights
W_SEMANTIC  = 0.55
W_ANCHOR    = 0.25
W_SUBDOMAIN = 0.10
W_CATEGORY  = 0.05
W_NEGATIVE  = 0.20

# 0.62 was blocking legitimate ~0.58 papers while old keyword-rich papers
# still passed at 0.70+. Drift penalty now does the real filtering work.
_SEMANTIC_HARD_GATE = 0.55

# DOMAIN BUCKETS

_DOMAIN_BUCKETS: list[tuple[str, list[str], int]] = [
    ("power_systems",    ["inverter", "photovoltaic", "grid-connected", "harmonic",
                          "pwm", "converter", "power quality", "renewable energy",
                          "solar panel", "wind turbine", "mppt"], 2),
    ("medical_devices",  ["insulin", "glucose", "diabetes", "clinical trial",
                          "patient", "dosage", "glycemic"], 2),
    ("process_control",  ["wastewater", "treatment plant", "greenhouse gas", "effluent",
                          "bioreactor", "aeration", "dissolved oxygen"], 2),
    ("manufacturing",    ["lean six sigma", "supply chain", "production line",
                          "quality management", "kaizen", "six sigma"], 2),
    ("elearning",        ["e-learning", "adaptive learning", "question bank",
                          "student motivation", "learning management", "moodle",
                          "game-based learning", "mobile game", "gamification",
                          "educational game", "e-learning platform"], 2),
    ("ndt_inspection",   ["nondestructive testing", "weld inspection", "ultrasonic",
                          "ndt"], 2),
    ("robotics_control", ["robot arm", "autonomous vehicle", "path planning",
                          "obstacle avoidance", "servo motor"], 2),
]

# DOMAIN-INTENT ANCHOR MAP
# Keys = intent words likely to appear in a query.
# Values = domain-discriminating phrases that separate relevant papers from
#          foundational/off-topic papers that share the topic keyword.
#
# Example: query "fuzzy logic testing approach"
#   Old _auto_anchors → ["fuzzy", "logic", "testing", "approach"]
#   "fuzzy" matches the 1987 theory book → high anchor score → passes
#
#   New _domain_intent_anchors → ["software testing", "test case", "test suite",
#                                  "regression", "coverage", "fault detection"]
#   The 1987 book has none of these → anchor score = 0.0 → filtered out

_INTENT_ANCHOR_MAP: Dict[str, List[str]] = {
    # Software testing domain
    "testing":      ["software testing", "test case", "test suite", "regression testing",
                     "coverage", "fault detection", "test generation", "test oracle",
                     "mutation testing", "test automation"],
    "test":         ["software testing", "test case", "test suite", "test coverage",
                     "fault detection", "test generation"],
    # Detection / classification
    "detection":    ["detection algorithm", "classifier", "false positive", "precision recall",
                     "anomaly detection", "intrusion detection", "object detection"],
    # Reliability / quality
    "reliability":  ["software reliability", "fault tolerance", "failure rate", "mtbf",
                     "reliability estimation", "dependability"],
    "quality":      ["software quality", "quality metric", "defect prediction",
                     "quality assurance", "code quality"],
    # Security
    "vulnerability": ["vulnerability detection", "security flaw", "cve", "exploit",
                      "static analysis", "taint analysis"],
    "security":     ["cybersecurity", "intrusion detection", "penetration testing",
                     "malware", "authentication", "encryption"],
    # NLP / LLM
    "hallucination": ["hallucination", "factuality", "faithfulness", "grounding",
                      "llm evaluation", "language model"],
    "language":     ["natural language processing", "language model", "transformer",
                     "text classification", "nlp"],
    # Privacy / federated
    "privacy":      ["differential privacy", "federated learning", "data privacy",
                     "anonymization", "gdpr"],
    "federated":    ["federated learning", "distributed training", "privacy preserving",
                     "client aggregation", "model aggregation"],
    # Verification / formal methods
    "verification": ["formal verification", "model checking", "theorem proving",
                     "specification", "correctness"],
    # Prediction / ML
    "prediction":   ["prediction model", "machine learning", "neural network",
                     "classification", "regression model", "deep learning"],
    # Optimization
    "optimization": ["optimization algorithm", "genetic algorithm", "metaheuristic",
                     "evolutionary", "search-based"],
}


def _domain_intent_anchors(query: str, research_focus: Optional[str]) -> List[str]:
    """
    Returns domain-discriminating anchor phrases based on intent words
    found in the query, NOT raw query words.

    Falls back to filtered word extraction only if no intent word matches —
    this prevents the fallback from producing useless single-word anchors
    like "fuzzy" or "logic".
    """
    text = f"{query} {research_focus or ''}".lower()
    anchors: List[str] = []

    for intent_word, phrases in _INTENT_ANCHOR_MAP.items():
        if intent_word in text:
            anchors.extend(phrases)

    if anchors:
        seen: set = set()
        deduped = []
        for a in anchors:
            if a not in seen:
                seen.add(a)
                deduped.append(a)
        return deduped[:12]

    STOP = {"using", "with", "from", "that", "this", "paper", "study",
            "analysis", "method", "approach", "system", "model", "based",
            "for", "and", "the", "of", "in", "a", "an", "are", "have",
            "been", "more", "also", "when", "each", "they", "their",
            "fuzzy", "logic", "neural", "network"}
    words = re.findall(r"\b[a-zA-Z]{5,}\b", text)
    return [w for w in words if w not in STOP][:8]


def _domain_drift_penalty(query: str, paper: Any) -> float:
    """Returns multiplier in [0.10, 1.0]."""
    text  = f"{_get_title(paper)} {_get_summary(paper)}".lower()
    q_low = query.lower()
    for _, keywords, min_hits in _DOMAIN_BUCKETS:
        p_hits = sum(1 for kw in keywords if kw in text)
        if p_hits < min_hits:
            continue
        q_hits = sum(1 for kw in keywords if kw in q_low)
        if q_hits > 0:
            continue
        return max(0.10, 1.0 - (p_hits - min_hits + 1) * 0.25)
    return 1.0


def _recency_multiplier(paper: Any) -> float:
    """
    Penalises old papers multiplicatively AFTER composite scoring.
    A paper from 1987 with a great semantic score still gets cut down here.
    """
    year = getattr(paper, "year", None)
    if year is None and isinstance(paper, dict):
        year = paper.get("year", 0)
    year = year or 0
    if year >= 2020: return 1.00
    if year >= 2018: return 0.92
    if year >= 2015: return 0.80
    if year >= 2010: return 0.65
    return 0.45  # pre-2010 papers get heavily penalised


def _build_reference_text(query: str, research_focus: Optional[str]) -> str:
    if research_focus and research_focus.strip():
        return f"{query} {research_focus.strip()}"
    return query


# PUBLIC ENTRY POINT

def llm_relevance_filter(
    papers: List[Any],
    query: str,
    groq_client,
    embed_model,
    domain_context: Optional[Dict] = None,
    *,
    research_focus: Optional[str] = None,
) -> Tuple[List[Any], List[str]]:

    if not papers:
        return [], []

    logger.info("[L2.5] papers=%d | query='%s' | focus='%s'",
                len(papers), query[:60], (research_focus or "")[:40])

    ctx = domain_context or {}

    if not research_focus:
        research_focus = ctx.get("research_focus") or ctx.get("focus") or None

    aspects  = _get_aspects(query, research_focus)
    ref_text = _build_reference_text(query, research_focus)

    logger.info("[L2.5] ref_text='%s' | aspects=%s", ref_text[:80], aspects)

    papers = _score_and_attach_semantic(papers, ref_text, embed_model)

    scored = _composite_score_all(papers, query, research_focus, aspects, ctx)

    if LOG_FILTERING_DECISIONS:
        for item in scored[:10]:
            logger.info("[L2.5] score=%.3f sem=%.3f drift=%.2f recency=%.2f | %s",
                        item["score"], item["semantic"], item["drift"],
                        item.get("recency", 1.0),
                        _get_title(item["paper"])[:70])

    # Dynamic threshold relaxation
    threshold = INITIAL_RELEVANCE_THRESHOLD
    filtered  = []
    for _ in range(3):
        filtered = [p for p in scored if p.get("score", 0.0) >= threshold]
        if filtered or threshold <= MIN_RELEVANCE_THRESHOLD:
            break
        threshold = max(MIN_RELEVANCE_THRESHOLD, threshold - RELEVANCE_RELAX_STEP)
        logger.info("[L2.5] Relaxing threshold → %.2f", threshold)

    filtered.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    if not filtered and ENABLE_BEST_EFFORT_MODE and scored:
        logger.warning("[L2.5] Best-effort: top %d below threshold", BEST_EFFORT_MAX_PAPERS)
        filtered = sorted(scored, key=lambda x: x.get("score", 0.0), reverse=True)[:BEST_EFFORT_MAX_PAPERS]

    final_papers = []
    for item in filtered:
        paper = item["paper"]
        if hasattr(paper, "relevance_score"):
            paper.relevance_score = item["score"]
        final_papers.append(paper)

    logger.info("[L2.5] %d/%d papers passed", len(final_papers), len(papers))
    return final_papers, aspects


# BGE SCORING

def _score_and_attach_semantic(
    papers: List[Any], ref_text: str, embed_model
) -> List[Any]:
    if embed_model is None:
        logger.warning("[L2.5] No embed_model — defaulting semantic to 0.5")
        for p in papers:
            _set_attr(p, "_semantic_score", 0.5)
        return papers
    try:
        q_vec = embed_model.encode([ref_text], normalize_embeddings=True,
                                   show_progress_bar=False)[0]
        texts = [f"{_get_title(p)} {_get_summary(p)[:300]}" for p in papers]
        vecs  = embed_model.encode(texts, normalize_embeddings=True,
                                   batch_size=64, show_progress_bar=False)
        scored = []
        for p, v in zip(papers, vecs):
            sim = float(np.clip(np.dot(q_vec, v), 0.0, 1.0))
            _set_attr(p, "_semantic_score", sim)
            scored.append((p, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored]
    except Exception as e:
        logger.warning("[L2.5] BGE failed (%s) — defaulting to 0.5", e)
        for p in papers:
            _set_attr(p, "_semantic_score", 0.5)
        return papers


def _set_attr(p: Any, key: str, val: Any) -> None:
    if hasattr(p, "__dict__"):
        setattr(p, key, val)
    elif isinstance(p, dict):
        p[key] = val


# COMPOSITE SCORER

def _composite_score_all(
    papers: List[Any],
    query: str,
    research_focus: Optional[str],
    aspects: List[str],
    ctx: Optional[Dict],
) -> List[Dict]:
    if not papers:
        return []

    ctx_anchors = [str(a).lower() for a in (ctx or {}).get("domain_anchors", []) if a]
    anchors     = ctx_anchors or _domain_intent_anchors(query, research_focus)

    subdomains = [str(s).lower() for s in (ctx or {}).get("subdomains", []) if s]
    negatives  = [str(n).lower() for n in (ctx or {}).get("negative_constraints", []) if n]
    target_cat = str((ctx or {}).get("arxiv_category", "")).lower()
    full_query = f"{query} {research_focus or ''}".strip()

    logger.info("[L2.5] anchors=%s", anchors[:5])

    results = []
    for paper in papers:
        text = f"{_get_title(paper)} {_get_summary(paper)}".lower()

        sem = getattr(paper, "_semantic_score", None)
        if sem is None:
            sem = paper.get("_semantic_score", 0.5) if isinstance(paper, dict) else 0.5

        if sem < _SEMANTIC_HARD_GATE:
            results.append({"paper": paper, "score": 0.0, "semantic": sem,
                            "anchor": 0.0, "drift": 0.0, "recency": 1.0})
            continue

        anchor_score    = (sum(1 for a in anchors if a in text) / max(1, len(anchors))
                           if anchors else 0.0)
        subdomain_score = (sum(1 for s in subdomains if s in text) / max(1, len(subdomains))
                           if subdomains else 0.0)
        category_score  = 1.0 if target_cat and target_cat in _get_cats(paper) else 0.0
        negative_score  = (sum(1 for n in negatives if n in text) / max(1, len(negatives))
                           if negatives else 0.0)
        aspect_score    = (sum(1 for a in aspects if a in text) / max(1, len(aspects))
                           if aspects else 0.0)

        base = (W_SEMANTIC  * sem
                + W_ANCHOR    * max(anchor_score, aspect_score)
                + W_SUBDOMAIN * subdomain_score
                + W_CATEGORY  * category_score
                - W_NEGATIVE  * negative_score)

        drift   = _domain_drift_penalty(full_query, paper)
        recency = _recency_multiplier(paper)
        final   = float(np.clip(base * drift * recency, 0.0, 1.0))

        results.append({"paper": paper, "score": final, "semantic": sem,
                        "anchor": anchor_score, "drift": drift, "recency": recency})
    return results


# HELPERS

def _get_title(p: Any) -> str:
    if hasattr(p, "title"):    return getattr(p, "title", "") or ""
    return p.get("title", "") if isinstance(p, dict) else ""

def _get_summary(p: Any) -> str:
    if hasattr(p, "abstract"): return getattr(p, "abstract", "") or ""
    return p.get("abstract", p.get("summary", "")) if isinstance(p, dict) else ""

def _get_cats(p: Any) -> List[str]:
    if hasattr(p, "categories"):
        return [str(c).lower() for c in (getattr(p, "categories", []) or [])]
    return [str(c).lower() for c in p.get("categories", [])] if isinstance(p, dict) else []

def _get_aspects(query: str, research_focus: Optional[str] = None) -> List[str]:
    STOP = {"using", "with", "from", "that", "this", "paper", "study",
            "analysis", "method", "approach", "system", "model", "based",
            "for", "and", "the", "of", "in", "a", "an", "are", "have",
            "been", "more", "also", "when", "each", "they", "their"}
    text  = f"{query} {research_focus or ''}".lower()
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text)
    return [w for w in words if w not in STOP][:10]


__all__ = ["llm_relevance_filter"]
