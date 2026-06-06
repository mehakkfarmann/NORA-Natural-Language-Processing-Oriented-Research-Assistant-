"""NORA — Layer 2: Semantic paper ranker with drift detection and recency scoring."""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import numpy as np
from sentence_transformers.util import cos_sim
from tqdm import tqdm

from backend.layers.layer1_fetcher import Paper

logger = logging.getLogger(__name__)


# DOMAIN DRIFT DETECTOR

# (domain_name, keywords, min_hits_to_trigger, hard_cap_when_penalised)
_DOMAIN_BUCKETS: list[tuple[str, list[str], int, float]] = [
    ("power_systems",   ["inverter", "photovoltaic", "grid-connected", "harmonic distortion",
                         "pwm", "converter", "voltage regulator", "power quality",
                         "renewable energy", "solar panel", "wind turbine", "mppt"], 2, 0.28),
    ("medical_devices", ["insulin", "glucose", "pancreas", "blood sugar", "diabetes",
                         "clinical trial", "patient", "dosage", "glycemic", "cgm"],  2, 0.28),
    ("process_control", ["wastewater", "treatment plant", "greenhouse gas", "effluent",
                         "bioreactor", "aeration", "dissolved oxygen", "nutrient removal"], 2, 0.28),
    ("manufacturing",   ["lean six sigma", "manufacturing", "supply chain", "production line",
                         "defect rate", "quality management", "kaizen", "six sigma"],  2, 0.28),
    ("elearning",       ["e-learning", "adaptive learning", "question bank",
                         "student motivation", "learning management", "moodle",
                         "game-based learning", "mobile game", "gamification",
                         "educational game", "e-learning platform",
                         "adaptive test", "adaptive assessment"], 2, 0.22),
    ("ndt_inspection",  ["nondestructive testing", "weld inspection", "ultrasonic",
                         "x-ray inspection", "defect sizing", "ndt"],                  2, 0.25),
    ("robotics_control",["robot arm", "autonomous vehicle", "path planning",
                         "obstacle avoidance", "servo motor", "pid controller"],       2, 0.28),
]

_METHODOLOGY_SIGNALS = [
    "testing methodology", "evaluation framework", "benchmark", "test suite",
    "software testing", "reliability estimation", "completeness measure",
    "test case", "fault detection", "coverage", "mutation testing",
    "formal verification", "model checking", "static analysis",
    "testing approach", "validation method", "quality metric",
    "regression testing", "test oracle", "test automation", "test generation",
    "software quality", "defect prediction", "test adequacy", "test effectiveness",
]


def _drift_penalty(query: str, paper: Paper) -> tuple[float, float]:
    """
    Returns (penalty_multiplier, hard_cap).
    penalty_multiplier ∈ [0.15, 1.0]
    hard_cap           ∈ [0.22, 1.0]
    """
    text  = f"{paper.title} {paper.abstract}".lower()
    q_low = query.lower()

    for domain, keywords, min_hits, cap in _DOMAIN_BUCKETS:
        p_hits = sum(1 for kw in keywords if kw in text)
        if p_hits < min_hits:
            continue
        q_hits = sum(1 for kw in keywords if kw in q_low)
        if q_hits > 0:
            continue  # user asked for this domain — no penalty
        penalty = max(0.15, 1.0 - (p_hits - min_hits + 1) * 0.20)
        logger.debug("[Layer2] drift '%s' | hits=%d | penalty=%.2f | cap=%.2f | '%s'",
                     domain, p_hits, penalty, cap, paper.title[:50])
        return penalty, cap

    return 1.0, 1.0


def _intent_bonus(paper: Paper) -> float:
    """Additive bonus [0.0, 0.15] for papers studying a methodology."""
    text = f"{paper.title} {paper.abstract}".lower()
    hits = sum(1 for s in _METHODOLOGY_SIGNALS if s in text)
    return min(0.15, hits * 0.04)


def _recency_score(paper: Paper) -> float:
    """
    Explicit recency component for the ranker.
    Returns [0.0, 1.0] — 1.0 for 2024+, decays linearly to 0 at 2005 and below.
    This is separate from Layer 1's quality_score recency, which doesn't persist
    into the ranker's base score calculation.
    """
    year = getattr(paper, "year", 0) or 0
    if year >= 2024: return 1.00
    if year >= 2022: return 0.90
    if year >= 2020: return 0.80
    if year >= 2018: return 0.65
    if year >= 2015: return 0.45
    if year >= 2010: return 0.25
    return 0.10  # anything pre-2010 is near-zero


# RANKER

class SemanticPaperRanker:
    """
    Weights:
      semantic  45%  — raw cosine [0,1]
      aspect    25%  — user research aspects
      recency   15%  — publication year score
      citation  10%  — log-normalised + age-dampened
      domain     5%  — category match

    Post-score:
      drift_penalty × multiplicative, then hard_cap ceiling
      intent_bonus  + additive
    """

    _WEIGHTS = {"semantic": 0.45, "aspect": 0.25, "recency": 0.15,
                "citation": 0.10, "domain": 0.05}

    def __init__(self, embed_model, target_category: Optional[str] = None,
                 strict_mode: bool = False) -> None:
        self.embed_model     = embed_model
        self.target_category = target_category
        self.strict_mode     = strict_mode
        self.min_score = 0.25 if strict_mode else 0.20

    def rank_papers(self, query: str, papers: List[Paper],
                    aspects: Optional[List[str]] = None) -> List[Paper]:
        if not papers:
            return []
        aspects = aspects or []
        logger.info("[Layer2] Ranking %d papers | query='%s'", len(papers), query[:60])

        texts   = [query] + [f"{p.title} {p.abstract[:300]}" for p in papers]
        all_emb = self.embed_model.encode(texts, normalize_embeddings=True,
                                          batch_size=64, show_progress_bar=False)
        q_emb  = all_emb[0]
        p_embs = all_emb[1:]

        scored = []
        for paper, p_emb in tqdm(zip(papers, p_embs), total=len(papers),
                                  desc="Ranking", leave=False):
            try:
                score = self._score(query, q_emb, p_emb, paper, aspects)
                if score >= self.min_score:
                    paper.relevance_score    = score
                    paper.domain_align_score = score
                    scored.append((paper, score))
            except Exception as e:
                logger.warning("[Layer2] score failed '%s': %s",
                               getattr(paper, "title", "?")[:40], e)

        scored.sort(key=lambda x: x[1], reverse=True)

        for i, (p, s) in enumerate(scored[:5]):
            logger.info("[Layer2] #%d %.3f yr=%d | %s",
                        i + 1, s, getattr(p, "year", 0), p.title[:70])

        return [p for p, _ in scored]

    def _score(self, query: str, q_emb: np.ndarray, p_emb: np.ndarray,
               paper: Paper, aspects: List[str]) -> float:

        sem = max(0.0, min(1.0, float(cos_sim(q_emb, p_emb))))
        paper.original_query_relevance = sem

        if aspects:
            full = f"{paper.title} {paper.abstract}".lower()
            asp  = min(1.0, sum(1 for a in aspects if a.lower() in full) / len(aspects))
        else:
            asp = 0.0

        rec = _recency_score(paper)

        year       = getattr(paper, "year", 0) or 0
        age_factor = max(0.20, 1.0 - max(0, 2025 - year) / 30.0)
        cit_raw    = min(1.0, math.log10(getattr(paper, "citations", 0) + 1) / 4.0)
        cit        = cit_raw * age_factor

        dom = (1.0 if self.target_category
               and getattr(paper, "primary_category", "") == self.target_category
               else 0.5)

        base = (sem * self._WEIGHTS["semantic"]
                + asp * self._WEIGHTS["aspect"]
                + rec * self._WEIGHTS["recency"]
                + cit * self._WEIGHTS["citation"]
                + dom * self._WEIGHTS["domain"])

        penalty, hard_cap = _drift_penalty(query, paper)
        bonus             = _intent_bonus(paper)

        final = min(hard_cap, base * penalty + bonus)
        final = min(1.0, max(0.0, final))

        if penalty < 1.0:
            logger.debug(
                "[Layer2] penalty=%.2f cap=%.2f age_factor=%.2f | "
                "base=%.3f → final=%.3f | yr=%d | '%s'",
                penalty, hard_cap, age_factor, base, final,
                year, paper.title[:50],
            )
        return final


__all__ = ["SemanticPaperRanker"]
