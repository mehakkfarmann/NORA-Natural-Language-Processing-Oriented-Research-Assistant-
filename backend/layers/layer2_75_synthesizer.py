"""NORA — Layer 2.75: Relevance Gate + Gap Synthesis + Consensus Extraction"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# CONFIG
GAP_SIMILARITY_THRESHOLD = 0.82   # cosine threshold for gap dedup
MIN_PAPERS_OUT           = 3      # floor when all papers fail threshold
MAX_CONSENSUS_GAPS       = 3      # max latent consensus gaps to extract
MAX_GAPS_TOTAL           = 20     # hard cap on gaps passed to Layer 4

# Gap taxonomy — used to classify extracted gaps
GAP_TAXONOMY = [
    "missing_benchmark",
    "missing_dataset",
    "missing_framework",
    "method_limitation",
    "scalability_gap",
    "explainability_gap",
    "generalization_failure",
    "context_deficiency",
    "data_scarcity",
    "computational_cost",
    "unsolved_problem",
]


# PUBLIC ENTRY POINT

def synthesize_results(
    ranked_papers: List[Dict],
    original_query: str,
    smart_query: str,
    embed_model,
    groq_client=None,
    gaps: Optional[List[Dict]] = None,
    relevance_threshold: float = 0.35,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Gate papers by relevance, deduplicate gaps semantically,
    and extract latent consensus themes across papers.

    Returns:
        filtered_papers  — papers that passed threshold (or floor)
        enriched_gaps    — deduped + consensus gaps with canonical fields
        meta             — diagnostics dict
    """
    logger.info(
        "[Layer2.75] Starting | papers=%d | gaps=%d | threshold=%.2f",
        len(ranked_papers), len(gaps) if gaps else 0, relevance_threshold,
    )

    gaps = gaps or []

    filtered_papers = _score_and_filter(
        ranked_papers, original_query, embed_model, relevance_threshold
    )

    enriched = [_enrich_gap(g) for g in gaps]

    deduped = _dedup_gaps_semantic(enriched, embed_model)
    deduped = deduped[:MAX_GAPS_TOTAL]

    consensus_gaps: List[Dict] = []
    if groq_client and filtered_papers and deduped:
        consensus_gaps = _extract_consensus_gaps(
            papers=filtered_papers,
            existing_gaps=deduped,
            query=original_query,
            groq_client=groq_client,
        )
        logger.info("[Layer2.75] Consensus gaps extracted: %d", len(consensus_gaps))

    final_gaps = consensus_gaps + deduped

    meta = {
        "papers_before":        len(ranked_papers),
        "papers_after":         len(filtered_papers),
        "gaps_before_dedup":    len(enriched),
        "gaps_after_dedup":     len(deduped),
        "consensus_gaps":       len(consensus_gaps),
        "relevance_threshold":  relevance_threshold,
    }

    logger.info(
        "[Layer2.75] Done | papers=%d→%d | gaps=%d→%d | consensus=%d",
        len(ranked_papers), len(filtered_papers),
        len(enriched), len(deduped), len(consensus_gaps),
    )

    return filtered_papers, final_gaps, meta


# STEP 1 — PAPER RELEVANCE FILTER

def _score_and_filter(
    papers: List[Dict],
    query: str,
    embed_model,
    threshold: float,
) -> List[Dict]:
    """Score papers by cosine similarity to query, apply threshold with floor."""
    if not papers:
        return []

    try:
        q_vec = embed_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0].astype(np.float32)

        texts = []
        for p in papers:
            if hasattr(p, "title"):
                title = getattr(p, "title", "")
                abstract = getattr(p, "abstract", getattr(p, "summary", ""))
            else:
                title = p.get("title", "")
                abstract = p.get("summary", p.get("abstract", ""))
            texts.append(f"{title} {abstract[:300]}")

        vecs = embed_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

        scored = []
        for paper, vec in zip(papers, vecs):
            if hasattr(paper, "to_dict") and callable(paper.to_dict):
                p = paper.to_dict()
            elif hasattr(paper, "__dict__"):
                p = vars(paper).copy()
            else:
                p = dict(paper) if isinstance(paper, dict) else {}

            relevance = round(float(np.dot(q_vec, vec)), 3)
            p["original_query_relevance"] = relevance
            scored.append(p)

        scored.sort(key=lambda x: x["original_query_relevance"], reverse=True)
        filtered = [p for p in scored if p["original_query_relevance"] >= threshold]

        if not filtered:
            filtered = scored[:MIN_PAPERS_OUT]
            logger.info("[Layer2.75] Floor triggered — kept top %d", len(filtered))
        else:
            logger.info(
                "[Layer2.75] Threshold %.2f: kept %d/%d papers | score range: %.3f–%.3f",
                threshold, len(filtered), len(scored),
                scored[-1]["original_query_relevance"],
                scored[0]["original_query_relevance"],
            )

        return filtered

    except Exception as exc:
        logger.warning("[Layer2.75] Scoring failed (%s) — returning all papers", exc)
        result = []
        for p in papers:
            if hasattr(p, "to_dict") and callable(p.to_dict):
                result.append(p.to_dict())
            elif hasattr(p, "__dict__"):
                result.append(vars(p).copy())
            else:
                result.append(dict(p) if isinstance(p, dict) else {})
        return result

# STEP 2 — GAP ENRICHMENT (canonical fields)

_PROBLEM_TYPE_KEYWORDS = {
    "missing_benchmark":     ["benchmark", "evaluation protocol", "standard", "metric"],
    "missing_dataset":       ["dataset", "corpus", "data collection", "annotated data"],
    "missing_framework":     ["framework", "architecture", "system", "pipeline", "tool"],
    "method_limitation":     ["limitation", "fails", "cannot", "does not", "unable"],
    "scalability_gap":       ["scalab", "large-scale", "enterprise", "industrial"],
    "explainability_gap":    ["explainab", "interpretab", "transparent", "black-box"],
    "generalization_failure":["generaliz", "domain adaptation", "cross-domain", "transfer"],
    "context_deficiency":    ["context", "contextual", "adaptive", "dynamic"],
    "data_scarcity":         ["scarce", "limited data", "few-shot", "low-resource"],
    "computational_cost":    ["computational", "expensive", "overhead", "efficiency"],
}


def _enrich_gap(gap: Dict) -> Dict:
    """Add canonical structured fields to a gap dict."""
    g = dict(gap)
    desc = g.get("gap_description", "").lower()

    if not g.get("problem_type"):
        g["problem_type"] = _classify_gap_type(desc)

    if not g.get("missing_capability"):
        g["missing_capability"] = _extract_missing_capability(desc)

    if not g.get("constraint"):
        g["constraint"] = _extract_constraint(desc)

    if not g.get("existing_limitation"):
        g["existing_limitation"] = g.get("affected_methods", [])

    return g


def _classify_gap_type(desc: str) -> str:
    for gap_type, keywords in _PROBLEM_TYPE_KEYWORDS.items():
        if any(kw in desc for kw in keywords):
            return gap_type
    return "unsolved_problem"


def _extract_missing_capability(desc: str) -> str:
    """Extract what is missing from the gap description."""
    patterns = [
        r"no (\w[\w\s]{2,30}) (?:exists|available|has been)",
        r"lack(?:s|ing) (?:of )?(?:a |an )?(\w[\w\s]{2,30})",
        r"(?:absence|without) (?:of )?(?:a |an )?(\w[\w\s]{2,30})",
        r"(\w[\w\s]{2,30}) (?:is|are) (?:missing|absent|unavailable)",
    ]
    for pat in patterns:
        m = re.search(pat, desc)
        if m:
            return m.group(1).strip()[:60]
    return ""


def _extract_constraint(desc: str) -> str:
    """Extract the deployment/application context."""
    patterns = [
        r"(?:for|in|under|within) (enterprise[\w\s]{0,30})",
        r"(?:for|in|under|within) (industrial[\w\s]{0,30})",
        r"(?:for|in|under|within) ([\w\s]{2,25} (?:environment|setting|context|workflow|pipeline))",
    ]
    for pat in patterns:
        m = re.search(pat, desc)
        if m:
            return m.group(1).strip()[:60]
    return ""


# STEP 3 — SEMANTIC GAP DEDUP

def _dedup_gaps_semantic(gaps: List[Dict], embed_model) -> List[Dict]:
    """
    Collapse vocabulary-variant gaps into one canonical gap.
    Uses BGE cosine similarity — gaps above GAP_SIMILARITY_THRESHOLD
    are considered duplicates; highest-confidence one is kept.
    """
    if len(gaps) <= 1:
        return gaps

    try:
        texts = [g.get("gap_description", "") for g in gaps]
        vecs = embed_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
    except Exception as exc:
        logger.warning("[Layer2.75] Gap embedding failed (%s) — skipping dedup", exc)
        return gaps

    n = len(gaps)
    kept = [True] * n

    for i in range(n):
        if not kept[i]:
            continue
        for j in range(i + 1, n):
            if not kept[j]:
                continue
            sim = float(np.dot(vecs[i], vecs[j]))
            if sim >= GAP_SIMILARITY_THRESHOLD:
                ci = gaps[i].get("confidence", 0.5)
                cj = gaps[j].get("confidence", 0.5)
                if ci >= cj:
                    kept[j] = False
                else:
                    kept[i] = False
                    break

    result = [g for g, k in zip(gaps, kept) if k]
    logger.info("[Layer2.75] Gap dedup: %d → %d", n, len(result))
    return result


# STEP 4 — CONSENSUS SYNTHESIZER

_CONSENSUS_PROMPT = """You are a research synthesis expert.

Below are abstracts from {n_papers} papers and {n_gaps} extracted research gaps.

Query: {query}

Paper abstracts:
{abstracts}

Extracted gaps:
{gap_list}

Task: Identify {max_consensus} latent, cross-cutting research deficiencies that appear IMPLICITLY across MULTIPLE papers — even if expressed with different vocabulary.

These should be GENERALIZED field-level problems, not observations from one paper.

For each consensus gap, output a JSON object with:
{{
  "gap_description": "No [artifact] exists for [specific cross-cutting problem].",
  "gap_type": "<one of: {taxonomy}>",
  "problem_type": "<same taxonomy>",
  "missing_capability": "<what artifact is absent>",
  "constraint": "<deployment/application context>",
  "confidence": 0.75,
  "gap_tier": "consensus",
  "source": "consensus_synthesis",
  "evidence_quote": "<brief phrase capturing the implicit theme>",
  "affected_methods": []
}}

Return ONLY a JSON array of {max_consensus} objects. No explanation, no markdown.
Array:"""


def _extract_consensus_gaps(
    papers: List[Dict],
    existing_gaps: List[Dict],
    query: str,
    groq_client,
) -> List[Dict]:
    """
    Ask the LLM to find latent themes across papers that individual
    gap extraction may have missed. Returns consensus gap dicts.
    """
    abstracts = "\n\n".join(
       f"[{i+1}] {p.get('title','')}: {p.get('abstract', p.get('summary', ''))[:300]}"
       for i, p in enumerate(papers[:8])
    )
    gap_list = "\n".join(
        f"- {g.get('gap_description','')}"
        for g in existing_gaps[:10]
    )
    taxonomy_str = ", ".join(GAP_TAXONOMY)

    prompt = _CONSENSUS_PROMPT.format(
        n_papers=min(len(papers), 8),
        n_gaps=min(len(existing_gaps), 10),
        query=query,
        abstracts=abstracts,
        gap_list=gap_list,
        max_consensus=MAX_CONSENSUS_GAPS,
        taxonomy=taxonomy_str,
    )

    try:
        raw = groq_client.generate(prompt=prompt, temperature=0.3, max_tokens=800)
        cleaned = _clean_json(raw)
        items = json.loads(cleaned)
        if not isinstance(items, list):
            raise ValueError("not a list")

        result = []
        for item in items[:MAX_CONSENSUS_GAPS]:
            if not isinstance(item, dict):
                continue
            if not item.get("gap_description", "").strip():
                continue
            item.setdefault("gap_tier", "consensus")
            item.setdefault("source", "consensus_synthesis")
            item.setdefault("confidence", 0.75)
            item.setdefault("affected_methods", [])
            result.append(item)

        return result

    except Exception as exc:
        logger.warning("[Layer2.75] Consensus extraction failed (%s)", exc)
        return []


# UTIL

def _clean_json(text: str) -> str:
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else parts[0]
        if text.startswith("json"):
            text = text[4:]
    for ch, end in [("[", "]"), ("{", "}")]:
        idx = text.find(ch)
        if idx != -1:
            last = text.rfind(end)
            if last > idx:
                return text[idx:last + 1]
    return text.strip()
