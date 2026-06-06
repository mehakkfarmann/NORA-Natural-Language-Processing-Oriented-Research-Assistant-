from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MIN_QUOTE_LEN = 20
_WORD_OVERLAP_THRESHOLD    = 0.55
_TRIGRAM_OVERLAP_THRESHOLD = 0.25

_GAP_SECTIONS = [
    "limitations", "future_work", "discussion", "conclusion", "related_work", "results",
]

_SIGNAL_PHRASES = [
    r"in future work", r"future work (will|should|could|may|might|includes?)",
    r"we (did not|do not|have not|could not|cannot)", r"we (plan to|intend to|aim to|will|hope to)",
    r"(is|are|remains?) (an? )?(open|unsolved|challenging|difficult)",
    r"(limitation|limitations?) (of|is|are|include)",
    r"(not|never) (evaluated?|tested?|compared?|validated?|benchmarked?)",
    r"(left|deferred?) (for|to) future", r"(lack[s]?|without|no) (benchmark|baseline|comparison|evaluation)",
    r"does not (scale|generali[sz]e|handle|support|address)",
    r"(this study|this paper|this work|our approach|our method) (did not|does not|was not)",
    r"we are interested to", r"(further|more) (research|investigation|work|study) (is|are|needed|required)",
]
_SIGNAL_RE = re.compile("|".join(_SIGNAL_PHRASES), re.IGNORECASE)

CONTRIBUTION_BLOCK = re.compile(
    r"^(we|this paper|this work)\s+"
    r"(propose|present|introduce|develop|build|design|implement|demonstrate|show)",
    re.IGNORECASE,
)

_NOVELTY_CLAIM_RE = re.compile(
    r"("
    r"(did|could|can|do)\s+not\s+find\s+any\s+(similar|comparable|existing|related)"
    r"|to\s+the\s+best\s+of\s+our\s+knowledge"
    r"|first\s+(study|paper|approach|method|work)\s+to\s+"
    r"(measure|propose|introduce|present|apply|use|develop)"
    r"|no\s+(similar|comparable|existing|related)\s+"
    r"(approach|method|study|work|framework|technique|tool|system|paper)"
    r"|no\s+existing\s+(method|approach|work|framework|technique|study)"
    r"|(novel|unique|original)\s+(contribution|approach|method|framework|study)"
    r")",
    re.IGNORECASE,
)

_VALID_CATEGORIES = {"explicit_limitation", "future_work", "open_problem", "missing_evaluation"}

def _assign_gap_axis(gap: Dict) -> str:
    desc = ((gap.get("gap_description") or "") + " " + (gap.get("gap_statement") or "")).lower()
    if any(kw in desc for kw in ["dataset", "data", "annotation", "label", "corpus", "sample", "scarcity"]): return "data"
    if any(kw in desc for kw in ["evaluat", "metric", "benchmark", "baseline", "comparison", "auc", "f1", "accuracy"]): return "evaluation"
    if any(kw in desc for kw in ["deploy", "production", "real-world", "clinical", "industrial", "latency", "efficiency", "resource", "computational"]): return "deployment"
    if any(kw in desc for kw in ["robust", "noise", "adversarial", "out-of-distribution", "domain shift", "generaliz"]): return "robustness"
    if any(kw in desc for kw in ["fair", "bias", "ethic", "privacy", "demographic"]): return "ethical/fairness"
    return "model"

def extract_gaps(papers: list, groq_client, model: str, domain_context: Optional[Dict] = None, embed_model=None) -> List[Dict]:
    if not papers: return []
    ctx        = domain_context or {}
    user_query = ctx.get("original_query") or ctx.get("final_query", "research topic")

    print(f"\n{'='*55}")
    print(f"[GAP EXTRACTOR] Starting with {len(papers)} papers")
    print(f"[GAP EXTRACTOR] Mode: FAITHFUL EXTRACTION (No Dedup Collapse)")
    print(f"{'='*55}")

    all_gaps: List[Dict] = []
    for i, paper in enumerate(papers):
        paper_index = i + 1
        def _get(key, default=""):
            return paper.get(key, default) if isinstance(paper, dict) else getattr(paper, key, default)

        paper_id  = _get("paper_id",  f"paper_{i}")
        title     = _get("title",     "Untitled")
        year      = _get("year",      "?")
        abstract  = (_get("abstract", "") or "").strip()
        sections  = _get("sections",  None) or {}
        full_text = _get("full_text", "") or ""

        print(f"\n[GAP EXTRACTOR] Paper {paper_index}: '{title[:60]}'")
        extraction_text, text_source = _build_extraction_text(abstract, sections, full_text)
        if len(extraction_text) < 80:
            print(f"  → Skipped: not enough text ({len(extraction_text)} chars)")
            continue

        raw_gaps = _extract_author_statements(paper_index, title, year, extraction_text, user_query, groq_client, model, "full_text" in text_source)
        corpus = _build_validation_corpus(abstract, sections, full_text)
        validated_gaps = _validate_quotes(raw_gaps, corpus)
        genuine_gaps = _filter_novelty_claims(validated_gaps)
        constrained_gaps = _apply_per_paper_constraints(genuine_gaps)

        for gap in constrained_gaps:
            formatted = _format_gap(gap, paper, paper_id, paper_index)
            all_gaps.append(formatted)

    # STEP 2: Removed collapse in dedup - handled in Layer 4
    all_gaps = _dedup_gaps(all_gaps)

    _CATEGORY_RANK = {"open_problem": 0, "explicit_limitation": 1, "future_work": 2, "missing_evaluation": 3}
    all_gaps.sort(key=lambda g: (_CATEGORY_RANK.get(g.get("category", "future_work"), 9), -g.get("extraction_confidence", 0)))

    print(f"\n[GAP EXTRACTOR] Final: {len(all_gaps)} validated gaps (All preserved)")
    print(f"{'='*55}\n")
    return all_gaps

def _apply_per_paper_constraints(gaps: List[Dict]) -> List[Dict]:
    if not gaps: return []
    for g in gaps: g["_axis"] = _assign_gap_axis(g)
    
    eval_gaps = [g for g in gaps if g.get("category") == "missing_evaluation" or g.get("_axis") == "evaluation"]
    if len(eval_gaps) > 1:
        eval_gaps.sort(key=lambda x: x.get("extraction_confidence", 0), reverse=True)
        kept_eval = [eval_gaps[0]]
        metrics_seen = set(re.findall(r"\b(accuracy|f1|precision|recall|auc|dice|iou|mae|rmse|bleu|rouge)\b", eval_gaps[0].get("gap_description", "").lower()))
        for g in eval_gaps[1:]:
            g_metrics = set(re.findall(r"\b(accuracy|f1|precision|recall|auc|dice|iou|mae|rmse|bleu|rouge)\b", g.get("gap_description", "").lower()))
            if g_metrics and not g_metrics.intersection(metrics_seen):
                kept_eval.append(g)
                metrics_seen.update(g_metrics)
            if len(kept_eval) >= 2: break
        gaps = [g for g in gaps if g not in eval_gaps] + kept_eval

    axis_counts = {}
    constrained_gaps = []
    gaps.sort(key=lambda x: x.get("extraction_confidence", 0), reverse=True)
    for g in gaps:
        axis = g["_axis"]
        if axis_counts.get(axis, 0) < 2:
            constrained_gaps.append(g)
            axis_counts[axis] = axis_counts.get(axis, 0) + 1
    return constrained_gaps

def _build_extraction_text(abstract: str, sections: Dict[str, str], full_text: str) -> Tuple[str, str]:
    parts, found_sections = [], []
    for sec in _GAP_SECTIONS:
        text = (sections.get(sec) or "").strip()
        if text:
            parts.append(f"[{sec.upper()}]\n{text[:4000]}")
            found_sections.append(sec)
    if abstract: parts.insert(0, f"[ABSTRACT]\n{abstract[:800]}")
    if parts:
        extra_signals = _extract_signal_sentences(full_text, already_have=parts)
        if extra_signals: parts.append(f"[ADDITIONAL SIGNAL SENTENCES FROM FULL TEXT]\n{extra_signals}")
        return "\n\n".join(parts)[:20000], f"sections({','.join(found_sections)})"
    if full_text and len(full_text) >= 300:
        signals = _extract_signal_sentences(full_text, already_have=[])
        header  = f"[ABSTRACT]\n{abstract[:800]}\n\n" if abstract else ""
        if signals: return header + f"[SIGNAL SENTENCES (future work / limitations)]\n{signals}\n\n" + f"[FULL TEXT]\n{full_text[:15000]}", "full_text+signals"
        return header + f"[FULL TEXT]\n{full_text[:18000]}", "full_text"
    if abstract: return f"[ABSTRACT]\n{abstract}", "abstract_only"
    return "", "none"

def _extract_signal_sentences(full_text: str, already_have: list) -> str:
    if not full_text: return ""
    existing_text = " ".join(already_have).lower()
    sentences = re.split(r"(?<=[.!?])\s+", full_text)
    found, unique, seen = [], [], []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 30 or len(sent) > 400 or not _SIGNAL_RE.search(sent): continue
        sent_lower = sent.lower()
        key_words  = set(re.findall(r"\b[a-zA-Z]{5,}\b", sent_lower))
        if key_words and sum(1 for w in key_words if w in existing_text) / len(key_words) > 0.70: continue
        found.append(sent)
    for sent in found:
        words = set(re.findall(r"\b[a-zA-Z]{4,}\b", sent.lower()))
        if not any(len(words & s) / max(len(words), len(s)) > 0.70 for s in seen if s):
            unique.append(sent)
            seen.append(words)
    return "\n".join(f"- {s}" for s in unique[:20])

def _extract_author_statements(paper_index: int, title: str, year, extraction_text: str, user_query: str, groq_client, model: str, pdf_mode: bool = False) -> List[Dict]:
    EXCLUSION_BLOCK = """
=== DO NOT EXTRACT (hard exclusions) ===
Reject ANY sentence that matches one of these patterns — they are not research gaps:
CONTRIBUTION STATEMENTS (starts with any of these verbs):
  - "We propose", "We present", "We introduce", "We demonstrate", "We develop", "We build", "We design", "We implement", "We achieve", "We show that", "In this paper we", "This paper presents", "This paper proposes", "This work introduces"
BACKGROUND / FIELD DESCRIPTIONS:
  - General statements about the field that name no specific limit of THIS paper.
GENERIC OBSERVATIONS (no specific limitation attached):
  - "More work is needed." / "Deep learning is a growing area."
"""
    REFINEMENT_INSTRUCTION = """
=== HOW TO WRITE gap_statement ===
Convert the raw sentence into a NEUTRAL limitation description. 
CRITICAL RULES FOR gap_statement:
1. DO NOT propose improvements, methods, or solutions.
2. DO NOT use questions (e.g., "How can we...", "What approach...").
3. DO NOT use solution framing (e.g., "we propose...", "requires a new method...").
4. ONLY describe the limitation, missing element, or failure in academic neutral form.
Keep gap_statement between 1–2 sentences. No bullet points.
"""
    prompt_base = f"""You are a research gap extractor for the NORA system.
Paper: [{paper_index}] {title} ({year})
Research topic: {user_query}
--- PAPER TEXT ---
{extraction_text[:16000]}
--- END TEXT ---
{EXCLUSION_BLOCK}
{REFINEMENT_INSTRUCTION}
Return ONLY a JSON array. No explanation. No markdown. No preamble.
[
  {{
    "category": "explicit_limitation|future_work|open_problem|missing_evaluation",
    "evidence_quote": "The EXACT sentence copied verbatim from the paper text",
    "gap_statement": "Your neutral limitation description (1-2 sentences)",
    "section": "abstract|introduction|related_work|methodology|results|conclusion|limitations|future_work|discussion|unknown",
    "task_relevant_baselines": "Author (year) if passing the task gate, else: none identified",
    "method_summary": "Completed template with actual values from the paper"
  }}
]"""
    try:
        raw = groq_client.generate(prompt=prompt_base, model=model, temperature=0.1, max_tokens=3000)
    except Exception as exc:
        logger.warning("[GapExtractor] LLM call failed for paper %d: %s", paper_index, exc)
        return []
    return _parse_json(raw)

def _build_validation_corpus(abstract: str, sections: Dict[str, str], full_text: str) -> List[str]:
    def _norm_ws(t: str) -> str: return re.sub(r"  +", " ", t).strip() if t else ""
    corpus = []
    if abstract: corpus.append(_norm_ws(abstract))
    for text in sections.values():
        if text: corpus.append(_norm_ws(text))
    if full_text: corpus.append(_norm_ws(full_text[:20000]))
    return corpus

_VERB_PATTERNS = re.compile(r"\b(is|are|was|were|will|would|should|could|have|has|had|do|does|did|plan|intend|aim|investigate|measure|compare|evaluate|extend|test|develop|propose|address|remain|include|require|need|found|show|identify|indicate|suggest|demonstrate|note|report|state|claim|not\s+\w+|did\s+not|does\s+not|cannot|could\s+not|interested\s+to|we\s+are|we\s+were|we\s+will|we\s+did)\b", re.IGNORECASE)

def _is_complete_sentence(quote: str) -> bool:
    q = quote.strip()
    if not q or len(q) < _MIN_QUOTE_LEN: return False
    return bool(_VERB_PATTERNS.search(q))

def _validate_quotes(raw_gaps: List[Dict], corpus: List[str]) -> List[Dict]:
    if not corpus: return [g | {"_validated": False} for g in raw_gaps]
    _LIMITATION_KEYWORDS = frozenset(["limited", "cannot", "restricted", "only", "small-scale", "not generalizable", "did not", "does not", "was not", "could not", "unable", "constraint", "limitation"])
    accepted = []
    for gap in raw_gaps:
        quote = (gap.get("evidence_quote") or "").strip()
        if CONTRIBUTION_BLOCK.search(quote): continue
        section = (gap.get("section") or "").strip().lower()
        if section in ("methodology", "introduction"):
            if not any(kw in quote.lower() for kw in _LIMITATION_KEYWORDS): continue
        if not _is_complete_sentence(quote): continue
        if len(quote) < _MIN_QUOTE_LEN:
            accepted.append(gap | {"_validated": False})
            continue
        q_words, q_tris = _keywords(quote), _trigrams(quote)
        if not q_words:
            accepted.append(gap | {"_validated": False})
            continue
        best_word, best_tri = 0.0, 0.0
        for chunk in corpus:
            c_words, c_tris = _keywords(chunk), _trigrams(chunk)
            if c_words and q_words: best_word = max(best_word, len(q_words & c_words) / len(q_words))
            if c_tris and q_tris: best_tri = max(best_tri, len(q_tris & c_tris) / len(q_tris))
        if best_word >= _WORD_OVERLAP_THRESHOLD or best_tri >= _TRIGRAM_OVERLAP_THRESHOLD:
            accepted.append(gap | {"_validated": True, "_word_overlap": round(best_word, 2)})
    return accepted

def _filter_novelty_claims(gaps: List[Dict]) -> List[Dict]:
    kept = []
    for gap in gaps:
        quote = (gap.get("evidence_quote") or "").strip()
        if CONTRIBUTION_BLOCK.search(quote) or _NOVELTY_CLAIM_RE.search(quote): continue
        kept.append(gap)
    return kept

def _get_paper_title(paper) -> str:
    if isinstance(paper, dict):
        return paper.get("title", "Untitled")
    return getattr(paper, "title", "Untitled")

def _format_gap(raw: Dict, paper, paper_id: str, paper_index: int) -> Dict:
    category = raw.get("category", "explicit_limitation")
    if category not in _VALID_CATEGORIES: category = "explicit_limitation"
    quote   = (raw.get("evidence_quote") or "").strip()
    refined = (raw.get("gap_statement")  or "").strip()
    section = (raw.get("section") or "unknown").strip()
    desc = refined if refined else quote
    validated = raw.get("_validated", False)
    word_overlap = raw.get("_word_overlap", 0.0)
    base_conf = 0.75 if validated else 0.55
    if section in ("limitations", "future_work"): base_conf = min(0.90, base_conf + 0.10)
    elif section in ("introduction", "related_work"): base_conf = min(0.85, base_conf + 0.05)
    
    _sig = 0.40
    if section in ("limitations", "future_work"): _sig += 0.20
    elif section in ("discussion", "conclusion"): _sig += 0.15
    if validated: _sig += 0.15
    if category in ("explicit_limitation", "open_problem"): _sig += 0.10
    elif category == "missing_evaluation": _sig += 0.05
    research_significance = round(min(1.0, _sig), 2)

    _CATEGORY_TO_TYPE = {"explicit_limitation": "methodological_gap", "future_work": "future_work", "open_problem": "field_gap", "missing_evaluation": "evaluation_gap"}
    gap_type = _CATEGORY_TO_TYPE.get(category, "methodological_gap")

    _q = quote.lower() + " " + desc.lower()

    # STEP 1: ADD research_axis FIELD
    if any(kw in _q for kw in ("fairness", "bias", "demographic", "inequity")):
        research_axis = "fairness"
    elif any(kw in _q for kw in ("auc", "metric", "benchmark", "evaluation", "compare")):
        research_axis = "evaluation"
    elif any(kw in _q for kw in ("generaliz", "domain shift", "cross-domain", "out-of-distribution")):
        research_axis = "generalization"
    elif any(kw in _q for kw in ("efficiency", "latency", "runtime", "compute")):
        research_axis = "efficiency"
    elif any(kw in _q for kw in ("explain", "interpret", "feature")):
        research_axis = "interpretability"
    else:
        research_axis = "general"

    if any(kw in _q for kw in ("dataset", "annotated", "labeled", "training data", "benchmark data", "corpus", "data scarcity", "data imbalance", "class imbalance")): gap_type = "dataset_gap"
    elif any(kw in _q for kw in ("generali", "cross-domain", "domain shift", "out-of-distribution", "unseen domain", "transfer", "different dataset", "new dataset", "low-resource", "external validity")): gap_type = "generalization_gap"
    elif any(kw in _q for kw in ("computational", "scalab", "efficiency", "latency", "throughput", "memory", "parameter", "inference time", "real-time", "runtime", "resource", "cost")): gap_type = "efficiency_gap"
    elif any(kw in _q for kw in ("deploy", "production", "real-world application", "clinical", "industrial", "practitioner", "adoption", "integration", "system")): gap_type = "deployment_gap"
    elif any(kw in _q for kw in ("uncertainty", "calibrat", "confidence", "reliability", "robust", "noise", "adversarial", "out-of-distribution detection", "epistemic")): gap_type = "uncertainty_gap"
    elif category == "missing_evaluation" or any(kw in _q for kw in ("evaluat", "benchmark", "metric", "baseline", "comparison", "ablation", "experiment", "ground truth")): gap_type = "evaluation_gap"

    return {
        "gap_description": desc, "gap_statement": refined, "gap_type": gap_type,
        "evidence_quote": quote, "section": section, "category": category,
        "paper_title": _get_paper_title(paper),
        "axis": raw.get("_axis", _assign_gap_axis(raw)),
        "research_axis": research_axis,
        "supporting_paper_ids": [paper_id], "source_paper_id": paper_id,
        "evidence_count": 1, "cross_paper_support": 0, "is_grounded": validated,
        "extraction_confidence": round(base_conf, 2), "research_significance": research_significance,
        "gap_quality": "grounded" if validated else "weak", "is_fallback": False, "gap_source": "faithful_extraction",
        "grounding_evidence": [{"paper_index": paper_index, "type": category, "text": quote, "confidence": round(base_conf, 2), "detector": "faithful"}],
        "supporting_paper_indices": [paper_index],
        "supporting_citations": paper.get("citations", 0) if isinstance(paper, dict) else getattr(paper, "citations", 0),
        "domain_alignment": round(base_conf, 2), "contradictions": [], "distinct_methods": [], "consequence": "",
        "confidence_breakdown": {"base": base_conf, "validated": validated, "word_overlap": word_overlap, "section": section},
        "task_relevant_baselines": (raw.get("task_relevant_baselines") or "none identified").strip(),
        "method_summary": (raw.get("method_summary") or "").strip(),
    }

# STEP 2: REMOVE COLLAPSE IN DEDUP
def _dedup_gaps(gaps: List[Dict]) -> List[Dict]:
    """
    No similarity-based merging or exact deduplication here.
    We handle grouping and diversity in Layer 4 instead to preserve all valid gaps.
    """
    return gaps

def _keywords(text: str) -> set: return set(re.findall(r"\b[a-zA-Z]{4,}\b", text.lower()))
def _trigrams(text: str) -> set:
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {" ".join(words[i:i+3]) for i in range(len(words) - 2)} if len(words) >= 3 else set()

def _parse_json(raw: str) -> List[Dict]:
    if not raw: return []
    text = raw.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith("["): text = part; break
    start = text.find("[")
    if start == -1: return []
    depth, end = 0, -1
    for i, ch in enumerate(text[start:], start=start):
        if ch == "[": depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0: end = i; break
    if end == -1: return []
    try:
        parsed = json.loads(text[start:end + 1])
        if not isinstance(parsed, list): return []
        return [item for item in parsed if isinstance(item, dict) and len(item.get("evidence_quote", "")) >= _MIN_QUOTE_LEN and len(item.get("gap_statement", "")) >= 20]
    except (json.JSONDecodeError, ValueError): return []

class GapExtractor:
    def __init__(self, groq_client, model: str, domain_context: Optional[Dict] = None):
        self.groq_client, self.model, self.domain_context = groq_client, model, domain_context or {}
    def process(self, papers: list) -> List[Dict]: return extract_gaps(papers, self.groq_client, self.model, self.domain_context)
    def extract_papers(self, papers: list) -> List[Dict]: return self.process(papers)

extract_research_gaps = extract_gaps