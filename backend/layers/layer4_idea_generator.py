from __future__ import annotations
import json
import logging
import re
from typing import Dict, List, Optional
import numpy as np
from backend.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)
MAX_IDEAS_OUT = 8
MIN_PROPOSAL_STRENGTH = 0.35
MIN_FEASIBILITY = 0.40

_SATURATED_METHODS = frozenset([
    "u-net", "unet", "resnet", "vgg", "alexnet", "googlenet", "inception", "densenet", "mobilenet", "fcn", "squeezenet", "shufflenet",
    "svm", "random forest", "k-nn", "knn", "decision tree", "naive bayes", "logistic regression", "linear regression", "gradient boosting", "xgboost",
    "cnn", "lstm", "rnn", "gru", "mlp", "bert", "gpt", "transformer", "gan", "generative adversarial", "diffusion model", "gnn", "graph neural",
    "data augmentation", "improve accuracy", "build framework", "evaluation framework", "improve performance",
])
_GENERIC_METHOD_PHRASES = ["adaptive framework", "context-aware attention", "hierarchical token mixer", "multi-scale semantic encoder", "fusion network", "dynamic routing", "self-supervised representation learning", "novel attention mechanism"]
_VAGUE_GAP_SIGNALS = frozenset(["improve accuracy", "improve performance", "better results", "more experiments", "future work needed", "further research", "use deep learning", "use machine learning", "apply ai", "build a framework", "build an evaluation", "more data needed"])

_THREE_STAGE_PROMPT = """You are building a structured research proposal from an author-stated gap.
============================================================
DOMAIN LOCK — READ THIS BEFORE ANYTHING ELSE
USER QUERY   : {query}
RESEARCH DOMAIN: {domain}
Every part of your proposal MUST remain strictly inside this domain and query topic.

PAPER CONTEXT:
{paper_context}
AUTHOR-STATED GAP:
{gap_desc}
WHAT THE AUTHOR WROTE:
{evidence_quote}
ADDITIONAL AUTHOR EVIDENCE:
{grounding_evidence}
GAP TYPE: {gap_type}
RESEARCH FOCUS CONSTRAINT: {focus_constraint}
{forbidden_note}

YOU MUST FOLLOW DIVERSITY RULE:

1. Each research_axis can produce ONLY ONE idea unless explicitly required.
2. Do NOT generate multiple fairness-based ideas unless gap is fairness-related.
3. Every idea MUST come from a different research_axis when possible.
4. Never default evaluation/generalization/efficiency gaps into fairness methods.
5. Preserve axis diversity in final output.

YOUR TASK: Produce a 3-stage research proposal grounded in what the author wrote.
STAGE 1 — PROBLEM STATEMENT: One clear sentence starting with "How can..." or "What approach..." naming the domain.
STAGE 2 — METHOD DIRECTION: Name the specific technique. ONE named method only. Do NOT propose saturated methods (U-Net, ResNet, generic Transformer, etc.).
STAGE 3 — IMPLEMENTATION OUTLINE: Concrete details (input, output, dataset, baseline, metric).

ALSO PROVIDE:
title: 8-12 words describing this specific proposal.
novelty_category: "incremental_extension" | "moderate_novelty" | "high_novelty" | "potentially_saturated"
innovation_type: ONE of "architectural" | "training_objective" | "inference" | "evaluation" | "data-centric"
feasibility: float 0.0-1.0
why_feasible_now: one sentence concrete reason.
why_novel: one sentence naming the specific mechanism.
failure_mode_addressed: exact failure case from the paper.
causal_trace: {{ "why_this_method": "PART1 [specific artefact] because PART2 [why it makes this method natural]", "builds_on": "specific formula/metric", "limitation_solved": "exact limitation" }}

Return ONLY a JSON object. No markdown.
{{
  "title": "...",
  "proposal_problem": "Stage 1...",
  "proposal_method": "Stage 2...",
  "proposal_implement": {{ "input": "...", "output": "...", "dataset": "...", "baseline": "...", "metric": "..." }},
  "novelty_category": "...",
  "innovation_type": "...",
  "feasibility": 0.0,
  "why_feasible_now": "...",
  "why_novel": "...",
  "failure_mode_addressed": "...",
  "causal_trace": {{ "why_this_method": "...", "builds_on": "...", "limitation_solved": "..." }}
}}
"""

_NOVELTY_SCORE_MAP = {"high_novelty": 0.85, "moderate_novelty": 0.65, "incremental_extension": 0.40, "potentially_saturated": 0.20}

def _score_proposal_strength(idea_text: str, gap_vec, embed_model, lit_vecs=None) -> float:
    if not idea_text or embed_model is None: return 0.55
    try:
        iv = embed_model.encode([idea_text], normalize_embeddings=True)[0].astype(np.float32)
        if gap_vec is not None:
            gap_sim = float(np.dot(gap_vec.astype(np.float32), iv))
            return round(max(0.1, min(1.0, 1.0 - abs(gap_sim - 0.70) * 1.8)), 3)
        return 0.55
    except Exception: return 0.55

def _extract_json_object(raw: str) -> Optional[Dict]:
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except json.JSONDecodeError: pass
    start = raw.find("{")
    if start == -1: return None
    depth, end = 0, -1
    for i, ch in enumerate(raw[start:], start=start):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: end = i; break
    if end == -1: return None
    try: return json.loads(raw[start:end + 1])
    except json.JSONDecodeError: return None

def _format_grounding_evidence(gap: Dict) -> str:
    grounding = gap.get("grounding_evidence", [])
    if not grounding: return gap.get("evidence_quote", "") or "No explicit author statements available."
    return "\n".join(f"[Paper {ev.get('paper_index', '?')}][{ev.get('type', 'unknown')}] (conf={ev.get('confidence', 0.5):.1f}): {ev.get('text', '')}" for ev in grounding[:5])

def _reject_generic_pre_filter(gap_desc: str) -> bool:
    if not gap_desc: return False
    lower = gap_desc.lower()
    hit_count = sum(1 for sig in _VAGUE_GAP_SIGNALS if sig in lower)
    if hit_count >= 3: return True
    if len(gap_desc.split()) < 15 and hit_count >= 1: return True
    return False

def _penalise_template_reuse(idea: Dict, accepted_ideas: List[Dict], embed_model, penalty_threshold: float = 0.92) -> float:
    if not accepted_ideas or embed_model is None: return 1.0
    new_text = idea.get("proposal_method", "")
    if not new_text: return 1.0
    try:
        new_vec = embed_model.encode([new_text], normalize_embeddings=True)[0].astype(np.float32)
        for prev in accepted_ideas:
            prev_text = prev.get("proposal_method", "")
            if not prev_text: continue
            prev_vec = embed_model.encode([prev_text], normalize_embeddings=True)[0].astype(np.float32)
            if float(np.dot(new_vec, prev_vec)) >= penalty_threshold: return 0.0
    except Exception: pass
    return 1.0

def _map_approach_track(innovation_type: str) -> str:
    if innovation_type in ("architectural", "training_objective", "inference"):
        return "foundational"
    if innovation_type in ("data-centric", "evaluation"):
        return "applied"
    return "hybrid"

def _is_generic_method_semantic(method_text: str, embed_model) -> bool:
    lower_text = method_text.lower()
    if any(sm in lower_text for sm in _SATURATED_METHODS): return True
    if embed_model and len(method_text.split()) > 2:
        try:
            method_vec = embed_model.encode([method_text], normalize_embeddings=True)[0].astype(np.float32)
            generic_vecs = embed_model.encode(_GENERIC_METHOD_PHRASES, normalize_embeddings=True).astype(np.float32)
            if max(float(np.dot(method_vec, gv)) for gv in generic_vecs) > 0.78: return True
        except Exception: pass
    return False

def _call_llm(gap: Dict, query: str, domain: str, groq: LLMClient, mock_mode: bool, papers_context: Optional[List[Dict]] = None) -> Optional[Dict]:
    if mock_mode:
        return {
            "title": "Mock Proposal", "proposal_problem": "How can we improve X?", "proposal_method": "Method Y.",
            "proposal_implement": {"input": "data", "output": "result", "dataset": "DS", "baseline": "B", "metric": "M"},
            "novelty_category": "moderate_novelty", "feasibility": 0.70, "why_feasible_now": "Data exists.",
            "causal_trace": {"why_this_method": "Extends X because Y", "builds_on": "Eq 1", "limitation_solved": "Lim"},
        }

    gap_type = gap.get("gap_type", "")
    gap_category = gap.get("category", "")
    if gap_category == "future_work" and gap_type == "methodological_gap": gap_type = "future_work"
    elif gap_category == "missing_evaluation" and gap_type == "methodological_gap": gap_type = "evaluation_gap"

    forbidden_map = {"evaluation_gap": (["GNN", "Transformer", "RL"], "Focus on evaluation frameworks."), "missing_evaluation": (["GNN", "Transformer", "RL"], "Focus on evaluation frameworks.")}
    forbidden_entry = forbidden_map.get(gap_type, ([], "Mechanistic justification required."))
    forbidden_note = f"Forbidden methods: {forbidden_entry[0]}. {forbidden_entry[1]}" if forbidden_entry[0] else "No forbidden methods."

    if query and " — focus: " in query: focus_constraint = query.split(" — focus: ", 1)[1].strip()
    elif domain and domain.lower() not in ("general", "unknown", ""): focus_constraint = f"Proposals must remain within {domain}, directly addressing: {query}"
    elif query: focus_constraint = f"Proposals must directly address the topic: {query}"
    else: focus_constraint = "No specific constraint."

    paper_context = "Not available."
    if papers_context:
        indices = gap.get("supporting_paper_indices") or []
        snippets = []
        for pctx in papers_context:
            if pctx.get("paper_index") in indices or not indices:
                title = pctx.get("title", "")
                abstract = (pctx.get("raw_abstract") or "")[:3000]
                if title or abstract: snippets.append(f"Title: {title}\nPaper Content:\n{abstract}")
            if len(snippets) >= 2: break
        if snippets: paper_context = "\n\n---\n\n".join(snippets)

    prompt = _THREE_STAGE_PROMPT.format(
        query=query or "Not specified", domain=domain or "Not specified", paper_context=paper_context,
        gap_desc=gap.get("gap_description", ""), evidence_quote=(gap.get("evidence_quote") or "").strip(),
        grounding_evidence=_format_grounding_evidence(gap), gap_type=gap_type,
        focus_constraint=focus_constraint, forbidden_note=forbidden_note
    )

    try:
        raw = groq.generate(prompt=prompt, temperature=0.2, max_tokens=1400)
        obj = _extract_json_object(raw)
        if not obj or not all(obj.get(f) for f in ("proposal_problem", "proposal_method", "proposal_implement")): return None
        
        # FIX: Sanitize lists to strings to prevent downstream join() crashes
        def _safe_str(val):
            if val is None: return ""
            if isinstance(val, list): return ", ".join(str(v) for v in val)
            if isinstance(val, dict): return ", ".join(f"{k}: {v}" for k, v in val.items())
            return str(val)

        impl = obj.get("proposal_implement", {})
        obj["proposal_implement"] = {k: _safe_str(v) for k, v in impl.items()}
        obj["feasibility"] = max(0.0, min(1.0, float(obj.get("feasibility", 0.5))))
        return obj
    except Exception as e:
        logger.warning("[Layer4] LLM parse failed: %s", e)
        return None

def run_layer4(gaps: List[Dict], query: str, aspects: List[str], domain: str, groq: LLMClient, embed_model, mock_mode: bool = False, literature_vecs: Optional[List] = None, paper_evidences: Optional[List[Dict]] = None) -> List[Dict]:
    print(f"\n{'='*60}")
    print(f"[LAYER 4] STARTING — V8.2 (Diversity Enforced, Math-Fixed)")
    if not gaps: return []

    grounded_gaps = [g for g in gaps if g.get("is_grounded")]
    ungrounded_gaps = [g for g in gaps if not g.get("is_grounded")]
    ordered_gaps = grounded_gaps + ungrounded_gaps
    all_ideas = []

    for gap_idx, gap in enumerate(ordered_gaps):
        gap_desc = gap.get("gap_description", "")
        if _reject_generic_pre_filter(gap_desc): continue
        
        gap_vec = None
        if embed_model and gap_desc:
            try: gap_vec = embed_model.encode([gap_desc], normalize_embeddings=True)[0]
            except Exception: pass

        raw = _call_llm(gap, query, domain, groq, mock_mode, papers_context=paper_evidences)
        if not raw or not raw.get("causal_trace", {}).get("why_this_method"): continue

        feasibility = float(raw.get("feasibility", 0.5))
        raw["feasibility"] = max(0.0, min(1.0, feasibility))
        if raw["feasibility"] < MIN_FEASIBILITY: continue

        impl = raw.get("proposal_implement", {})
        
        # Defensive string conversion
        def _to_str(v): return ", ".join(str(i) for i in v) if isinstance(v, list) else (str(v) if v is not None else "")

        full_idea_text = " ".join([
            _to_str(raw.get("proposal_problem", "")), _to_str(raw.get("proposal_method", "")),
            _to_str(impl.get("input", "")), _to_str(impl.get("output", "")), _to_str(impl.get("metric", "")),
        ])

        proposal_strength = _score_proposal_strength(full_idea_text, gap_vec, embed_model, literature_vecs)
        raw["proposal_strength"] = proposal_strength
        raw["novelty_score"] = _NOVELTY_SCORE_MAP.get(raw.get("novelty_category", ""), 0.50)
        raw["approach_track"] = _map_approach_track(raw.get("innovation_type", ""))

        if _is_generic_method_semantic(raw.get("proposal_method", ""), embed_model):
            raw["novelty_category"] = "potentially_saturated"
            proposal_strength = round(min(proposal_strength, 0.50), 3)
            raw["proposal_strength"] = proposal_strength
            if proposal_strength < MIN_PROPOSAL_STRENGTH: continue

        if proposal_strength < MIN_PROPOSAL_STRENGTH: continue
        if _penalise_template_reuse(raw, all_ideas, embed_model) == 0.0: continue

        raw.update({
            "gap_description": gap.get("gap_description", ""), "gap_type": gap.get("gap_type", ""),
            "paper_title": gap.get("paper_title", ""), "is_grounded": gap.get("is_grounded", False),
            "evidence_count": gap.get("evidence_count", 0), "grounding_evidence": gap.get("grounding_evidence", []),
            "research_axis": gap.get("research_axis", "general"),
        })
        all_ideas.append(raw)

    all_ideas.sort(key=lambda x: (1 if x.get("is_grounded") else 0, x.get("feasibility", 0), x.get("proposal_strength", 0)), reverse=True)

    results = []
    for idea in all_ideas[:MAX_IDEAS_OUT]:
        ps, feas = idea["proposal_strength"], idea["feasibility"]
        grounding_bonus = 1.0 if idea.get("is_grounded") else 0.0
        idea_score = round(0.40 * feas + 0.35 * ps + 0.25 * grounding_bonus, 3)

        novelty_cat = idea.get("novelty_category", "")
        label_map = {"high_novelty": "High Novelty", "moderate_novelty": "Moderate Novelty", "incremental_extension": "Incremental Extension", "potentially_saturated": "Potentially Saturated"}
        base_label = label_map.get(novelty_cat, "Incremental Extension")
        novelty_label = base_label if literature_vecs else f"{base_label} (unverified)"

        impl = idea.get("proposal_implement", {})
        primary_method = (idea.get("proposal_method") or "").split(".")[0].strip()
        
        results.append({
            "title": idea["title"],
            "approach_track": idea.get("approach_track", "hybrid"),
            "description": " ".join(filter(None, [idea.get("proposal_problem", ""), idea.get("proposal_method", "")])),
            "methodology": f"Method: {primary_method} | Data: {impl.get('dataset', '')} | Baselines: {impl.get('baseline', '')} | Metrics: {impl.get('metric', '')}",
            "feasibility": feas, "why_feasible_now": idea.get("why_feasible_now", ""),
            "novelty_label": novelty_label, "proposal_strength": ps, "novelty_score": idea.get("novelty_score", ps),
            "idea_score": idea_score, "causal_trace": idea.get("causal_trace", {}),
            "gap_description": idea["gap_description"], "gap_type": idea["gap_type"],
            "primary_method": primary_method,
            "dataset": impl.get("dataset", ""),
            "is_grounded": idea.get("is_grounded", False), "grounding_evidence": idea.get("grounding_evidence", []),
            "evidence_count": idea.get("evidence_count", 0), "evidence_quality": "author_stated" if idea.get("is_grounded") else "inferred",
            "why_novel": idea.get("why_novel", ""), "innovation_type": idea.get("innovation_type", ""),
            "failure_mode_addressed": idea.get("failure_mode_addressed", ""),
            "proposal_problem": idea.get("proposal_problem", ""), "proposal_method": idea.get("proposal_method", ""),
            "proposal_implement": {"input": impl.get("input", ""), "output": impl.get("output", ""), "dataset": impl.get("dataset", ""), "baseline": impl.get("baseline", ""), "metric": impl.get("metric", "")},
            "research_axis": idea.get("research_axis", "general"),
            "paper_title": idea.get("paper_title", ""),
        })

    # STEP 3: ENFORCE DIVERSITY RULE PROGRAMMATICALLY
    # The prompt alone cannot enforce cross-gap diversity since each gap is 
    # processed in an independent LLM call. We enforce 1 idea per research_axis here.
    final_results = []
    seen_axes = set()
    fallback_results = []
    
    # Sort by idea_score descending to keep the best idea per axis
    results.sort(key=lambda x: x.get("idea_score", 0), reverse=True)
    
    for idea in results:
        axis = idea.get("research_axis", "general")
        if axis not in seen_axes:
            final_results.append(idea)
            seen_axes.add(axis)
        else:
            fallback_results.append(idea)
            
    # If we have fewer than MAX_IDEAS_OUT, fill the rest with fallbacks
    while len(final_results) < MAX_IDEAS_OUT and fallback_results:
        final_results.append(fallback_results.pop(0))
        
    results = final_results

    print(f"\n[LAYER 4] DONE — returning {len(results)} proposal(s) (Diversity Enforced)")
    print(f"{'='*60}\n")
    return results

class IdeaGenerator:
    def __init__(self, groq: LLMClient = None, embed_model=None, mock_mode: bool = False):
        self.groq, self.embed_model, self.mock_mode = groq, embed_model, mock_mode

    def validate_gap(self, gap: Dict) -> bool:
        gap_desc = gap.get("gap_description", "")
        return bool(gap_desc) and not _reject_generic_pre_filter(gap_desc)

    def generate_idea(self, gap: Dict, query: str = "", domain: str = "", papers_context: Optional[List[Dict]] = None) -> Optional[Dict]:
        if not self.validate_gap(gap): return None
        if self.groq is None: raise RuntimeError("IdeaGenerator.groq is None")
        return _call_llm(gap, query, domain, self.groq, self.mock_mode, papers_context)

    def process(self, gaps: List[Dict], query: str = "", domain: str = "", aspects: Optional[List[str]] = None, literature_vecs: Optional[List] = None, paper_evidences: Optional[List[Dict]] = None) -> List[Dict]:
        if self.groq is None and not self.mock_mode: raise RuntimeError("IdeaGenerator.groq is None")
        return run_layer4(gaps, query, aspects or [], domain, self.groq, self.embed_model, self.mock_mode, literature_vecs, paper_evidences)