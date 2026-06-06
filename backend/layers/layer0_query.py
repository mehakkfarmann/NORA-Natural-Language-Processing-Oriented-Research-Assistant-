"""
NORA — Layer 0: Intent-Aware Query Processor (v2 — Production)
No LLM. No hallucination. Pure deterministic pipeline.

Architecture:
  Layer0QueryProcessor
    detect_domain()         → fine-grained domain + subdomain + arxiv_category
    extract_tasks()         → methods, tasks, artifacts from matched Intent
    generate_strict_query() → single high-precision retrieval query
    build_filters()         → must_include / must_exclude / year_range / bias flags
    generate_variants()     → ≤5 controlled, domain-locked query variants
    process(query) → dict   → full structured output for Layer 1+

Design Rules:
  Domain confidence < threshold  → narrowing mode (no expansion, stricter query)
  Broad/vague query detected      → enforces narrowing strategy, not expansion
  survey / review / taxonomy      → permanently excluded via must_exclude
  Variants stay within semantic field — zero cross-domain leakage
  All output keys are stable — downstream layers depend on them
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from layer0_domain import extract_domain_context, Intent, _INTENT_RULES, _FALLBACK_INTENT  # noqa
    _DOMAIN_MODULE_AVAILABLE = True
except ImportError:  # pragma: no cover — isolated runs / tests
    _DOMAIN_MODULE_AVAILABLE = False
    _INTENT_RULES = []

logger = logging.getLogger(__name__)


# CONSTANTS

# Minimum phrase-length score to trust a domain match.
# Short triggers (≤4 chars) like "llm", "rag" are accepted but flag low confidence.
_CONFIDENCE_THRESHOLD = 5

# Maximum query variants emitted. Caps API calls in Layer 1.
_MAX_VARIANTS = 5

# Fixed year window — keeps retrieval recent without hard-coding "current year".
_DEFAULT_YEAR_RANGE = "2020-2026"

# Signals that a query is suspiciously broad / survey-like.
# Triggers narrowing mode instead of expansion.
_BROAD_SIGNALS = {
    "overview", "review", "survey", "introduction", "tutorial",
    "comprehensive", "recent advances", "state of the art",
    "state-of-the-art", "sota", "progress", "challenges",
}

# Papers that match these tokens will be de-prioritised / excluded.
# Applied globally regardless of domain.
_GLOBAL_SURVEY_TOKENS = [
    "survey", "review", "overview", "tutorial", "introduction",
    "comprehensive survey", "recent advances", "systematic review",
    "literature review", "survey paper", "taxonomy",
]

# Cross-domain noise terms that pollute nearly every generic ML search.
_GLOBAL_NOISE_EXCLUSIONS = [
    "education", "pedagogy", "curriculum", "classroom",
    "student grading", "e-learning", "adaptive learning",
    "course recommendation",
]


# DATACLASS — structured output contract

@dataclass
class ProcessedQuery:
    """
    Full structured output of Layer0QueryProcessor.process().
    This is the contract between Layer 0 and all downstream layers.
    """
    # Core identity
    original_query:   str
    strict_query:     str
    domain:           str
    subdomain:        str
    arxiv_category:   str
    methods:          List[str]
    tasks:            List[str]
    query_variants:   List[str]

    # Retrieval filter block
    retrieval_filters: Dict

    # Metadata for downstream diagnostics
    domain_confidence: float
    narrowing_mode:   bool
    matched_phrase:   str

    # Provider-specific pass-throughs (used by Layer 1)
    openalex_concepts: List[str]
    s2_fields:         List[str]
    negative_constraints: List[str]

    def to_dict(self) -> Dict:
        """Serialise to the flat dict format Layer 1 already consumes."""
        return {
            "original_query":        self.original_query,
            "strict_query":          self.strict_query,
            "domain":                self.domain,
            "subdomain":             self.subdomain,
            "category":              self.arxiv_category,
            "methods":               self.methods,
            "tasks":                 self.tasks,
            "query_variants":        self.query_variants,
            "retrieval_filters":     self.retrieval_filters,
            # Metadata
            "domain_confidence":     self.domain_confidence,
            "narrowing_mode":        self.narrowing_mode,
            "matched_phrase":        self.matched_phrase,
            # Layer 1 pass-throughs
            "arxiv_category":        self.arxiv_category,
            "openalex_concepts":     self.openalex_concepts,
            "s2_fields":             self.s2_fields,
            "negative_constraints":  self.negative_constraints,
            # Backward-compat keys (existing pipeline reads these)
            "intent_domain":         self.domain,
            "intent_subdomain":      self.subdomain,
            "intent_methods":        self.methods,
            "intent_tasks":          self.tasks,
            "query_expansions":      self.query_variants,
            "query_is_valid":        True,
            "interpretation_confidence": self.domain_confidence,
            "anchor_mode":           "strict" if not self.narrowing_mode else "narrow",
        }


# MAIN CLASS

class Layer0QueryProcessor:
    """
    Production-grade Layer 0 query processor for NORA.

    Usage:
        processor = Layer0QueryProcessor()
        result    = processor.process("llm hallucination detection")
        layer1_input = result.to_dict()

    All public methods are independently testable.
    """

    def __init__(self) -> None:
        self._compiled_rules: List[Tuple[re.Pattern, str, object]] = []

        if _DOMAIN_MODULE_AVAILABLE:
            for phrase, intent in _INTENT_RULES:
                pattern = re.compile(
                    r"\b" + re.escape(phrase.lower()) + r"\b",
                    re.IGNORECASE,
                )
                self._compiled_rules.append((pattern, phrase, intent))

        self._broad_pattern = re.compile(
            r"\b(" + "|".join(re.escape(s) for s in _BROAD_SIGNALS) + r")\b",
            re.IGNORECASE,
        )

    # PUBLIC API

    def process(self, query: str) -> ProcessedQuery:
        """
        Main entry point. Accepts a raw user query string, returns ProcessedQuery.

        Pipeline:
          1. detect_domain     → match Intent rule, compute confidence
          2. extract_tasks     → pull methods/tasks from matched Intent
          3. generate_strict_query → build high-precision retrieval string
          4. build_filters     → construct retrieval filter block
          5. generate_variants → controlled, domain-locked expansion variants
          6. Assemble and return ProcessedQuery
        """
        query = query.strip()
        if not query:
            return self._fallback_result(query)

        intent, matched_phrase, confidence = self.detect_domain(query)
        methods, tasks, artifacts = self.extract_tasks(intent)
        strict_query = self.generate_strict_query(query, intent, matched_phrase, confidence)
        retrieval_filters = self.build_filters(intent, matched_phrase, confidence)

        narrowing = self._is_narrowing_mode(query, confidence)
        variants  = self.generate_variants(query, intent, matched_phrase, narrowing)

        result = ProcessedQuery(
            original_query    = query,
            strict_query      = strict_query,
            domain            = intent.domain,
            subdomain         = intent.subdomain,
            arxiv_category    = intent.arxiv_category,
            methods           = methods,
            tasks             = tasks,
            query_variants    = variants,
            retrieval_filters = retrieval_filters,
            domain_confidence = confidence,
            narrowing_mode    = narrowing,
            matched_phrase    = matched_phrase,
            openalex_concepts = intent.openalex_concepts,
            s2_fields         = intent.s2_fields,
            negative_constraints = self._merge_negatives(intent),
        )

        self._log(result)
        return result


    def detect_domain(self, query: str) -> Tuple[object, str, float]:
        """
        Matches query against Intent rules using longest-phrase-wins heuristic.

        Returns:
          (intent, matched_phrase, confidence_score: float 0.0–1.0)

        Confidence scoring:
          Phrase length → longer phrases = more specific = higher confidence
          Normalised by a reference length of 20 chars (a typical specific phrase)
          Clamped to [0.1, 1.0] so even the fallback carries a minimal signal
        """
        q = query.lower()
        best_intent   = None
        best_phrase   = ""
        best_len      = 0

        for pattern, phrase, intent in self._compiled_rules:
            if pattern.search(q):
                score = len(phrase)
                if score > best_len:
                    best_intent = intent
                    best_phrase = phrase
                    best_len    = score

        if best_intent is None:
            if _DOMAIN_MODULE_AVAILABLE:
                best_intent = _FALLBACK_INTENT
            else:
                best_intent = _InlineFallbackIntent()
            best_phrase = ""
            best_len    = 0

        confidence = min(1.0, max(0.1, best_len / 20.0))
        return best_intent, best_phrase, confidence


    def extract_tasks(self, intent) -> Tuple[List[str], List[str], List[str]]:
        """Extracts methods, tasks, artifacts from a matched Intent object."""
        methods   = list(getattr(intent, "methods",   []))
        tasks     = list(getattr(intent, "tasks",     []))
        artifacts = list(getattr(intent, "artifacts", []))
        return methods, tasks, artifacts


    def generate_strict_query(
        self,
        raw_query:     str,
        intent,
        matched_phrase: str,
        confidence:     float,
    ) -> str:
        """
        Builds the single best retrieval string for high-precision search.

        Strategy:
          High confidence (≥ threshold): use first Intent query_expansion
            (pre-authored, domain-expert quality) — strip boolean AND syntax
          Low confidence or no expansion defined: fall back to raw query
            surrounded by its two most-specific task terms (if available)
          Never adds out-of-domain terms
          Never modifies user's core phrasing — wraps it, doesn't replace it

        This query goes to arXiv / S2 as the primary retrieval string.
        """
        expansions = list(getattr(intent, "query_expansions", []))

        if confidence >= (_CONFIDENCE_THRESHOLD / 20.0) and expansions:
            best_expansion = expansions[0]
            clean = re.sub(r'\s+AND\s+', ' ', best_expansion).replace('"', '').strip()
            return clean

        tasks = list(getattr(intent, "tasks", []))
        if tasks:
            anchor = tasks[0]
            if anchor.lower() not in raw_query.lower():
                return f"{raw_query} {anchor}"

        return raw_query


    def build_filters(
        self,
        intent,
        matched_phrase: str,
        confidence:     float,
    ) -> Dict:
        """
        Constructs the retrieval_filters block consumed by Layer 1.

        must_include   → terms that MUST appear in abstract/title
        must_exclude   → terms that disqualify a paper
                         (surveys + domain-specific negatives + global noise)
        year_range     → default 2020-2026 (recent papers only)
        paper_type_bias → survey_penalty and application_boost flags

        Hard constraint: if confidence < threshold
          must_include list is extended (more restrictive), not shortened.
        """
        tasks     = list(getattr(intent, "tasks",     []))
        artifacts = list(getattr(intent, "artifacts", []))
        negatives = list(getattr(intent, "negative_constraints", []))

        must_include: List[str] = []
        if tasks:
            must_include.extend(tasks[:2])
        if artifacts:
            must_include.append(artifacts[0])

        if confidence < (_CONFIDENCE_THRESHOLD / 20.0) and len(tasks) > 2:
            must_include.extend(tasks[2:4])

        must_exclude: List[str] = []
        must_exclude.extend(_GLOBAL_SURVEY_TOKENS)
        must_exclude.extend(negatives[:8])
        must_exclude.extend(_GLOBAL_NOISE_EXCLUSIONS)

        must_include = _dedup(must_include)
        must_exclude = _dedup(must_exclude)

        return {
            "must_include": must_include,
            "must_exclude": must_exclude,
            "year_range":   _DEFAULT_YEAR_RANGE,
            "paper_type_bias": {
                "survey_penalty":      True,
                "application_boost":   True,
            },
        }


    def generate_variants(
        self,
        raw_query:      str,
        intent,
        matched_phrase: str,
        narrowing_mode: bool,
    ) -> List[str]:
        """
        Generates controlled query variants for multi-provider retrieval in Layer 1.

        Narrowing mode (broad query / low confidence):
          Emits ONLY 1–2 variants, all derived from strict_query
          Zero cross-domain leakage
          Forces specificity — raw broad query is NOT included as a variant

        Normal mode:
          Uses pre-authored Intent.query_expansions first (highest quality)
          Builds additional task×method combinations if expansions are sparse
          Always includes raw_query as a safety variant for recall
          Caps at _MAX_VARIANTS (default 5)

        Shared rules (both modes):
          Strip boolean AND syntax (S2/OA don't support it)
          Strip quotation marks from expansion strings
          Deduplicate, preserve order
        """
        expansions = list(getattr(intent, "query_expansions", []))
        tasks      = list(getattr(intent, "tasks",            []))
        methods    = list(getattr(intent, "methods",          []))

        variants: List[str] = []

        if narrowing_mode:
            if expansions:
                clean = _clean_boolean(expansions[0])
                variants.append(clean)
            if tasks and matched_phrase:
                variants.append(f"{matched_phrase} {tasks[0]}")
            elif tasks:
                variants.append(f"{raw_query} {tasks[0]}")

        else:
            for exp in expansions:
                clean = _clean_boolean(exp)
                if clean:
                    variants.append(clean)

            if len(variants) < 3 and tasks and methods:
                for task in tasks[:3]:
                    for method in methods[:2]:
                        combo = f"{task} {method}"
                        variants.append(combo)
                        if len(variants) >= _MAX_VARIANTS - 1:
                            break
                    if len(variants) >= _MAX_VARIANTS - 1:
                        break

            if raw_query not in variants:
                variants.append(raw_query)

        return _dedup(variants)[:_MAX_VARIANTS]

    # PRIVATE HELPERS

    def _is_narrowing_mode(self, query: str, confidence: float) -> bool:
        """
        Returns True when the processor should restrict rather than expand.

        Conditions:
          1. Domain confidence below threshold (vague/ambiguous query)
          2. Query contains broad-signal words ("survey", "overview", etc.)
          3. Query is very short AND confidence is not high
             (e.g. "AI" alone → could mean anything)
        """
        if confidence < (_CONFIDENCE_THRESHOLD / 20.0):
            return True

        if self._broad_pattern.search(query):
            return True

        word_count = len(query.split())
        if word_count <= 2 and confidence < 0.7:
            return True

        return False

    def _merge_negatives(self, intent) -> List[str]:
        """Merges domain-specific negatives with global noise exclusions."""
        domain_neg = list(getattr(intent, "negative_constraints", []))
        merged = _dedup(domain_neg + _GLOBAL_NOISE_EXCLUSIONS)
        return merged

    def _fallback_result(self, query: str) -> ProcessedQuery:
        """Returns a safe minimal ProcessedQuery for empty input."""
        filters = {
            "must_include": [],
            "must_exclude": _dedup(_GLOBAL_SURVEY_TOKENS + _GLOBAL_NOISE_EXCLUSIONS),
            "year_range":   _DEFAULT_YEAR_RANGE,
            "paper_type_bias": {"survey_penalty": True, "application_boost": True},
        }
        return ProcessedQuery(
            original_query    = query,
            strict_query      = query,
            domain            = "computer_science",
            subdomain         = "general",
            arxiv_category    = "",
            methods           = [],
            tasks             = [],
            query_variants    = [query] if query else [],
            retrieval_filters = filters,
            domain_confidence = 0.0,
            narrowing_mode    = True,
            matched_phrase    = "",
            openalex_concepts = ["Computer Science"],
            s2_fields         = ["Computer Science"],
            negative_constraints = _dedup(_GLOBAL_NOISE_EXCLUSIONS),
        )

    def _log(self, result: ProcessedQuery) -> None:
        bar = "=" * 64
        print(f"\n{bar}")
        print("  [LAYER 0] QUERY PROCESSOR  (v2 — Production)")
        print(bar)
        print(f"  Original:    '{result.original_query}'")
        print(f"  Strict:      '{result.strict_query}'")
        print(f"  Domain:      {result.domain}")
        print(f"  Subdomain:   {result.subdomain}")
        print(f"  Category:    {result.arxiv_category}")
        print(f"  Confidence:  {result.domain_confidence:.2f}  "
              f"{'[NARROWING MODE]' if result.narrowing_mode else '[EXPANSION MODE]'}")
        print(f"  Matched:     '{result.matched_phrase}'")
        print(f"  Methods:     {result.methods[:4]}")
        print(f"  Tasks:       {result.tasks[:3]}")
        print(f"  Variants ({len(result.query_variants)}):")
        for i, v in enumerate(result.query_variants, 1):
            print(f"    {i}. {v}")
        print(f"  Must-Include: {result.retrieval_filters['must_include'][:5]}")
        print(f"  Must-Exclude: {result.retrieval_filters['must_exclude'][:5]} …")
        print(bar)
        logger.info(
            "[Layer0v2] domain=%s | sub=%s | cat=%s | conf=%.2f | narrow=%s | "
            "variants=%d | query='%s'",
            result.domain, result.subdomain, result.arxiv_category,
            result.domain_confidence, result.narrowing_mode,
            len(result.query_variants), result.original_query[:60],
        )


# Module-level convenience wrapper
# Keeps backward compatibility with callers that used extract_domain_context()

_processor = Layer0QueryProcessor()


def process_query(query: str) -> Dict:
    """
    Module-level convenience function.
    Equivalent to Layer0QueryProcessor().process(query).to_dict()
    Safe to use as a drop-in replacement for extract_domain_context() in Layer 1.
    """
    return _processor.process(query).to_dict()


# UTILITIES

def _clean_boolean(s: str) -> str:
    """Strips boolean AND syntax and quote chars; returns plain search string."""
    return re.sub(r'\s+AND\s+', ' ', s).replace('"', '').strip()


def _dedup(lst: List[str]) -> List[str]:
    """Deduplicates a list preserving insertion order."""
    seen: set = set()
    out: List[str] = []
    for item in lst:
        low = item.lower()
        if low not in seen:
            seen.add(low)
            out.append(item)
    return out


# Emergency inline fallback (used when layer0_domain.py is not importable)

class _InlineFallbackIntent:
    """Minimal Intent-compatible object used when domain module is absent."""
    domain             = "computer_science"
    subdomain          = "general"
    arxiv_category     = ""
    methods:           List[str] = []
    tasks:             List[str] = []
    artifacts:         List[str] = []
    negative_constraints: List[str] = ["education", "pedagogy"]
    query_expansions:  List[str] = []
    openalex_concepts: List[str] = ["Computer Science"]
    s2_fields:         List[str] = ["Computer Science"]


# EXAMPLE RUNS  (python layer0_query_processor.py)

if __name__ == "__main__":
    import json

    processor = Layer0QueryProcessor()

    TEST_QUERIES = [
        "llm hallucination detection",
        "federated learning privacy",
        "deepfake detection temporal",
        "mutation testing software",
        "fuzzy logic software testing",
        "machine learning survey",
        "recent advances in AI overview",
        "deep learning",
        "",
        "quantum circuit optimization",
    ]

    for q in TEST_QUERIES:
        result = processor.process(q)
        d = result.to_dict()

        spec_output = {
            "original_query":   d["original_query"],
            "strict_query":     d["strict_query"],
            "domain":           d["domain"],
            "subdomain":        d["subdomain"],
            "category":         d["category"],
            "methods":          d["methods"],
            "tasks":            d["tasks"],
            "query_variants":   d["query_variants"],
            "retrieval_filters": d["retrieval_filters"],
        }
        print(json.dumps(spec_output, indent=2))
        print()
