from __future__ import annotations

import hashlib, json, logging, math, os, re, sqlite3, threading, time
import datetime as _dt
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import arxiv, itertools, requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
_CURRENT_YEAR = _dt.date.today().year

from backend.config_production import FALLBACK_TO_ORIGINAL, MAX_QUERY_LENGTH

MIN_PAPERS_PER_ROUND = 4
_QUALITY_FLOOR       = 0.35   # raised — low-quality papers waste gap extraction budget


# RATE LIMITER

class _RateLimiter:
    def __init__(self, rate: float, capacity: float) -> None:
        self._rate, self._capacity = rate, capacity
        self._tokens, self._last   = capacity, time.monotonic()
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
            self._last   = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            deficit = (1.0 - self._tokens) / self._rate
            self._tokens = 0.0
        time.sleep(deficit)

_s2_lim  = _RateLimiter(1.0, 3.0)
_oa_lim  = _RateLimiter(5.0, 10.0)
_ax_lim  = _RateLimiter(0.5, 2.0)


# PAPER MODEL

@dataclass
class Paper:
    paper_id: str; title: str; abstract: str; source: str; year: int
    authors:           List[str]      = field(default_factory=list)
    venue:             Optional[str]  = None
    doi:               Optional[str]  = None
    arxiv_id:          Optional[str]  = None
    s2_id:             Optional[str]  = None
    openalex_id:       Optional[str]  = None
    citations:         int            = 0
    categories:        List[str]      = field(default_factory=list)
    pdf_url:           Optional[str]  = None
    relevance_score:   float          = 0.0
    citation_influence:float          = 0.0
    domain_align_score:float          = 0.0
    quality_score:     float          = 0.0
    raw_abstract_length: int          = 0
    fetch_timestamp:   float          = field(default_factory=time.time)
    sections:          Optional[Dict[str, str]] = None
    full_text:         Optional[str]           = None  # raw text for full_text_fallback mode
    def is_valid(self) -> bool:
        return bool(
            self.title and len(self.title.strip()) >= 5
            and any([self.doi, self.arxiv_id, self.s2_id, self.openalex_id])
        )

    def to_dict(self)         -> Dict[str, Any]: return asdict(self)
    @classmethod
    def from_dict(cls, d)     -> "Paper":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _quality(abstract: str, citations: int, doi: Optional[str], year: int) -> float:
    s  = (0.40 if len((abstract or "").strip()) >= 50 else 0.15)
    s += (0.15 if citations >= 100 else 0.10 if citations >= 20 else 0.05 if citations >= 5 else 0.0)
    s += 0.10 if doi else 0.0
    s += (0.35 if year >= 2022 else 0.25 if year >= 2020 else 0.15 if year >= 2018 else 0.0)
    return round(min(1.0, s), 3)

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


# DOMAIN → CONTRADICTING CATEGORIES MAP
# Used by pre-ingestion filter to drop clearly wrong-domain papers.
# These are category strings that appear in paper.categories but contradict
# the detected intent domain. Soft filter — only drops papers with 2+ hits.

_DOMAIN_CONTRADICTIONS: Dict[str, List[str]] = {
    "software_engineering": [
        "agriculture", "hydroponics", "irrigation", "farming",
        "electrical engineering", "power systems", "battery",
        "lithium", "photovoltaic", "wind energy", "hvac",
        "bioreactor", "wastewater", "sewage",
        "motor control", "servo", "robotics",
    ],
    "machine_learning": [
        "agriculture", "hydroponics", "power systems", "wastewater",
        "motor control",
    ],
    "cybersecurity": [
        "agriculture", "hydroponics", "power systems", "battery",
        "motor control", "wastewater",
    ],
    "natural_language_processing": [
        "agriculture", "hydroponics", "power systems", "battery",
        "motor control", "wastewater",
    ],
    "computer_vision": [
        "agriculture", "hydroponics", "power systems", "battery",
        "motor control", "wastewater",
    ],
    "artificial_intelligence": [
        "agriculture", "hydroponics", "power systems", "battery",
        "motor control", "wastewater",
    ],
}


# CACHE

class PaperCache:
    def __init__(self, db_path: str = "data/cache/nora_papers.db", ttl: int = 604_800) -> None:
        self._db  = Path(db_path)
        self._ttl = ttl
        self._db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db) as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("""CREATE TABLE IF NOT EXISTS fetch_cache(
                query_hash TEXT PRIMARY KEY, query_text TEXT NOT NULL,
                papers_json TEXT NOT NULL, fetched_at REAL NOT NULL,
                source_filter TEXT, config_hash TEXT)""")

    def _fp(self, query: str, *, neg: List[str] = [], prec: bool = False,
            focus: str = "", cat: str = "") -> str:
        raw = json.dumps({"q": query.strip().lower(), "neg": sorted(neg),
                          "prec": prec, "focus": focus.strip().lower(),
                          "cat": cat.strip().lower(), "sv": "v8"},
                         sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    def get(self, query: str, **kw) -> Optional[List[Paper]]:
        try:
            with sqlite3.connect(self._db) as c:
                row = c.execute(
                    "SELECT papers_json FROM fetch_cache WHERE query_hash=? AND fetched_at>?",
                    (self._fp(query, **kw), time.time() - self._ttl)).fetchone()
                return [Paper.from_dict(x) for x in json.loads(row[0])] if row else None
        except Exception as e:
            logger.warning("[Cache.get] %s", e); return None

    def set(self, query: str, papers: List[Paper], **kw) -> None:
        if not papers: return
        try:
            with sqlite3.connect(self._db) as c:
                c.execute("INSERT OR REPLACE INTO fetch_cache VALUES(?,?,?,?,?,?)",
                          (self._fp(query, **kw), query.strip(),
                           json.dumps([p.to_dict() for p in papers], ensure_ascii=False),
                           time.time(), "all", ""))
        except Exception as e:
            logger.warning("[Cache.set] %s", e)


# PROVIDERS

@runtime_checkable
class PaperProvider(Protocol):
    name: str
    def search(self, query: str, max_results: int) -> List[Paper]: ...


class SemanticScholarProvider:
    name = "semantic_scholar"
    _BASE   = "https://api.semanticscholar.org/graph/v1"
    _FIELDS = "paperId,title,abstract,authors,year,externalIds,citationCount,fieldsOfStudy,url,openAccessPdf"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

    def _hdr(self) -> Dict:
        h = {"User-Agent": "NORA/7.0"}
        if self._key: h["x-api-key"] = self._key
        return h

    def search(
        self,
        query: str,
        max_results: int = 20,
        fields_of_study: Optional[List[str]] = None,  # BUG 3 FIX
    ) -> List[Paper]:
        _s2_lim.wait()
        params: Dict[str, Any] = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": self._FIELDS,
        }
        # BUG 3 FIX: add fieldsOfStudy constraint when available
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        try:
            r = requests.get(f"{self._BASE}/paper/search",
                             params=params, headers=self._hdr(), timeout=20)
            r.raise_for_status()
            items = r.json().get("data") or []
        except Exception as e:
            logger.warning("[S2] %s", e); return []
        return [p for item in items if (p := self._parse(item))][:max_results]

    def neighborhood(self, paper_id: str, max_results: int = 20) -> List[Paper]:
        seen, out = set(), []
        for ep in ("references", "citations"):
            _s2_lim.wait()
            try:
                r = requests.get(f"{self._BASE}/paper/{paper_id}/{ep}",
                                 params={"fields": self._FIELDS, "limit": max_results},
                                 headers=self._hdr(), timeout=20)
                if r.status_code != 200: continue
                entries = r.json().get("data") or []
            except Exception as e:
                logger.warning("[S2.%s] %s", ep, e); continue
            for entry in entries:
                d = entry.get("citedPaper") or entry.get("citingPaper") or entry
                if not isinstance(d, dict): continue
                uid = d.get("paperId") or (d.get("externalIds") or {}).get("DOI")
                if not uid or uid in seen: continue
                seen.add(uid)
                if p := self._parse(d): out.append(p)
        return out[:max_results]

    def _parse(self, d: Dict) -> Optional[Paper]:
        if not isinstance(d, dict) or not (title := (d.get("title") or "").strip()):
            return None
        ext   = d.get("externalIds") or {}
        cites = d.get("citationCount") or 0
        doi   = ext.get("DOI")
        # Prefer open-access PDF URL over generic S2 page URL.
        # openAccessPdf.url points directly to a fetchable PDF when S2 has one.
        oa_pdf_url = (d.get("openAccessPdf") or {}).get("url")
        best_pdf_url = oa_pdf_url or d.get("url")

        p = Paper(
            paper_id=f"s2:{d.get('paperId')}", title=title,
            abstract=(d.get("abstract") or "").strip(), source="semantic_scholar",
            year=d.get("year") or 0,
            authors=[a["name"] for a in (d.get("authors") or []) if a.get("name")],
            doi=doi, arxiv_id=ext.get("ArXiv"), s2_id=d.get("paperId"),
            citations=cites, categories=d.get("fieldsOfStudy") or [],
            pdf_url=best_pdf_url,
            citation_influence=min(1.0, cites / 100.0),
            raw_abstract_length=len(d.get("abstract") or ""),
            quality_score=_quality(d.get("abstract") or "", cites, doi, d.get("year") or 0),
        )
        return p if p.is_valid() else None


class OpenAlexProvider:
    name = "openalex"
    _BASE = "https://api.openalex.org/works"
    _SEL  = ("id,title,abstract_inverted_index,authorships,publication_year,"
             "doi,cited_by_count,primary_location,concepts,ids")

    def search(
        self,
        query: str,
        max_results: int = 20,
        concept_filter: Optional[List[str]] = None,  # BUG 3 FIX
    ) -> List[Paper]:
        _oa_lim.wait()
        params: Dict[str, Any] = {
            "search":   query,
            "per-page": min(max_results, 50),
            "select":   self._SEL,
            "mailto":   "nora@example.com",
        }
        # BUG 3 FIX: concept boosting — sort by relevance AND concept match
        # OpenAlex doesn't support hard concept filters in free-text search,
        # so we use sort=relevance_score (default) and post-filter by concept.
        # This is a soft filter, not hard — avoids losing poorly-classified papers.
        try:
            r = requests.get(self._BASE, params=params, timeout=20)
            r.raise_for_status()
            items = r.json().get("results") or []
        except Exception as e:
            logger.warning("[OA] %s", e); return []

        papers = [p for item in items if (p := self._parse(item))]

        # BUG 3 FIX: boost papers whose concepts match the intent
        if concept_filter and papers:
            papers = self._boost_by_concept(papers, concept_filter, max_results)

        return papers[:max_results]

    def _boost_by_concept(
        self, papers: List[Paper], concepts: List[str], max_results: int
    ) -> List[Paper]:
        """
        Re-ranks papers so those matching intent concepts come first.
        Does NOT hard-filter — just prioritizes. Preserves recall.
        """
        concept_lower = [c.lower() for c in concepts]

        def _concept_score(p: Paper) -> int:
            cats_lower = [c.lower() for c in p.categories]
            return sum(1 for c in concept_lower
                       if any(c in cat for cat in cats_lower))

        papers.sort(key=_concept_score, reverse=True)
        return papers

    @staticmethod
    def _abstract(inv: Optional[Dict]) -> str:
        if not inv: return ""
        try:
            pairs = [(pos, w) for w, positions in inv.items() for pos in positions]
            return " ".join(w for _, w in sorted(pairs))
        except Exception:
            return ""

    def _parse(self, d: Dict) -> Optional[Paper]:
        if not isinstance(d, dict) or not (title := (d.get("title") or "").strip()):
            return None
        abstract = self._abstract(d.get("abstract_inverted_index"))
        doi      = (d.get("doi") or "").replace("https://doi.org/", "") or None
        cites    = d.get("cited_by_count") or 0
        year     = d.get("publication_year") or 0
        arxiv_id = next((v.split("/")[-1] for v in (d.get("ids") or {}).values()
                         if isinstance(v, str) and "arxiv" in v.lower()), None)
        p = Paper(
            paper_id=f"oa:{d.get('id', '')}", title=title, abstract=abstract,
            source="openalex", year=year,
            authors=[a.get("author", {}).get("display_name", "")
                     for a in (d.get("authorships") or [])
                     if a.get("author", {}).get("display_name")],
            doi=doi, arxiv_id=arxiv_id, openalex_id=d.get("id", ""),
            citations=cites,
            categories=[c.get("display_name", "") for c in (d.get("concepts") or [])[:5]],
            citation_influence=min(1.0, cites / 100.0),
            raw_abstract_length=len(abstract),
            quality_score=_quality(abstract, cites, doi, year),
        )
        return p if p.is_valid() else None


class ArxivProvider:
    name = "arxiv"

    def search(
        self,
        query: str,
        max_results: int = 20,
        arxiv_category: Optional[str] = None,  # BUG 3 FIX
    ) -> List[Paper]:
        _ax_lim.wait()

        # BUG 3 FIX: prepend cat: constraint so arXiv searches within domain
        # e.g. cat:cs.SE AND ("software testing" AND "fuzzy logic")
        if arxiv_category:
            constrained_query = f"cat:{arxiv_category} AND ({query})"
        else:
            constrained_query = query

        logger.info("[arXiv] query='%s'", constrained_query[:100])

        try:
            results = list(arxiv.Client().results(
                arxiv.Search(
                    query=constrained_query,
                    max_results=min(max_results, 50),
                    sort_by=arxiv.SortCriterion.Relevance,
                )
            ))
        except Exception as e:
            logger.warning("[arXiv] constrained search failed (%s) — retrying bare", e)
            # Fallback to bare query if cat: constraint returns nothing
            try:
                results = list(arxiv.Client().results(
                    arxiv.Search(query=query, max_results=min(max_results, 50),
                                 sort_by=arxiv.SortCriterion.Relevance)
                ))
            except Exception as e2:
                logger.warning("[arXiv] bare fallback also failed: %s", e2)
                return []

        papers = []
        for r in results:
            abstract = (r.summary or "").strip()
            year     = r.published.year if r.published else 0
            p = Paper(
                paper_id=f"arxiv:{r.get_short_id()}",
                title=(r.title or "").strip(),
                abstract=abstract, source="arxiv", year=year,
                authors=[str(a) for a in r.authors],
                doi=r.doi, arxiv_id=r.get_short_id(),
                categories=list(r.categories or []), pdf_url=r.pdf_url,
                raw_abstract_length=len(abstract),
                quality_score=_quality(abstract, 0, r.doi, year),
            )
            if p.is_valid():
                papers.append(p)
        return papers[:max_results]


# PRE-INGESTION DOMAIN FILTER  (BUG 4 FIX)

def _intent_filter(papers: List[Paper], intent_domain: str, neg_constraints: List[str]) -> List[Paper]:
    """
    Drops papers that clearly contradict the detected intent domain.
    Applied immediately after retrieval, before dedup/ranking.

    Logic:
      - Check paper title + abstract against negative_constraints
      - Check paper categories against _DOMAIN_CONTRADICTIONS[intent_domain]
      - Drop only when 2+ contradiction signals hit (avoids false positives)
    """
    if not intent_domain:
        return papers

    contradictions = [c.lower() for c in _DOMAIN_CONTRADICTIONS.get(intent_domain, [])]
    neg_lower      = [n.lower() for n in neg_constraints]

    kept    = []
    dropped = 0

    for paper in papers:
        text  = f"{paper.title} {paper.abstract}".lower()
        cats  = " ".join(paper.categories).lower()

        # Count negative constraint hits in text
        neg_hits = sum(1 for n in neg_lower if n in text)

        # Count domain contradiction hits in categories
        cat_hits = sum(1 for c in contradictions if c in cats)

        # Drop if 2+ neg hits OR 2+ category contradiction hits
        # Single hit tolerated — avoids dropping papers that mention
        # a topic in passing (e.g. a fuzzy logic paper that mentions "plant model")
        if neg_hits >= 2 or cat_hits >= 2:
            logger.debug(
                "[Layer1-Filter] DROPPED neg_hits=%d cat_hits=%d | '%s'",
                neg_hits, cat_hits, paper.title[:60],
            )
            dropped += 1
        else:
            kept.append(paper)

    if dropped:
        logger.info("[Layer1-Filter] Dropped %d/%d off-domain papers", dropped, len(papers))
    return kept


# LLM RELEVANCE FILTER  (Layer-1 Precision Gate)
#
# Calls the Anthropic API to act as a strict domain-aware paper filter.
# Applied AFTER _intent_filter (heuristic gate) and quality floor, so the
# LLM only sees papers that already passed basic sanity checks.
#
# Design goals:
#   • Precision > recall — drop cross-domain / generic ML papers hard
#   • Return structured JSON so we can attach scores + reasons downstream
#   • Batch papers in one API call to stay within rate limits
#   • Silent fallback: if the API call fails, return papers unchanged
#   • Configurable: set NORA_LLM_FILTER_ENABLED=false to disable

class LLMRelevanceFilter:
    """
    Sends retrieved papers to Claude and asks it to act as a strict
    academic paper filter for research-gap discovery (as described in
    the NORA Layer-1 system prompt).

    Returns (filtered_papers, rejected_papers, raw_response_json).
    On any failure returns (papers, [], None) — never raises.
    """

    _MODEL   = "claude-sonnet-4-20250514"
    _TIMEOUT = 60
    _MAX_BATCH = 25          # papers per LLM call; keeps prompt size reasonable
    _MIN_SCORE = 0.45        # papers below this relevance_score are dropped

    # System prompt that mirrors the NORA Layer-1 spec
    _SYSTEM = """You are a Retrieval Optimization Engine for a research-gap discovery system (NORA).

Your ONLY job is to decide which papers are truly relevant to the given research query at a
domain-specific, dataset-aware level. You are a strict academic paper filter — not a semantic
search engine and not a summariser.

Rules:
1. UNDERSTAND THE DOMAIN STRICTLY from the query. Lock to that domain only.
2. HARD FILTER: reject papers that belong to an unrelated domain, are generic ML/DL surveys,
   or are pure theory with no grounding in the query's specific application area.
3. PRIORITIZE in this order:
   a. Dataset / benchmark papers for the query domain
   b. Task-specific architecture papers
   c. Domain-specific preprocessing / augmentation papers
   d. Clinical / applied papers in the domain
   e. Domain-specific review papers (last resort; must be tightly scoped)
4. PENALISE: survey papers, multi-domain AI papers, general CNN/DL explanation papers,
   papers without dataset specificity.
5. Precision > Recall. Returning fewer but highly relevant papers is always better.

Output ONLY valid JSON, no preamble, no markdown fences, exactly this schema:
{
  "filtered_papers": [
    {
      "title": "<exact title as given>",
      "relevance_score": <float 0.0-1.0>,
      "reason": "<one sentence>",
      "dataset_or_domain": "<dataset name or narrow domain>",
      "why_important_for_gap_extraction": "<one sentence>"
    }
  ],
  "rejected_papers": [
    {
      "title": "<exact title as given>",
      "reason_for_rejection": "<one sentence>"
    }
  ]
}"""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._enabled = os.environ.get("NORA_LLM_FILTER_ENABLED", "true").lower() != "false"

    def filter(
        self,
        papers: List[Paper],
        query: str,
        domain_context: Optional[Dict] = None,
    ) -> List[Paper]:
        """
        Run LLM precision filter.  Returns filtered list; never raises.
        Attaches relevance_score from LLM onto each kept paper.
        """
        if not self._enabled or not papers:
            return papers
        if not self._key:
            logger.warning("[LLMFilter] ANTHROPIC_API_KEY not set — skipping LLM filter")
            return papers

        try:
            import requests as _req
            kept: List[Paper] = []
            # Process in batches so prompt stays manageable
            for batch_start in range(0, len(papers), self._MAX_BATCH):
                batch = papers[batch_start: batch_start + self._MAX_BATCH]
                kept.extend(self._filter_batch(batch, query, domain_context, _req))
            return kept
        except Exception as e:
            logger.warning("[LLMFilter] Unexpected error — returning papers unfiltered: %s", e)
            return papers

    def _filter_batch(
        self,
        batch: List[Paper],
        query: str,
        dc: Optional[Dict],
        _req,
    ) -> List[Paper]:
        user_msg = self._build_user_message(batch, query, dc)
        try:
            resp = _req.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":         self._key,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":      self._MODEL,
                    "max_tokens": 4096,
                    "system":     self._SYSTEM,
                    "messages":   [{"role": "user", "content": user_msg}],
                },
                timeout=self._TIMEOUT,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("[LLMFilter] API call failed (%s) — batch returned unfiltered", e)
            return batch

        raw = resp.json()
        text = (raw.get("content") or [{}])[0].get("text", "")
        return self._parse_response(text, batch)

    @staticmethod
    def _build_user_message(batch: List[Paper], query: str, dc: Optional[Dict]) -> str:
        dc = dc or {}
        domain_hint = dc.get("intent_domain") or dc.get("domain") or ""
        subdomain   = dc.get("intent_subdomain") or ""
        neg         = ", ".join(dc.get("negative_constraints") or [])

        lines = [
            f"RESEARCH QUERY: {query}",
        ]
        if domain_hint:
            lines.append(f"DOMAIN: {domain_hint}" + (f" / {subdomain}" if subdomain else ""))
        if neg:
            lines.append(f"NEGATIVE CONSTRAINTS (must not be about): {neg}")
        lines.append("")
        lines.append("PAPERS TO EVALUATE:")
        for i, p in enumerate(batch, 1):
            abstract_snip = (p.abstract or "")[:300].replace("\n", " ")
            lines.append(
                f"{i}. Title: {p.title}\n"
                f"   Year: {p.year}  Citations: {p.citations}\n"
                f"   Categories: {', '.join(p.categories[:4]) or 'N/A'}\n"
                f"   Abstract (excerpt): {abstract_snip}"
            )
        return "\n".join(lines)

    def _parse_response(self, text: str, batch: List[Paper]) -> List[Paper]:
        # Strip accidental markdown fences
        text = re.sub(r"```(?:json)?", "", text).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("[LLMFilter] JSON parse error (%s) — batch returned unfiltered", e)
            return batch

        # Build title → Paper index for fast lookup (case-insensitive)
        title_index: Dict[str, Paper] = {
            _norm(p.title): p for p in batch
        }

        kept: List[Paper] = []
        for item in data.get("filtered_papers") or []:
            title_key = _norm(item.get("title") or "")
            score     = float(item.get("relevance_score") or 0.0)
            paper     = title_index.get(title_key)
            if paper is None:
                # Fuzzy fallback: pick best title overlap
                paper = self._fuzzy_match(title_key, title_index)
            if paper and score >= self._MIN_SCORE:
                paper.relevance_score = score
                # Attach filter metadata as a lightweight annotation
                paper.domain_align_score = score
                kept.append(paper)
                logger.debug(
                    "[LLMFilter] KEPT  %.2f | %s", score, paper.title[:70]
                )

        for item in data.get("rejected_papers") or []:
            logger.debug(
                "[LLMFilter] DROP  | %s | %s",
                (item.get("title") or "")[:60],
                (item.get("reason_for_rejection") or "")[:80],
            )

        if not kept:
            # Safety net: if LLM rejected everything, return the batch as-is
            # rather than starving downstream layers.
            logger.warning(
                "[LLMFilter] LLM rejected all %d papers — returning batch unfiltered "
                "(check query/domain_context alignment)", len(batch)
            )
            return batch

        logger.info(
            "[LLMFilter] %d/%d papers kept after LLM precision filter",
            len(kept), len(batch),
        )
        return kept

    @staticmethod
    def _fuzzy_match(title_key: str, index: Dict[str, Paper]) -> Optional[Paper]:
        """Word-overlap fallback when LLM returns a slightly different title string."""
        query_words = set(title_key.split())
        best_score, best_paper = 0.0, None
        for candidate_key, paper in index.items():
            cand_words = set(candidate_key.split())
            overlap = len(query_words & cand_words) / max(len(query_words | cand_words), 1)
            if overlap > best_score:
                best_score, best_paper = overlap, paper
        return best_paper if best_score >= 0.70 else None


# DEDUP + MERGE

def _dedup(papers_a: List[Paper], papers_b: List[Paper], max_results: int) -> List[Paper]:
    merged: Dict[str, Paper] = {}

    def _key(p: Paper) -> str:
        if p.arxiv_id:
           base = re.sub(r'v\d+$', '', p.arxiv_id.strip())
           return f"arxiv:{base}"
        if p.doi:
           return f"doi:{p.doi.lower()}"
        if p.s2_id:
           return f"s2:{p.s2_id}"
        if p.openalex_id:
           return f"oa:{p.openalex_id}"
        return f"title:{_norm(p.title)}"

    def _rank(p: Paper) -> float:
        recency = max(0.0, min(1.0, 1.0 - (_CURRENT_YEAR - p.year) / 20.0))
        return p.quality_score * 0.5 + recency * 0.3 + min(p.citations / 500.0, 1.0) * 0.2

    for p in papers_a + papers_b:
        k = _key(p)
        if k not in merged or _rank(p) > _rank(merged[k]):
            merged[k] = p
    return sorted(merged.values(), key=_rank, reverse=True)[:max_results]


# INTENT-AWARE ROUTER  (BUG 1, 2, 3 FIX)

class IntentAwareRouter:
    """
    Routes queries to providers using structured intent constraints.
    Each provider receives:
      - A targeted query variant (not the raw user string)
      - Its own constraint (arxiv cat, s2 fields, oa concept boost)
    Results from all variants are deduped before returning.
    """

    def __init__(self, providers: List[Any]) -> None:
        self._s2  = next((p for p in providers if p.name == "semantic_scholar"), None)
        self._oa  = next((p for p in providers if p.name == "openalex"), None)
        self._ax  = next((p for p in providers if p.name == "arxiv"), None)

    def search(
        self,
        query_variants: List[str],
        max_results: int,
        arxiv_category: str,
        s2_fields: List[str],
        openalex_concepts: List[str],
    ) -> List[Paper]:
        all_papers: List[Paper] = []
        per_variant = max(8, max_results // max(1, len(query_variants)))

        for variant in query_variants:
            # FIX 2: normalize query — strip boolean AND syntax and bare quotes
            # S2 and OpenAlex treat AND as a literal token, not a boolean operator.
            variant = variant.replace(" AND ", " ").replace('"', '').strip()
            logger.info("[Router] variant='%s'", variant[:80])

            # ArXiv — category-constrained
            if self._ax:
                try:
                    batch = self._ax.search(variant, per_variant,
                                            arxiv_category=arxiv_category or None)
                    logger.info("[Router] arXiv → %d | cat=%s", len(batch), arxiv_category)
                    all_papers.extend(batch)
                except Exception as e:
                    logger.warning("[Router] arXiv failed: %s", e)

            # Semantic Scholar — field-constrained
            if self._s2:
                try:
                    batch = self._s2.search(variant, per_variant,
                                            fields_of_study=s2_fields or None)
                    logger.info("[Router] S2 → %d | fields=%s", len(batch), s2_fields)
                    all_papers.extend(batch)
                except Exception as e:
                    logger.warning("[Router] S2 failed: %s", e)

            # OpenAlex — concept-boosted
            if self._oa:
                try:
                    batch = self._oa.search(variant, per_variant,
                                            concept_filter=openalex_concepts or None)
                    logger.info("[Router] OA → %d | concepts=%s", len(batch), openalex_concepts[:2])
                    all_papers.extend(batch)
                except Exception as e:
                    logger.warning("[Router] OA failed: %s", e)

        return _dedup(all_papers, [], max_results)


def _default_router(s2_key: Optional[str] = None) -> IntentAwareRouter:
    return IntentAwareRouter([
        SemanticScholarProvider(s2_key),
        OpenAlexProvider(),
        ArxivProvider(),
    ])


# TITLE-MATCH NEIGHBORHOOD

def _neighborhood_via_title(
    query: str, max_results: int,
    s2: SemanticScholarProvider, embed_model=None,
    threshold: float = 0.82,
) -> List[Paper]:
    _s2_lim.wait()
    try:
        r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
                         params={"query": query, "limit": 1, "fields": "paperId,title"},
                         headers=s2._hdr(), timeout=20)
        r.raise_for_status()
        items = r.json().get("data") or []
    except Exception as e:
        logger.warning("[S2-title] %s", e); return []
    if not items: return []

    found_title = items[0].get("title") or ""
    paper_id    = items[0].get("paperId") or ""
    sim = _title_sim(query, found_title, embed_model)
    if sim < threshold:
        logger.info("[S2-title] sim=%.3f < %.2f — skip", sim, threshold)
        return []
    logger.info("[S2-title] matched '%s' (sim=%.3f)", found_title[:60], sim)
    return s2.neighborhood(paper_id, max_results)


def _title_sim(query: str, title: str, embed_model=None) -> float:
    if embed_model is not None:
        try:
            import numpy as np
            vecs = embed_model.encode([query, title], normalize_embeddings=True, batch_size=2)
            return float(np.dot(vecs[0], vecs[1]))
        except Exception as e:
            logger.warning("[TitleSim] fallback: %s", e)
    a, b = set(_norm(query).split()), set(_norm(title).split())
    return len(a & b) / max(len(a | b), 1)


# FULL-TEXT SECTION ENRICHMENT
#
# Tries to get Limitations / Future Work / Discussion / Conclusion for each
# paper so the evidence extractor has real author statements to work with.
#
# Priority order per paper:
#   1. arXiv ID present  → fetch HTML from ar5iv.org (always open, easy to parse)
#   2. pdf_url is a PDF  → fetch and extract text    (works for OA PDFs from S2)
#   3. Neither           → skip, abstract only
#
# All failures are silent — paper.abstract is never touched.

_SECTION_PATTERNS = {
    "introduction": re.compile(
        r"^(\d+[\.\d]*\s*)?([IVXLC]+[\.\s]+)?"
        r"(introduction|background\s+and\s+motivation"
        r"|overview|problem\s+statement"
        r"|introduction\s+and\s+(background|motivation|overview)"
        r"|introduction\s+and\s+background)$",
        re.I,
    ),
    "related_work": re.compile(
        r"^(\d+[\.\d]*\s*)?([IVXLC]+[\.\s]+)?"
        r"(related\s+work|literature\s+review"
        r"|background|prior\s+work|state\s+of\s+the\s+art"
        r"|related\s+research|previous\s+work"
        r"|background\s+and\s+related\s+work)$",
        re.I,
    ),
    "motivation": re.compile(
        r"^(\d+[\.\d]*\s*)?([IVXLC]+[\.\s]+)?"
        r"(motivation|problem\s+(formulation|definition|statement)"
        r"|research\s+problem|challenges?|problem\s+description)$",
        re.I,
    ),
    "limitations": re.compile(
        r"^(\d+[\.\d]*\s*)?([IVXLC]+[\.\s]+)?"
        r"(limitations?|threats?\s+to\s+validity"
        r"|limitations?\s+and\s+(future|discussion|scope)"
        r"|threats?\s+and\s+limitations?"
        r"|study\s+limitations?|research\s+limitations?"
        r"|scope\s+and\s+limitations?"
        r"|limitations?\s+of\s+(the\s+)?(study|research|work|approach|method)"
        r"|validity\s+threats?)$",
        re.I,
    ),
    "future_work": re.compile(
        r"^(\d+[\.\d]*\s*)?([IVXLC]+[\.\s]+)?"
        r"(future\s+work|further\s+research"
        r"|open\s+(problems?|issues?|questions?)"
        r"|future\s+directions?|future\s+research"
        r"|future\s+studies|recommendations?\s+for\s+future"
        r"|future\s+work\s+and\s+conclusion[s]?"
        r"|future\s+work\s+and\s+recommendations?"
        r"|directions?\s+for\s+future"
        r"|future\s+enhancements?|future\s+improvements?)$",
        re.I,
    ),
    "discussion": re.compile(
        r"^(\d+[\.\d]*\s*)?([IVXLC]+[\.\s]+)?"
        r"(discussion|analysis|interpretation"
        r"|discussion\s+and\s+conclusion[s]?"
        r"|discussion\s+and\s+future"
        r"|discussion\s+and\s+(analysis|findings?|implications?|results?)"
        r"|findings?\s+and\s+discussion"
        r"|analysis\s+and\s+discussion"
        r"|results?\s+and\s+discussion"
        r"|discussion\s+of\s+results?)$",
        re.I,
    ),
    "conclusion": re.compile(
        r"^(\d+[\.\d]*\s*)?([IVXLC]+[\.\s]+)?"
        r"(conclusion[s]?|summary|closing\s+remarks"
        r"|concluding\s+remarks"
        r"|conclusion[s]?\s+and\s+future"
        r"|conclusion[s]?\s+and\s+(recommendations?|implications?|contributions?)"
        r"|summary\s+and\s+conclusion[s]?"
        r"|conclusions?\s+and\s+future\s+work"
        r"|final\s+remarks?|overall\s+conclusion[s]?"
        r"|conclusion[s]?\s+and\s+contributions?)$",
        re.I,
    ),
    "results": re.compile(
        r"^(\d+[\.\d]*\s*)?([IVXLC]+[\.\s]+)?"
        r"(results?|findings?|experiments?|evaluation|performance"
        r"|experimental\s+results?|evaluation\s+results?"
        r"|experimental\s+evaluation"
        r"|quantitative\s+results?|qualitative\s+results?"
        r"|experiments?\s+and\s+results?"
        r"|ablation\s+study|performance\s+evaluation"
        r"|implementation\s+details|training\s+details"
        r"|experimental\s+setup)$",
        re.I,
    ),
    "abstract": re.compile(
        r"^(\d+[\.\d]*\s*)?([IVXLC]+[\.\s]+)?"
        r"(abstract|summary|executive\s+summary)$",
        re.I,
    ),
    "methodology": re.compile(
        r"^(\d+[\.\d]*\s*)?([IVXLC]+[\.\s]+)?"
        r"(methodology|method[s]?|approach|proposed\s+(method|framework|system|model|architecture|approach)"
        r"|research\s+method|materials?\s+and\s+methods?"
        r"|system\s+(architecture|overview|design)"
        r"|proposed\s+methodology"
        r"|model\s+architecture|network\s+architecture"
        r"|dataset[s]?\s+and\s+preprocessing"
        r"|data\s+collection|dataset[s]?)$",
        re.I,
    ),
}

_SECTION_CHAR_CAP = 5000
_ARXIV_ID_RE      = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")


def _is_arxiv_id(paper_id: str) -> Optional[str]:
    """Return bare arXiv ID if paper_id looks like one, else None."""
    raw = paper_id.replace("arxiv:", "").strip()
    m   = _ARXIV_ID_RE.match(raw)
    return m.group(1) if m else None


def _extract_sections_from_text(text: str) -> Dict[str, str]:
    """Parse raw text into section buckets by scanning for section headers."""
    # Normalize multi-space sequences (common in pdfminer output from
    # two-column or justified PDF layouts, e.g. "future  work" → "future work")
    text = re.sub(r'  +', ' ', text)

    # Pre-compile a relaxed pattern for long-line matching.
    # Some PDF extractors put the header and body text on the same line,
    # e.g. "1. Introduction: Soft computing techniques ..."
    # We extract the leading 80 chars as a "candidate" header.
    _LONG_LINE_CACHE = {}
    for sec_name, pattern in _SECTION_PATTERNS.items():
        raw = pattern.pattern
        # Remove the trailing $ so the pattern matches at the start of a string
        # even when body text follows.
        relaxed = re.compile(raw.rstrip("$"), re.I)
        _LONG_LINE_CACHE[sec_name] = relaxed

    sections: Dict[str, str] = {}
    current: Optional[str]   = None
    buffer:  List[str]        = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Strip trailing colons/periods so "Abstract:", "1. Introduction:" match the patterns
        clean = line.rstrip(":.")
        matched = None
        matched_text = ""
        # Try strict pattern first (requires exact header match, i.e. header on its own line)
        for sec_name, pattern in _SECTION_PATTERNS.items():
            m = pattern.match(clean)
            if m:
                matched = sec_name
                matched_text = m.group(0)
                break
        if matched is None:
            # Relaxed: header may be fused with body text on the same line,
            # e.g. "1. Introduction: Soft computing techniques ..."
            for sec_name, relaxed in _LONG_LINE_CACHE.items():
                m = relaxed.match(clean)
                if m:
                    matched = sec_name
                    matched_text = m.group(0)
                    break
        if matched:
            if current and buffer and current not in sections:
                sections[current] = "\n".join(buffer)[:_SECTION_CHAR_CAP]
            current = matched
            buffer  = []
            # For fused headers (header + body on same line), extract body text
            rest = line[len(matched_text):].strip().lstrip(":;,. \t-–—")
            if len(rest) >= 30:
                buffer.append(rest)
        elif current:
            buffer.append(line)

    if current and buffer and current not in sections:
        sections[current] = "\n".join(buffer)[:_SECTION_CHAR_CAP]

    return sections


def _fetch_arxiv_full_text(arxiv_id: str) -> Optional[Dict[str, str]]:
    """
    Fetch full text from ar5iv.org (HTML mirror of arXiv).
    ar5iv is free, no auth needed, covers almost all arXiv papers.
    """
    url = f"https://ar5iv.org/abs/{arxiv_id}"
    try:
        _ax_lim.wait()
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "NORA/7.0 (research pipeline)"})
        if r.status_code != 200:
            return None
        # BUG 18 FIX: remove script/style blocks FIRST to prevent JS/CSS
        # leaking into section parser after generic tag stripping.
        text = re.sub(r"<script[\s\S]*?</script>", " ", r.text, flags=re.IGNORECASE)
        text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
        # then strip remaining tags
        text = re.sub(r"<[^>]+>", " ", text)
        text     = re.sub(r"\s{3,}", " ", text)
        sections = _extract_sections_from_text(text)
        return sections if sections else None
    except Exception as e:
        logger.debug("[Enrichment] ar5iv failed for %s: %s", arxiv_id, e)
        return None


def _fetch_pdf_full_text(pdf_url: str) -> Optional[Dict[str, str]]:
    """
    Fetch an open-access PDF and extract section text.
    Uses pdfminer if available, falls back to raw byte extraction.
    Only attempted when pdf_url ends in .pdf or contains /pdf/.
    """
    if not pdf_url:
        return None
    # Only attempt URLs that are clearly PDFs
    url_lower = pdf_url.lower()
    if not (url_lower.endswith(".pdf") or "/pdf/" in url_lower or "pdf" in url_lower):
        return None
    try:
        _s2_lim.wait()
        r = requests.get(pdf_url, timeout=20,
                         headers={"User-Agent": "NORA/7.0 (research pipeline)"},
                         stream=True)
        if r.status_code != 200:
            return None
        # Check content type — must be PDF
        ct = r.headers.get("Content-Type", "")
        if "pdf" not in ct.lower() and not url_lower.endswith(".pdf"):
            return None

        raw_bytes = b"".join(itertools.islice(r.iter_content(8192), 250))  # max ~2MB

        # Try pdfminer for clean text extraction
        try:
            import io
            from pdfminer.high_level import extract_text as pdfminer_extract
            text = pdfminer_extract(io.BytesIO(raw_bytes))
        except Exception:
            # Fallback: decode raw bytes, keep printable ASCII
            text = raw_bytes.decode("latin-1", errors="replace")
            text = re.sub(r"[^ -~\n]", " ", text)
        sections = _extract_sections_from_text(text)
        return sections if sections else None

    except Exception as e:
        logger.debug("[Enrichment] PDF fetch failed for %s: %s", pdf_url, e)
        return None


def enrich_paper_sections(paper: "Paper") -> "Paper":
    """
    Attempt to populate paper.sections with full-text section content.

    Strategy (in priority order):
      1. arXiv ID → ar5iv HTML (fast, reliable, free)
      2. pdf_url  → direct PDF fetch (works for S2 openAccessPdf URLs)

    Silent fallback — paper.abstract is never modified.
    """
    if paper.sections is not None:
        return paper

    bare_id = _is_arxiv_id(paper.paper_id)
    if not bare_id and paper.arxiv_id:
        bare_id = _is_arxiv_id(paper.arxiv_id)
    if not bare_id and paper.pdf_url:
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", paper.pdf_url or "")
        if m:
            bare_id = m.group(1)

    if bare_id:
        sections = _fetch_arxiv_full_text(bare_id)
        if sections:
            paper.sections = sections
            logger.debug("[Enrichment] arXiv '%s' → sections: %s",
                         paper.title[:50], list(sections.keys()))
            return paper

    if paper.pdf_url:
        sections = _fetch_pdf_full_text(paper.pdf_url)
        if sections:
            paper.sections = sections
            logger.debug("[Enrichment] PDF '%s' → sections: %s",
                         paper.title[:50], list(sections.keys()))

    return paper


def enrich_papers_batch(papers: List["Paper"], max_enrich: int = 10) -> List["Paper"]:
    """
    Run section enrichment on up to max_enrich papers.
    Tries arXiv papers first (highest success rate), then S2 OA PDF papers.
    """
    arxiv_papers = [
        p for p in papers
        if (p.source == "arxiv"
            or bool(p.arxiv_id)
            or bool(_is_arxiv_id(p.paper_id))
            or "arxiv" in (p.pdf_url or "").lower())
        and p.sections is None
    ]
    pdf_papers = [
        p for p in papers
        if p not in arxiv_papers
        and p.pdf_url
        and p.sections is None
        and (p.pdf_url.lower().endswith(".pdf") or "/pdf/" in p.pdf_url.lower())
    ]

    enriched = 0
    for paper in arxiv_papers:
        if enriched >= max_enrich:
            break
        enrich_paper_sections(paper)
        if paper.sections:
            enriched += 1

    for paper in pdf_papers:
        if enriched >= max_enrich:
            break
        enrich_paper_sections(paper)
        if paper.sections:
            enriched += 1

    logger.info(
        "[Enrichment] %d/%d papers enriched with full-text sections",
        enriched, len(papers),
    )
    return papers


# PUBLIC API

def fetch_papers(
    query: str,
    max_results: int               = 20,
    domain_context: Optional[Dict] = None,
    cache: Optional[PaperCache]    = None,
    embed_model                    = None,
    *,
    precision_mode: bool           = False,
    research_focus: Optional[str]  = None,
) -> List[Paper]:
    """
    Main entry point — V7.1.

    Intent-aware retrieval:
      1. Extracts arxiv_category, s2_fields, openalex_concepts, query_variants
         from domain_context (populated by Layer 0).
      2. Sends each query variant to each provider with its specific constraint.
      3. Applies pre-ingestion domain filter before dedup/cache.
      4. Falls back to raw query if constrained retrieval returns nothing.
    """
    query = query[:MAX_QUERY_LENGTH]
    if cache is None: cache = PaperCache()
    dc  = domain_context or {}

    # Extract intent from domain_context (set by Layer 0)
    neg_constraints    = dc.get("negative_constraints") or []
    arxiv_category     = dc.get("arxiv_category") or ""
    s2_fields          = dc.get("s2_fields") or []
    openalex_concepts  = dc.get("openalex_concepts") or []
    query_variants = dc.get("query_variants") or dc.get("query_expansions") or []
    intent_domain      = dc.get("intent_domain") or dc.get("domain") or ""

    # research_focus appended to query if set (user-supplied, not LLM)
    focus_str = (research_focus or "").strip()
    if focus_str and query_variants:
        query_variants = [f"{v} {focus_str}".strip() for v in query_variants]
    elif focus_str:
        query_variants = [f"{query} {focus_str}".strip()]

    if not query_variants:
        query_variants = [query]

    primary_variant = query_variants[0]
    fp_kw = dict(neg=neg_constraints, prec=precision_mode,
                 focus=focus_str, cat=arxiv_category)

    if cached := cache.get(primary_variant, **fp_kw):
        logger.info("[Layer1] Cache hit | variant='%s'", primary_variant[:60])

        # Detect stale cache entries: papers with no title or empty abstract
        # AND no sections — these are from old schema before text enrichment.
        stale = any(
            not p.title or p.title.strip() in ("", "Untitled")
            or (not (p.abstract or "").strip() and p.sections is None)
            for p in cached
        )
        if stale:
            logger.warning(
                "[Layer1] Cache entry is stale (empty titles/abstracts) — "
                "discarding and re-fetching."
            )
            # Fall through to live fetch below by NOT returning here
        else:
            # Re-run enrichment on cache hits in case cache was populated before
            # enrichment code existed (sections will be None on old cached papers).
            needs_enrichment = any(
                p.sections is None and (
                    bool(_is_arxiv_id(p.paper_id))
                    or bool(p.arxiv_id)
                    or bool(p.pdf_url)
                )
                for p in cached
            )
            if needs_enrichment:
                cached = enrich_papers_batch(cached, max_enrich=10)
                cache.set(primary_variant, cached, **fp_kw)
            return cached

    s2_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    router = _default_router(s2_key)

    logger.info(
        "[Layer1] Fetching | variants=%d | cat=%s | s2_fields=%s",
        len(query_variants), arxiv_category, s2_fields,
    )

    papers = router.search(
        query_variants  = query_variants,
        max_results     = max_results,
        arxiv_category  = arxiv_category,
        s2_fields       = s2_fields,
        openalex_concepts = openalex_concepts,
    )

    # Citation neighborhood on primary variant
    # FIX 4: only expand when intent_subdomain is defined (specific enough to
    # avoid generic CV/ML drift) AND similarity threshold raised to 0.86.
    intent_subdomain = dc.get("intent_subdomain") or ""
    if intent_subdomain and len(primary_variant.split()) >= 4:
        s2 = SemanticScholarProvider(api_key=s2_key)
        if hood := _neighborhood_via_title(primary_variant, max_results // 2,
                                           s2, embed_model, threshold=0.86):
            papers = _dedup(papers, hood, max_results)

    # BUG 4 FIX: pre-ingestion domain filter — drop contradicting papers NOW
    papers = _intent_filter(papers, intent_domain, neg_constraints)

    papers = [p for p in papers if p.quality_score >= _QUALITY_FLOOR]

    # LLM Precision Filter
    # Calls Claude to act as a strict domain-aware relevance judge.
    # Runs AFTER the heuristic _intent_filter so the LLM only sees papers
    # that passed basic quality/domain sanity checks — keeps prompt small.
    # Falls back silently on API failure (never raises, never loses papers).
    llm_filter = LLMRelevanceFilter(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    papers = llm_filter.filter(papers, query, domain_context)

    logger.info("[Layer1] → %d papers after filtering", len(papers))

    # Full-text enrichment — run BEFORE caching so sections are persisted.
    # Tries arXiv (ar5iv) first, then S2 openAccessPdf URLs.
    # max_enrich=10 keeps latency reasonable. Silent fallback on failure.
    if papers:
        papers = enrich_papers_batch(papers, max_enrich=10)
        enriched_count = sum(1 for p in papers if p.sections)
        logger.info(
            "[Layer1] Full-text enrichment: %d/%d papers got sections",
            enriched_count, len(papers),
        )

    if papers:
        cache.set(primary_variant, papers, **fp_kw)

    # Fallback: if constrained retrieval returned nothing, retry with raw query
    if not papers and FALLBACK_TO_ORIGINAL and (arxiv_category or focus_str):
        logger.info("[Layer1] Constrained retrieval empty — retrying with raw query")
        fallback_ctx = {k: v for k, v in dc.items()}
        fallback_ctx["query_variants"] = [query]
        fallback_ctx["arxiv_category"] = ""
        fallback_ctx["s2_fields"]      = []
        return fetch_papers(query, max_results, fallback_ctx, cache,
                            embed_model, precision_mode=precision_mode,
                            research_focus=None)
    return papers


# CONFIDENCE CONFIG

@dataclass
class ConfidenceConfig:
    rerank_weight: float = 0.50
    citation_weight: float = 0.20
    evidence_weight: float = 0.30

    def compute_confidence(self, rerank: float, citation: float, evidence: float) -> float:
        return round(
            self.rerank_weight * rerank
            + self.citation_weight * citation
            + self.evidence_weight * evidence, 3
        )


# Backward-compat alias
_fetch_with_fallback = fetch_papers

__all__ = [
    "Paper", "PaperCache", "PaperProvider",
    "SemanticScholarProvider", "OpenAlexProvider", "ArxivProvider",
    "IntentAwareRouter", "LLMRelevanceFilter",
    "fetch_papers", "_fetch_with_fallback",
    "ConfidenceConfig", "_quality",
    "enrich_paper_sections", "enrich_papers_batch",
]