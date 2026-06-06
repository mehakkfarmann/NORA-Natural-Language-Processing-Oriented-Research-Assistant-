# Centralised thresholds for all pipeline layers.
import os
S2_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "s2k-1MYiahtQpoeIpuz4qc3GJE92CybLayv4ANdYSILQ")

BGE_MODEL  = "BAAI/bge-large-en-v1.5"
GROQ_MODEL = "llama-3.3-70b-versatile"
DEBUG = False

# Layer 0 — Smart Query Builder
ALIGN_THRESHOLD = 0.72          # min semantic similarity: original ↔ rewritten query
PURITY_THRESHOLD = 0.70         # fraction of papers in dominant arXiv category
STD_THRESHOLD = 0.15            # max std-dev of embedding space for focused corpus
_PAPERS_PER_PROBE = 5

# Layer 1 — Paper Fetcher
ARXIV_MAX_RESULTS = 10
DEFAULT_MAX_RESULTS = 50
DATE_FILTER_YEARS = 5
MIN_ABSTRACT_CHARS = 100

# Layer 2 — Semantic Ranker
CATEGORY_MATCH_BOOST = 1.15
RECENCY_LAMBDA = 0.30

# Layer 2.5 — LLM Relevance Filter
LLM_BATCH_SIZE = 5

# Layer 2.75 — Synthesizer
RELEVANCE_THRESHOLD = 0.45      # floor for paper surviving synthesis
MIN_PAPERS_TO_KEEP = 8          # emergency fallback when zero pass threshold
GAP_SIMILARITY_THRESHOLD = 0.75
MIN_CONSENSUS_PAPERS = 2

# Layer 3 — Gap Extractor
MAX_PAPERS_FOR_GAP_EXTRACTION = 999
MAX_GAPS_PER_PAPER = 3
EXPLICIT_CONF_THRESHOLD = 0.65
INFERRED_CONF_CAP = 0.70
INFERRED_CONF_THRESHOLD = 0.45
PDF_TIMEOUT = 15

# Layer 4 — Idea Generator
MIN_GAP_CONFIDENCE = 0.50
NOVELTY_NOVEL_THRESHOLD = 0.80
NOVELTY_RELATED_THRESHOLD = 0.55

# System
MAX_RETRIES = 3
HTTP_TIMEOUT = 20
USER_AGENT = "NORA/1.0 (Research Assistant)"

# Logging
LOG_LEVEL = "INFO"
ENABLE_DEBUG_TIMING = False

