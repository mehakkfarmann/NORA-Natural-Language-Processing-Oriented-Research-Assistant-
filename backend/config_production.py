"""Tunable thresholds for graceful degradation across all layers."""

MAX_QUERY_LENGTH = 80
FALLBACK_TO_ORIGINAL = True
MAX_RETRIES_PER_SOURCE = 2
S2_BOOST_IF_ARXIV_FAILS = True
ARXIV_TIMEOUT_SECONDS = 30

INITIAL_RELEVANCE_THRESHOLD = 0.50
MIN_RELEVANCE_THRESHOLD = 0.35
RELEVANCE_RELAX_STEP = 0.05

INITIAL_DOMAIN_ALIGN_THRESHOLD = 0.20
MIN_DOMAIN_ALIGN_THRESHOLD = 0.05
DOMAIN_RELAX_STEP = 0.05

DEFAULT_DOMAIN_ALIGN_SCORE = 0.30
DEFAULT_CITATION_SCORE = 0.50

ENABLE_BEST_EFFORT_MODE = True
BEST_EFFORT_MAX_PAPERS = 5
BEST_EFFORT_CONFIDENCE_FLOOR = 0.20

ENABLE_GENERIC_GAP_FALLBACK = True
GENERIC_GAP_TEMPLATES = [
    "Limited evaluation of {method} in real-world {domain} settings.",
    "Lack of benchmark datasets for {task} in {domain}.",
    "Scalability challenges for {approach} at production scale.",
]

LOG_FILTERING_DECISIONS = True
TRACK_QUERY_METRICS = True
METRICS_TO_TRACK = [
    "query_length", "papers_retrieved", "papers_after_filter",
    "avg_relevance_score", "avg_domain_align", "gaps_extracted", "ideas_generated",
]