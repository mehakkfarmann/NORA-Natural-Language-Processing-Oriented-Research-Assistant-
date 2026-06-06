"""Deduplicate gap dicts by embedding similarity, keeping highest confidence per cluster."""
from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)
_DEFAULT_SIM_THRESHOLD = 0.78


def deduplicate_gaps(
    gaps: list[dict],
    embed_model,
    similarity_threshold: float = _DEFAULT_SIM_THRESHOLD,
) -> list[dict]:
    if not gaps:
        return []
    if len(gaps) == 1:
        return gaps[:]

    logger.info("[GapDedup] Deduplicating %d gaps | threshold=%.2f", len(gaps), similarity_threshold)

    texts = [g.get("gap_description", "").strip() for g in gaps]
    valid_indices = [i for i, t in enumerate(texts) if t]
    if not valid_indices:
        logger.warning("[GapDedup] All gaps have empty descriptions — returning empty")
        return []

    valid_gaps  = [gaps[i]  for i in valid_indices]
    valid_texts = [texts[i] for i in valid_indices]

    try:
        from sentence_transformers.util import cos_sim
        vecs = embed_model.encode(
            valid_texts, normalize_embeddings=True, show_progress_bar=False,
        )
    except Exception as exc:
        logger.warning("[GapDedup] Embedding failed (%s) — returning gaps unsorted", exc)
        return valid_gaps[:]

    sim_matrix = cos_sim(vecs, vecs).numpy()

    kept:    list[dict] = []
    dropped: set[int]   = set()

    for i in range(len(valid_gaps)):
        if i in dropped:
            continue
        cluster = [i]
        for j in range(i + 1, len(valid_gaps)):
            if j not in dropped and sim_matrix[i][j] >= similarity_threshold:
                cluster.append(j)
                dropped.add(j)
        best_idx = max(cluster, key=lambda k: float(valid_gaps[k].get("confidence", 0.0)))
        kept.append(valid_gaps[best_idx])

    kept.sort(key=lambda g: float(g.get("confidence", 0.0)), reverse=True)

    logger.info(
        "[GapDedup] Done | %d → %d gaps (removed %d duplicates)",
        len(valid_gaps), len(kept), len(valid_gaps) - len(kept),
    )
    return kept
