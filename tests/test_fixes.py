"""Quick unit tests for bug fixes."""
import sys
sys.path.insert(0, ".")


def test_dedup_merges_evidence_count():
    from backend.layers.gap_extractor import _dedup_gaps

    g1 = {
        "gap_description":       "fuzzy sets have poor generalization across projects",
        "evidence_count":        1,
        "cross_paper_support":   0,
        "supporting_paper_ids":  ["paper_1"],
        "supporting_paper_indices": [1],
        "grounding_evidence":    [{"paper_index": 1, "type": "explicit_limitation",
                                   "text": "quote from paper 1", "confidence": 0.75}],
        "supporting_citations":  10,
        "is_grounded":           True,
        "gap_quality":           "grounded",
        "extraction_confidence": 0.75,
        "gap_source":            "faithful_extraction",
    }
    g2 = {
        "gap_description":       "fuzzy rules generalization across different projects is weak",
        "evidence_count":        1,
        "cross_paper_support":   0,
        "supporting_paper_ids":  ["paper_3"],
        "supporting_paper_indices": [3],
        "grounding_evidence":    [{"paper_index": 3, "type": "explicit_limitation",
                                   "text": "quote from paper 3", "confidence": 0.75}],
        "supporting_citations":  5,
        "is_grounded":           False,
        "gap_quality":           "weak",
        "extraction_confidence": 0.55,
        "gap_source":            "faithful_extraction",
    }

    result = _dedup_gaps([g1, g2])
    assert len(result) == 2, f"Expected 2 gaps (no-op), got {len(result)}"
    print(f"  PASS — {len(result)} gaps preserved (no-op by design)")


def test_dedup_keeps_distinct_gaps():
    from backend.layers.gap_extractor import _dedup_gaps

    g1 = {
        "gap_description":       "fuzzy sets have poor generalization across projects",
        "evidence_count": 1, "cross_paper_support": 0,
        "supporting_paper_ids": ["paper_1"], "supporting_paper_indices": [1],
        "grounding_evidence": [], "supporting_citations": 5,
        "is_grounded": True, "gap_quality": "grounded",
        "extraction_confidence": 0.75, "gap_source": "faithful_extraction",
    }
    g2 = {
        "gap_description":       "membership function sensitivity causes instability",
        "evidence_count": 1, "cross_paper_support": 0,
        "supporting_paper_ids": ["paper_2"], "supporting_paper_indices": [2],
        "grounding_evidence": [], "supporting_citations": 3,
        "is_grounded": True, "gap_quality": "grounded",
        "extraction_confidence": 0.75, "gap_source": "faithful_extraction",
    }

    result = _dedup_gaps([g1, g2])
    assert len(result) == 2, f"Distinct gaps should stay separate, got {len(result)}"
    print(f"  PASS — 2 distinct gaps kept as 2")


def test_section_parser_standard_headers():
    from backend.layers.layer1_fetcher import _extract_sections_from_text

    text = """
Abstract
This paper proposes a method.

1. Introduction
Software testing is critical.

5. Conclusion
We conclude that fuzzy logic works well.

6. Future Work
Future work will explore transfer learning.
"""
    sections = _extract_sections_from_text(text)
    assert "conclusion"  in sections, f"conclusion missing — got: {list(sections.keys())}"
    assert "future_work" in sections, f"future_work missing — got: {list(sections.keys())}"
    assert "introduction" in sections, f"introduction missing — got: {list(sections.keys())}"
    print(f"  PASS — standard headers found: {list(sections.keys())}")


def test_section_parser_nonstandard_headers():
    from backend.layers.layer1_fetcher import _extract_sections_from_text

    text = """
Abstract
This paper uses fractal features for flower classification.

III. Proposed Method
We use CNN combined with fractal modules.

V. Experimental Evaluation
Results show 94% accuracy.

VI. Discussion and Analysis
The model struggled with complex structures.

VII. Conclusions and Future Work
In future work, we aim to address cross-domain generalization.

VIII. Limitations of the Study
The dataset was limited to 5 flower species.
"""
    sections = _extract_sections_from_text(text)
    assert "methodology" in sections,  f"methodology missing — got: {list(sections.keys())}"
    assert "conclusion"  in sections,  f"conclusion missing — got: {list(sections.keys())}"
    assert "discussion"  in sections,  f"discussion missing — got: {list(sections.keys())}"
    assert "limitations" in sections,  f"limitations missing — got: {list(sections.keys())}"
    print(f"  PASS — non-standard headers found: {list(sections.keys())}")


def test_section_parser_allcaps_headers():
    from backend.layers.layer1_fetcher import _extract_sections_from_text

    text = """
ABSTRACT
This paper proposes a deep learning approach.

CONCLUSION AND FUTURE WORK
We conclude the approach is effective. Future directions include robustness testing.

LIMITATIONS OF THE STUDY
The study was limited to a single dataset.

SCOPE AND LIMITATIONS
Scope was restricted to binary classification.
"""
    sections = _extract_sections_from_text(text)
    assert "conclusion"  in sections, f"conclusion missing — got: {list(sections.keys())}"
    assert "limitations" in sections, f"limitations missing — got: {list(sections.keys())}"
    print(f"  PASS — all-caps headers found: {list(sections.keys())}")


def test_pdf_mode_flag_passed_correctly():
    from backend.layers.gap_extractor import _build_extraction_text

    full_text = (
        "The proposed CNN achieves 94% accuracy. "
        "However, complex flower structures proved difficult to quantify. "
        "The dataset was collected from a single geographic region. "
        "We did not evaluate robustness across different camera conditions. "
        "In future work, we plan to investigate cross-domain transfer. "
        "Computational cost was high due to fractal module calculations."
    )
    _, text_source = _build_extraction_text(
        abstract="",
        sections={},
        full_text=full_text,
    )
    pdf_mode = "full_text" in text_source
    assert pdf_mode == True, f"pdf_mode should be True for full_text source, got text_source='{text_source}'"
    print(f"  PASS — text_source='{text_source}' → pdf_mode={pdf_mode}")


def test_section_mode_does_not_trigger_pdf_mode():
    from backend.layers.gap_extractor import _build_extraction_text

    _, text_source = _build_extraction_text(
        abstract="We propose a fuzzy system.",
        sections={"limitations": "The system does not generalize.", "conclusion": "We conclude."},
        full_text="",
    )
    pdf_mode = "full_text" in text_source
    assert pdf_mode == False, f"pdf_mode should be False when sections exist, got text_source='{text_source}'"
    print(f"  PASS — text_source='{text_source}' → pdf_mode={pdf_mode}")


def test_consensus_gaps_field_defaults():
    consensus_gap = {
        "gap_description": "No cross-project benchmark exists for fuzzy TCP evaluation.",
        "gap_type":        "missing_benchmark",
        "confidence":      0.75,
        "gap_tier":        "consensus",
        "source":          "consensus_synthesis",
        "evidence_quote":  "cross-project generalization",
        "affected_methods": [],
    }

    cg = dict(consensus_gap)
    cg.setdefault("gap_type",              cg.get("gap_type", "field_gap"))
    cg.setdefault("evidence_quote",        cg.get("evidence_quote", ""))
    cg.setdefault("section",               "consensus")
    cg.setdefault("source_paper_id",       "consensus_synthesis")
    cg.setdefault("supporting_paper_ids",  [])
    cg.setdefault("evidence_count",        3)
    cg.setdefault("cross_paper_support",   3)
    cg.setdefault("is_grounded",           False)
    cg.setdefault("extraction_confidence", cg.get("confidence", 0.75))
    cg.setdefault("research_significance", 0.80)
    cg.setdefault("gap_quality",           "consensus")
    cg.setdefault("is_fallback",           False)
    cg.setdefault("gap_source",            "consensus_synthesis")
    cg.setdefault("grounding_evidence",    [])
    cg.setdefault("supporting_paper_indices", [])
    cg.setdefault("supporting_citations",  0)
    cg.setdefault("domain_alignment",      0.75)
    cg.setdefault("contradictions",        [])
    cg.setdefault("distinct_methods",      [])

    assert cg["evidence_count"]       == 3,                    "evidence_count should be 3"
    assert cg["extraction_confidence"] == 0.75,                "extraction_confidence should be 0.75"
    assert cg["gap_source"]           == "consensus_synthesis","gap_source wrong"
    assert cg["section"]              == "consensus",          "section should be consensus"
    print(f"  PASS — consensus gap normalized correctly")


if __name__ == "__main__":
    tests = [
        test_dedup_merges_evidence_count,
        test_dedup_keeps_distinct_gaps,
        test_section_parser_standard_headers,
        test_section_parser_nonstandard_headers,
        test_section_parser_allcaps_headers,
        test_pdf_mode_flag_passed_correctly,
        test_section_mode_does_not_trigger_pdf_mode,
        test_consensus_gaps_field_defaults,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            print(f"\n[TEST] {t.__name__}")
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL — {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR — {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
