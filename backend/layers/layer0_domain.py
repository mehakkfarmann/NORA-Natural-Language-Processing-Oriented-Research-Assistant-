"""Deterministic domain detection + intent extraction. No LLM."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)



@dataclass
class Intent:
    """
    Structured representation of what the user actually wants.
    Layer 1 uses this to build provider-specific constrained queries.
    Layer 2.5 uses domain to verify paper-domain consistency.
    """
    domain:              str
    subdomain:           str
    arxiv_category:      str
    methods:             List[str]
    tasks:               List[str]
    artifacts:           List[str]
    negative_constraints: List[str]
    # Provider-specific query expansion terms — used by Layer 1
    query_expansions:    List[str]        = field(default_factory=list)
    # OpenAlex concept filter terms (display names, not IDs — more stable)
    openalex_concepts:   List[str]        = field(default_factory=list)
    s2_fields:           List[str]        = field(default_factory=list)



# Order matters: more specific phrases first, broad terms last
_INTENT_RULES: List[tuple[str, Intent]] = [

    ("fuzzy logic testing", Intent(
        domain="software_engineering", subdomain="software_testing",
        arxiv_category="cs.SE",
        methods=["fuzzy logic"],
        tasks=["software testing", "test case generation", "test suite optimization",
               "regression testing", "software quality assessment"],
        artifacts=["test case", "test suite", "test oracle"],
        negative_constraints=[
            "PID controller", "power system", "motor control", "robotics",
            "photovoltaic", "solar", "MPPT", "wind turbine", "inverter",
            "wastewater", "bioreactor", "sewage", "effluent",
            "hydroponics", "agriculture", "irrigation", "crop", "farming",
            "manufacturing", "industrial plant", "factory", "production line",
            "HVAC", "heating", "ventilation", "air conditioning",
            "lithium", "battery", "charging", "course recommendation",
            "student", "e-learning", "adaptive learning",
        ],
        query_expansions=[
            '"software testing" "fuzzy logic"',
            '"test case generation" "fuzzy logic"',
            '"software quality" "fuzzy logic"',
            '"regression testing" fuzzy',
        ],
        openalex_concepts=["Software Engineering", "Software Testing"],
        s2_fields=["Computer Science"],
    )),

    ("mutation testing", Intent(
        domain="software_engineering", subdomain="software_testing",
        arxiv_category="cs.SE",
        methods=["mutation testing"],
        tasks=["fault detection", "test adequacy", "test effectiveness"],
        artifacts=["mutant", "test suite", "test case"],
        negative_constraints=["education", "pedagogy", "biology", "genetics"],
        query_expansions=[
            '"mutation testing" software',
            '"mutation operator" test',
            '"fault detection" mutation',
        ],
        openalex_concepts=["Software Engineering", "Software Testing"],
        s2_fields=["Computer Science"],
    )),

    ("test generation", Intent(
        domain="software_engineering", subdomain="software_testing",
        arxiv_category="cs.SE",
        methods=["automated test generation"],
        tasks=["test case generation", "test automation", "coverage"],
        artifacts=["test case", "test suite"],
        negative_constraints=["education", "pedagogy"],
        query_expansions=[
            '"test case generation" automated',
            '"test generation" software coverage',
        ],
        openalex_concepts=["Software Engineering"],
        s2_fields=["Computer Science"],
    )),

    ("flaky test", Intent(
        domain="software_engineering", subdomain="software_testing",
        arxiv_category="cs.SE",
        methods=["flaky test detection"],
        tasks=["test reliability", "test stability"],
        artifacts=["test suite", "CI pipeline"],
        negative_constraints=["education"],
        query_expansions=['"flaky test" detection', '"non-deterministic test"'],
        openalex_concepts=["Software Engineering"],
        s2_fields=["Computer Science"],
    )),

    ("defect prediction", Intent(
        domain="software_engineering", subdomain="software_quality",
        arxiv_category="cs.SE",
        methods=["machine learning", "static analysis"],
        tasks=["defect prediction", "software quality", "fault localization"],
        artifacts=["software module", "code metrics"],
        negative_constraints=["education"],
        query_expansions=[
            '"defect prediction" software',
            '"bug prediction" machine learning',
            '"fault prediction" software quality',
        ],
        openalex_concepts=["Software Engineering", "Software Quality"],
        s2_fields=["Computer Science"],
    )),

    ("code generation", Intent(
        domain="software_engineering", subdomain="program_synthesis",
        arxiv_category="cs.SE",
        methods=["large language model", "neural network"],
        tasks=["code generation", "program synthesis", "code completion"],
        artifacts=["source code", "program"],
        negative_constraints=["education"],
        query_expansions=[
            '"code generation" LLM',
            '"program synthesis" neural',
            '"automatic programming"',
        ],
        openalex_concepts=["Software Engineering", "Programming Language"],
        s2_fields=["Computer Science"],
    )),

    ("program analysis", Intent(
        domain="software_engineering", subdomain="program_analysis",
        arxiv_category="cs.SE",
        methods=["static analysis", "dynamic analysis"],
        tasks=["program analysis", "bug detection", "vulnerability detection"],
        artifacts=["source code", "control flow graph"],
        negative_constraints=["education"],
        query_expansions=[
            '"program analysis" static',
            '"static analysis" bug detection',
        ],
        openalex_concepts=["Software Engineering"],
        s2_fields=["Computer Science"],
    )),

    ("normalization completeness", Intent(
        domain="software_engineering", subdomain="database_design",
        arxiv_category="cs.DB",
        methods=["fuzzy logic", "formal methods"],
        tasks=["normalization assessment", "schema quality", "completeness measurement"],
        artifacts=["conceptual model", "entity relationship diagram", "database schema"],
        negative_constraints=["education", "pedagogy", "signal normalization", "data normalization neural"],
        query_expansions=[
            '"normalization completeness" conceptual model',
            '"database normalization" fuzzy',
            '"schema quality" assessment',
            '"entity relationship" normalization',
        ],
        openalex_concepts=["Database", "Software Engineering"],
        s2_fields=["Computer Science"],
    )),

    ("conceptual model", Intent(
        domain="software_engineering", subdomain="database_design",
        arxiv_category="cs.DB",
        methods=["formal methods", "ontology"],
        tasks=["conceptual modeling", "schema design", "model quality"],
        artifacts=["conceptual model", "ER diagram", "UML diagram"],
        negative_constraints=["education", "mental model", "psychology"],
        query_expansions=[
            '"conceptual model" quality',
            '"conceptual schema" design',
            '"entity relationship" modeling',
        ],
        openalex_concepts=["Database", "Software Engineering"],
        s2_fields=["Computer Science"],
    )),

    ("knowledge graph", Intent(
        domain="artificial_intelligence", subdomain="knowledge_representation",
        arxiv_category="cs.AI",
        methods=["graph embedding", "ontology"],
        tasks=["knowledge graph completion", "entity linking", "relation extraction"],
        artifacts=["knowledge graph", "ontology", "triple store"],
        negative_constraints=["education"],
        query_expansions=[
            '"knowledge graph" completion',
            '"knowledge graph" embedding',
            '"entity linking" knowledge',
        ],
        openalex_concepts=["Artificial Intelligence", "Knowledge Graph"],
        s2_fields=["Computer Science"],
    )),

    ("fuzzy logic", Intent(
        domain="artificial_intelligence", subdomain="fuzzy_systems",
        arxiv_category="cs.AI",
        methods=["fuzzy logic", "fuzzy sets"],
        tasks=["decision making", "classification", "control"],
        artifacts=["fuzzy rule", "membership function", "fuzzy system"],
        negative_constraints=[
            "pedagogy", "education",
            "photovoltaic", "solar", "MPPT", "wind turbine",
            "wastewater", "bioreactor", "hydroponics", "irrigation",
            "manufacturing", "industrial plant", "HVAC",
            "lithium", "battery", "course recommendation",
        ],
        query_expansions=[
            '"fuzzy logic" application',
            '"fuzzy set" theory application',
        ],
        openalex_concepts=["Artificial Intelligence", "Fuzzy Logic"],
        s2_fields=["Computer Science"],
    )),

    ("fuzzy set", Intent(
        domain="artificial_intelligence", subdomain="fuzzy_systems",
        arxiv_category="cs.AI",
        methods=["fuzzy sets", "fuzzy logic"],
        tasks=["uncertainty modeling", "approximate reasoning"],
        artifacts=["membership function", "fuzzy set"],
        negative_constraints=["pedagogy"],
        query_expansions=['"fuzzy set" theory', '"fuzzy sets" application'],
        openalex_concepts=["Artificial Intelligence", "Fuzzy Logic"],
        s2_fields=["Computer Science"],
    )),

    ("federated learning", Intent(
        domain="machine_learning", subdomain="distributed_learning",
        arxiv_category="cs.LG",
        methods=["federated learning", "distributed training"],
        tasks=["privacy preserving ML", "model aggregation", "distributed training"],
        artifacts=["global model", "local model", "aggregation protocol"],
        negative_constraints=["centralized"],
        query_expansions=[
            '"federated learning" privacy',
            '"federated learning" aggregation',
            '"distributed learning" privacy preserving',
        ],
        openalex_concepts=["Machine Learning", "Federated Learning"],
        s2_fields=["Computer Science"],
    )),

    ("deep learning", Intent(
        domain="machine_learning", subdomain="deep_learning",
        arxiv_category="cs.LG",
        methods=["deep learning", "neural network"],
        tasks=["classification", "regression", "representation learning"],
        artifacts=["neural network", "model architecture"],
        negative_constraints=["education"],
        query_expansions=['"deep learning" application', '"neural network" training'],
        openalex_concepts=["Machine Learning", "Deep Learning"],
        s2_fields=["Computer Science"],
    )),

    ("reinforcement learning", Intent(
        domain="machine_learning", subdomain="reinforcement_learning",
        arxiv_category="cs.LG",
        methods=["reinforcement learning", "reward function"],
        tasks=["policy optimization", "reward maximization", "agent training"],
        artifacts=["policy", "reward function", "agent"],
        negative_constraints=["education"],
        query_expansions=[
            '"reinforcement learning" policy',
            '"reward function" optimization',
        ],
        openalex_concepts=["Machine Learning", "Reinforcement Learning"],
        s2_fields=["Computer Science"],
    )),

    ("machine learning", Intent(
        domain="machine_learning", subdomain="machine_learning",
        arxiv_category="cs.LG",
        methods=["machine learning"],
        tasks=["classification", "prediction", "feature engineering"],
        artifacts=["model", "dataset", "feature"],
        negative_constraints=["education"],
        query_expansions=['"machine learning" application', '"supervised learning"'],
        openalex_concepts=["Machine Learning"],
        s2_fields=["Computer Science"],
    )),

    ("vulnerability detection", Intent(
        domain="cybersecurity", subdomain="vulnerability_analysis",
        arxiv_category="cs.CR",
        methods=["static analysis", "dynamic analysis", "machine learning"],
        tasks=["vulnerability detection", "bug finding", "security analysis"],
        artifacts=["source code", "binary", "CVE"],
        negative_constraints=["education"],
        query_expansions=[
            '"vulnerability detection" static analysis',
            '"security vulnerability" machine learning',
            '"bug detection" automated',
        ],
        openalex_concepts=["Computer Security", "Software Engineering"],
        s2_fields=["Computer Science"],
    )),

    ("ai-generated code", Intent(
        domain="cybersecurity", subdomain="ai_code_security",
        arxiv_category="cs.CR",
        methods=["large language model", "static analysis"],
        tasks=["vulnerability detection", "code security", "code review"],
        artifacts=["generated code", "LLM output"],
        negative_constraints=["education"],
        query_expansions=[
            '"AI-generated code" vulnerability',
            '"LLM code" security',
            '"code generation" security vulnerability',
        ],
        openalex_concepts=["Computer Security", "Software Engineering"],
        s2_fields=["Computer Science"],
    )),

    ("code vulnerability", Intent(
        domain="cybersecurity", subdomain="vulnerability_analysis",
        arxiv_category="cs.CR",
        methods=["static analysis", "deep learning"],
        tasks=["vulnerability detection", "code security"],
        artifacts=["source code", "vulnerability"],
        negative_constraints=["education"],
        query_expansions=[
            '"code vulnerability" detection',
            '"software vulnerability" analysis',
        ],
        openalex_concepts=["Computer Security"],
        s2_fields=["Computer Science"],
    )),

    ("healthcare privacy", Intent(
        domain="cybersecurity", subdomain="privacy",
        arxiv_category="cs.CR",
        methods=["differential privacy", "federated learning", "encryption"],
        tasks=["privacy preservation", "data anonymization", "secure computation"],
        artifacts=["patient data", "health record", "medical dataset"],
        negative_constraints=["education"],
        query_expansions=[
            '"healthcare privacy" federated',
            '"medical data" privacy preserving',
            '"patient data" anonymization',
        ],
        openalex_concepts=["Computer Security", "Privacy"],
        s2_fields=["Computer Science"],
    )),

    ("security", Intent(
        domain="cybersecurity", subdomain="security",
        arxiv_category="cs.CR",
        methods=["cryptography", "intrusion detection"],
        tasks=["security analysis", "threat detection", "access control"],
        artifacts=["security protocol", "threat model"],
        negative_constraints=["education"],
        query_expansions=['"cybersecurity" detection', '"security" analysis automated'],
        openalex_concepts=["Computer Security"],
        s2_fields=["Computer Science"],
    )),

    ("hallucination", Intent(
        domain="natural_language_processing", subdomain="llm_reliability",
        arxiv_category="cs.CL",
        methods=["large language model", "retrieval augmented generation"],
        tasks=["hallucination detection", "factuality evaluation", "faithfulness"],
        artifacts=["LLM output", "generated text"],
        negative_constraints=["education"],
        query_expansions=[
            '"hallucination" LLM detection',
            '"factuality" language model',
            '"faithfulness" text generation',
        ],
        openalex_concepts=["Natural Language Processing", "Machine Learning"],
        s2_fields=["Computer Science"],
    )),

    ("large language model", Intent(
        domain="natural_language_processing", subdomain="large_language_models",
        arxiv_category="cs.CL",
        methods=["large language model", "transformer"],
        tasks=["text generation", "reasoning", "instruction following"],
        artifacts=["LLM", "prompt", "fine-tuned model"],
        negative_constraints=["education"],
        query_expansions=[
            '"large language model" evaluation',
            '"LLM" benchmark',
        ],
        openalex_concepts=["Natural Language Processing"],
        s2_fields=["Computer Science"],
    )),

    ("natural language", Intent(
        domain="natural_language_processing", subdomain="nlp",
        arxiv_category="cs.CL",
        methods=["NLP", "transformer"],
        tasks=["text classification", "information extraction", "sentiment analysis"],
        artifacts=["text corpus", "language model"],
        negative_constraints=["education"],
        query_expansions=['"natural language processing"', '"NLP" application'],
        openalex_concepts=["Natural Language Processing"],
        s2_fields=["Computer Science"],
    )),

    ("llm", Intent(
        domain="natural_language_processing", subdomain="large_language_models",
        arxiv_category="cs.CL",
        methods=["large language model"],
        tasks=["reasoning", "text generation", "evaluation"],
        artifacts=["LLM", "benchmark"],
        negative_constraints=["education"],
        query_expansions=['"LLM" evaluation', '"large language model" reasoning'],
        openalex_concepts=["Natural Language Processing"],
        s2_fields=["Computer Science"],
    )),

    ("deepfake detection", Intent(
        domain="computer_vision", subdomain="media_forensics",
        arxiv_category="cs.CV",
        methods=["deep learning", "temporal analysis", "GAN detection"],
        tasks=["deepfake detection", "media forensics", "face manipulation detection"],
        artifacts=["video", "face image", "detector model"],
        negative_constraints=["education"],
        query_expansions=[
            '"deepfake detection" temporal',
            '"face manipulation" detection',
            '"synthetic media" detection',
        ],
        openalex_concepts=["Computer Vision", "Image Processing"],
        s2_fields=["Computer Science"],
    )),

    ("object detection", Intent(
        domain="computer_vision", subdomain="object_detection",
        arxiv_category="cs.CV",
        methods=["deep learning", "YOLO", "transformer"],
        tasks=["object detection", "localization", "recognition"],
        artifacts=["bounding box", "detector model", "dataset"],
        negative_constraints=["education"],
        query_expansions=['"object detection" real-time', '"object detection" deep learning'],
        openalex_concepts=["Computer Vision"],
        s2_fields=["Computer Science"],
    )),

    ("computer vision", Intent(
        domain="computer_vision", subdomain="computer_vision",
        arxiv_category="cs.CV",
        methods=["deep learning", "convolutional neural network"],
        tasks=["image classification", "object detection", "segmentation"],
        artifacts=["image", "video", "model"],
        negative_constraints=["education"],
        query_expansions=['"computer vision" deep learning', '"image recognition"'],
        openalex_concepts=["Computer Vision"],
        s2_fields=["Computer Science"],
    )),

    ("image segmentation", Intent(
        domain="computer_vision", subdomain="image_segmentation",
        arxiv_category="eess.IV",
        methods=["deep learning", "semantic segmentation"],
        tasks=["image segmentation", "pixel classification"],
        artifacts=["segmentation mask", "image"],
        negative_constraints=["education"],
        query_expansions=['"image segmentation" semantic', '"segmentation" deep learning'],
        openalex_concepts=["Computer Vision", "Image Processing"],
        s2_fields=["Computer Science"],
    )),

    ("medical imaging", Intent(
        domain="computer_vision", subdomain="medical_imaging",
        arxiv_category="eess.IV",
        methods=["deep learning", "convolutional neural network"],
        tasks=["medical image analysis", "diagnosis", "segmentation"],
        artifacts=["MRI", "CT scan", "X-ray", "medical image"],
        negative_constraints=["education"],
        query_expansions=[
            '"medical imaging" deep learning',
            '"medical image" segmentation',
        ],
        openalex_concepts=["Medical Imaging", "Computer Vision"],
        s2_fields=["Computer Science", "Medicine"],
    )),

    ("neural network", Intent(
        domain="machine_learning", subdomain="neural_networks",
        arxiv_category="cs.LG",
        methods=["neural network", "deep learning"],
        tasks=["classification", "regression", "prediction"],
        artifacts=["model", "architecture"],
        negative_constraints=["education"],
        query_expansions=['"neural network" application training'],
        openalex_concepts=["Machine Learning", "Neural Networks"],
        s2_fields=["Computer Science"],
    )),

    ("transformer", Intent(
        domain="machine_learning", subdomain="transformers",
        arxiv_category="cs.LG",
        methods=["transformer", "attention mechanism"],
        tasks=["sequence modeling", "classification", "generation"],
        artifacts=["transformer model", "attention weights"],
        negative_constraints=["power", "electrical", "education"],
        query_expansions=['"transformer" attention mechanism', '"self-attention" model'],
        openalex_concepts=["Machine Learning", "Natural Language Processing"],
        s2_fields=["Computer Science"],
    )),

    ("database", Intent(
        domain="software_engineering", subdomain="database_systems",
        arxiv_category="cs.DB",
        methods=["query optimization", "indexing"],
        tasks=["database design", "query processing", "data management"],
        artifacts=["database schema", "query", "index"],
        negative_constraints=["education"],
        query_expansions=['"database" query optimization', '"database system" design'],
        openalex_concepts=["Database", "Software Engineering"],
        s2_fields=["Computer Science"],
    )),

    ("normalization", Intent(
        domain="software_engineering", subdomain="database_design",
        arxiv_category="cs.DB",
        methods=["formal methods"],
        tasks=["schema normalization", "data normalization", "database design"],
        artifacts=["database schema", "normal form"],
        negative_constraints=["education", "signal normalization", "batch normalization", "layer normalization"],
        query_expansions=[
            '"database normalization" schema',
            '"normal form" database',
        ],
        openalex_concepts=["Database"],
        s2_fields=["Computer Science"],
    )),

    ("ontology", Intent(
        domain="artificial_intelligence", subdomain="knowledge_representation",
        arxiv_category="cs.AI",
        methods=["ontology engineering", "description logic"],
        tasks=["knowledge representation", "ontology alignment", "reasoning"],
        artifacts=["ontology", "knowledge base"],
        negative_constraints=["education"],
        query_expansions=['"ontology" knowledge representation', '"ontology" alignment'],
        openalex_concepts=["Artificial Intelligence", "Knowledge Representation"],
        s2_fields=["Computer Science"],
    )),

    ("robotics", Intent(
        domain="robotics", subdomain="robotics",
        arxiv_category="cs.RO",
        methods=["motion planning", "control"],
        tasks=["robot navigation", "manipulation", "autonomous operation"],
        artifacts=["robot", "trajectory", "controller"],
        negative_constraints=["education"],
        query_expansions=['"robotics" autonomous', '"robot" navigation planning'],
        openalex_concepts=["Robotics"],
        s2_fields=["Computer Science"],
    )),

    ("cryptograph", Intent(
        domain="cybersecurity", subdomain="cryptography",
        arxiv_category="cs.CR",
        methods=["cryptography", "encryption"],
        tasks=["secure communication", "data encryption", "key management"],
        artifacts=["cipher", "protocol", "key"],
        negative_constraints=["education"],
        query_expansions=['"cryptography" encryption secure', '"cryptographic protocol"'],
        openalex_concepts=["Computer Security", "Cryptography"],
        s2_fields=["Computer Science"],
    )),

    ("quantum", Intent(
        domain="quantum_computing", subdomain="quantum_computing",
        arxiv_category="quant-ph",
        methods=["quantum algorithm", "quantum circuit"],
        tasks=["quantum computation", "quantum optimization"],
        artifacts=["qubit", "quantum circuit", "quantum algorithm"],
        negative_constraints=["education"],
        query_expansions=['"quantum computing" algorithm', '"quantum circuit" optimization'],
        openalex_concepts=["Quantum Computing"],
        s2_fields=["Physics", "Computer Science"],
    )),

    ("optimization", Intent(
        domain="mathematics", subdomain="optimization",
        arxiv_category="math.OC",
        methods=["mathematical optimization", "metaheuristic"],
        tasks=["optimization", "constraint satisfaction", "search"],
        artifacts=["objective function", "solution", "algorithm"],
        negative_constraints=["education"],
        query_expansions=['"optimization algorithm"', '"metaheuristic" optimization'],
        openalex_concepts=["Mathematics", "Operations Research"],
        s2_fields=["Mathematics", "Computer Science"],
    )),

    ("signal processing", Intent(
        domain="electrical_engineering", subdomain="signal_processing",
        arxiv_category="eess.SP",
        methods=["signal processing", "filtering"],
        tasks=["signal analysis", "noise reduction", "feature extraction"],
        artifacts=["signal", "filter", "spectrum"],
        negative_constraints=["education"],
        query_expansions=['"signal processing" analysis', '"digital signal processing"'],
        openalex_concepts=["Signal Processing", "Electrical Engineering"],
        s2_fields=["Engineering"],
    )),

    ("bioinformatics", Intent(
        domain="bioinformatics", subdomain="bioinformatics",
        arxiv_category="q-bio",
        methods=["sequence alignment", "machine learning"],
        tasks=["gene analysis", "protein structure prediction", "genomics"],
        artifacts=["genome", "protein sequence", "biological network"],
        negative_constraints=["education"],
        query_expansions=['"bioinformatics" genomics', '"sequence analysis" biological'],
        openalex_concepts=["Bioinformatics", "Biology"],
        s2_fields=["Biology", "Computer Science"],
    )),
]

_FALLBACK_INTENT = Intent(
    domain="computer_science", subdomain="general",
    arxiv_category="",
    methods=[], tasks=[], artifacts=[],
    negative_constraints=["education", "pedagogy", "curriculum", "classroom", "student grading"],
    query_expansions=[],
    openalex_concepts=["Computer Science"],
    s2_fields=["Computer Science"],
)


def extract_domain_context(query: str, groq_client=None, model_name: str = "") -> Dict:
    """
    Pure deterministic intent extraction. No LLM. No hallucination.
    Returns a dict compatible with the existing pipeline interface,
    plus new Intent fields that Layer 1 uses for constrained retrieval.
    """
    q = query.lower().strip()

    if not q:
        return _intent_to_dict(_FALLBACK_INTENT)

    import re as _re
    best_match = None
    best_score = 0

    for phrase, intent in _INTENT_RULES:
        pattern = r"\b" + _re.escape(phrase) + r"\b"
        if _re.search(pattern, q):
            score = len(phrase)  # longer phrase = more specific
            if score > best_score:
                best_match = intent
                best_score = score

    if best_match:
        logger.info(
            "[Layer0] Matched '%s' (score=%d) → domain=%s subdomain=%s category=%s",
            best_match.subdomain, best_score,
            best_match.domain, best_match.subdomain, best_match.arxiv_category,
        )
        return _intent_to_dict(best_match)

    logger.info("[Layer0] No match — fallback to cs.AI")
    return _intent_to_dict(_FALLBACK_INTENT)


def _intent_to_dict(intent: Intent) -> Dict:
    """
    Converts Intent to the dict format the pipeline expects.
    Backward-compatible: all old keys still present.
    New keys added for Layer 1 constrained retrieval.
    """
    return {
        # Backward-compatible keys (pipeline already uses these)
        "domain":                    intent.domain,
        "methodology_class":         intent.methods[0] if intent.methods else "other",
        "problem_type":              intent.tasks[0] if intent.tasks else "other",
        "key_concepts":              intent.tasks + intent.artifacts,
        "domain_anchors":            intent.tasks,      # Layer 2.5 uses this
        "subdomains":                [intent.subdomain],
        "negative_constraints":      intent.negative_constraints,
        "arxiv_category":            intent.arxiv_category,
        "query_is_valid":            True,
        "interpretation_confidence": 1.0,
        "anchor_mode":               "strict",

        # New keys for Layer 1 constrained retrieval
        "intent_domain":       intent.domain,
        "intent_subdomain":    intent.subdomain,
        "intent_methods":      intent.methods,
        "intent_tasks":        intent.tasks,
        "intent_artifacts":    intent.artifacts,
        "query_expansions":    intent.query_expansions,
        "openalex_concepts":   intent.openalex_concepts,
        "s2_fields":           intent.s2_fields,
    }


__all__ = ["extract_domain_context", "Intent"]
