# NORA — NLP-Oriented Research Assistant

> An end-to-end AI pipeline that reads scientific papers, finds research gaps, and generates grounded research ideas.

![Python](https://img.shields.io/badge/Python-83%25-blue?logo=python) ![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green?logo=fastapi) ![LLM](https://img.shields.io/badge/LLM-Groq%20API-orange)

---

## What It Does

Literature review is slow, manual, and easy to get wrong. NORA automates it:

- **Ingests** papers via keyword search (Semantic Scholar, OpenAlex, ArXiv) or direct PDF upload
- **Extracts** structured information — problems, objectives, methods, datasets, results, limitations, future work
- **Identifies** research gaps using faithful quote-based evidence (no hallucination — every gap traces to exact author sentences)
- **Synthesizes** consensus gaps across multiple papers
- **Generates** concrete methodology recommendations grounded in the literature

---
## Tech Stack
 
| Layer | Technology |
|---|---|
| Frontend | React.js |
| Backend API | FastAPI (Python) |
| Embeddings | SentenceTransformers (BGE) |
| LLM | Groq API |
| PDF Parsing | pdfminer |
| Database | SQLite |
| Paper Sources | Semantic Scholar · OpenAlex · ArXiv |
 
---
   
## Architecture
 
```
Input: Topic Keywords  ──OR──  PDF Upload
              ↓                        ↓
         Layer 0                  PDF Parser
     (Paper Fetcher)             (pdfminer)
              ↓                        ↓
         Layer 1               ────────┘
    (Query Processor)
              ↓
         Layer 2
      (Synthesizer — cross-paper gap consensus)
              ↓
         Layer 3
    (Gap Extractor — per-paper, LLM + rule-based)
              ↓
         Layer 4
   (Idea Generator — grounded research proposals)
```
 ## Setup
 
```bash
git clone https://github.com/mehakkfarmann/NORA-Natural-Language-Processing-Oriented-Research-Assistant-
cd NORA-Natural-Language-Processing-Oriented-Research-Assistant-
pip install -r requirements.txt
python run.py
```
 
---
 
## Project Status
 
Core pipeline functional. Actively refining gap extraction precision and idea generation grounding.
 
---
 
*Final Year Project — Computer Science, University of Agriculture Faisalabad*
 
