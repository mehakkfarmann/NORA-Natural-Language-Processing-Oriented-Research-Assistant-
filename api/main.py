"""
NORA — FastAPI backend.  Accepts topic queries, custom papers, and PDF uploads.
"""

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.database import get_db, init_db, create_run, update_run_status, get_run
from backend.pipeline.orchestrator import run_full_pipeline, run_custom_pipeline, run_pdf_pipeline

logger = logging.getLogger(__name__)

BASE_DIR     = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

PDF_MAX_BYTES = 20 * 1024 * 1024  # max upload size



@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _cleanup_stale_runs()
    logger.info("[Startup] NORA API ready | Docs: http://localhost:8000/docs")
    yield
    logger.info("[Shutdown] NORA API shutting down")




app = FastAPI(
    title="NORA API",
    description="NLP-Oriented Research Assistant Backend",
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.debug("[Init] BASE_DIR=%s", BASE_DIR)
logger.debug("[Init] FRONTEND_DIR=%s | exists=%s", FRONTEND_DIR, FRONTEND_DIR.exists())

if (FRONTEND_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")
    logger.info("[Init] Static files mounted from %s", FRONTEND_DIR / "static")
else:
    logger.warning("[Init] static/ directory not found — frontend assets will not be served")




class QueryRequest(BaseModel):
    query: str
    focus: Optional[str] = None


class CustomPapersRequest(BaseModel):
    papers: list[dict]             # [{title, abstract, year}]
    query:  str = "research gap analysis"
    focus:  Optional[str] = None




@app.get("/")
async def serve_frontend():
    idx = FRONTEND_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse(
        status_code=404,
        content={
            "error":        "frontend/index.html not found",
            "searched_path": str(idx),
            "base_dir":     str(BASE_DIR),
            "frontend_dir": str(FRONTEND_DIR),
        },
    )




@app.post("/api/run")
async def start_run(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Start a new topic-search pipeline run.
    Request body: { "query": "...", "focus": "..." (optional) }
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")
    if len(query) > 500:
        raise HTTPException(status_code=400, detail="Query too long (max 500 characters)")

    run_id = str(uuid.uuid4())
    create_run(run_id, query)
    background_tasks.add_task(run_full_pipeline, run_id, query, request.focus)

    logger.info("[API] Topic run started | run_id=%s | query=%s", run_id, query[:60])
    return {"run_id": run_id, "status": "processing", "message": "Pipeline started"}


@app.post("/api/run-custom")
async def start_custom_run(request: CustomPapersRequest, background_tasks: BackgroundTasks):
    """
    Run gap analysis on user-supplied paper abstracts.
    Request body: { "papers": [{title, abstract, year}], "query": "...", "focus": "..." }
    """
    if not request.papers:
        raise HTTPException(status_code=400, detail="No papers provided")
    if len(request.papers) > 5:
        raise HTTPException(status_code=400, detail="Max 5 papers allowed")

    run_id = str(uuid.uuid4())
    create_run(run_id, request.query)
    background_tasks.add_task(
        run_custom_pipeline, run_id, request.papers, request.query, request.focus
    )

    logger.info("[API] Custom run started | run_id=%s | papers=%d", run_id, len(request.papers))
    return {"run_id": run_id, "status": "processing", "message": "Pipeline started"}


@app.post("/api/run-pdf")
async def start_pdf_run(
    background_tasks: BackgroundTasks,
    file:  UploadFile = File(...),
    focus: Optional[str] = Form(None),
):
    """
    Upload a PDF, extract full text, run gap extraction and idea generation.

    Accepts multipart/form-data:
      - file:  PDF file (required, max 20MB)
      - focus: research focus string (optional)

    Returns: { "run_id": "uuid", "status": "processing", "message": "..." }
    """

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF (.pdf)")

    raw = await file.read()
    if len(raw) > PDF_MAX_BYTES:
        raise HTTPException(status_code=413, detail="PDF too large — max 20MB")
    if len(raw) < 100:
        raise HTTPException(status_code=400, detail="File appears to be empty")

    # extract text with fallback chain
    pdf_text = None
    _extraction_errors = []
    import io as _io

    try:

        from pdfminer.high_level import extract_text as _pdfminer_extract
        pdf_text = _pdfminer_extract(_io.BytesIO(raw))
        logger.info("[API/pdf] pdfminer extracted %d chars from '%s'", len(pdf_text or ""), file.filename)
    except Exception as e:
        _extraction_errors.append(f"pdfminer: {e}")

    if not pdf_text or len(pdf_text.strip()) < 50:
        try:
            import pdfplumber
            with pdfplumber.open(_io.BytesIO(raw)) as _pdf:
                _pages = [p.extract_text() or "" for p in _pdf.pages]
                pdf_text = "\n\n".join(_pages)
            logger.info("[API/pdf] pdfplumber extracted %d chars from '%s'", len(pdf_text or ""), file.filename)
        except Exception as e:
            _extraction_errors.append(f"pdfplumber: {e}")

    if not pdf_text or len(pdf_text.strip()) < 50:
        try:
            import fitz
            _doc = fitz.open(stream=raw, filetype="pdf")
            pdf_text = "\n\n".join(p.get_text() for p in _doc)
            logger.info("[API/pdf] PyMuPDF extracted %d chars from '%s'", len(pdf_text or ""), file.filename)
        except Exception as e:
            _extraction_errors.append(f"PyMuPDF: {e}")

    if not pdf_text or len(pdf_text.strip()) < 50:
        try:
            import PyPDF2
            _reader = PyPDF2.PdfReader(_io.BytesIO(raw))
            pdf_text = "\n\n".join(p.extract_text() or "" for p in _reader.pages)
            logger.info("[API/pdf] PyPDF2 extracted %d chars from '%s'", len(pdf_text or ""), file.filename)
        except Exception as e:
            _extraction_errors.append(f"PyPDF2: {e}")

    if not pdf_text or len(pdf_text.strip()) < 50:
        _detail = "; ".join(_extraction_errors) if _extraction_errors else "No extractable text found"
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract text from PDF: {_detail}"
        )



    query  = f"PDF: {file.filename}"
    run_id = str(uuid.uuid4())
    create_run(run_id, query)

    background_tasks.add_task(
        run_pdf_pipeline,
        run_id,
        pdf_text,
        file.filename,
        focus,
    )

    logger.info(
        "[API] PDF run started | run_id=%s | file=%s | chars=%d",
        run_id, file.filename, len(pdf_text),
    )
    return {
        "run_id":  run_id,
        "status":  "processing",
        "message": f"Extracting gaps from '{file.filename}'",
    }


@app.get("/api/status/{run_id}")
async def get_status(run_id: str):
    """
    Poll for pipeline status.

    Response schema:
        {
            "id":          "uuid",
            "status":      "pending" | "running" | "completed" | "failed",
            "progress":    "0"–"100",
            "result_json": { ... } | null,
            "error":       "..." | null
        }

    """
    data = get_run(run_id)
    if not data:
        raise HTTPException(status_code=404, detail="Run not found")

    if data.get("progress") is None:
        data["progress"] = "0"

    if isinstance(data.get("result_json"), str):
        try:
            data["result_json"] = json.loads(data["result_json"])
        except json.JSONDecodeError:
            logger.warning("[API] Failed to parse result_json for run %s", run_id)
            data["result_json"] = None
            if not data.get("error"):
                data["error"] = "Result parsing failed"

    if data.get("status") == "done":
        data["status"] = "completed"

    return data


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "nora-api", "version": "2.0.0"}




def _cleanup_stale_runs():
    """
    On startup, mark runs stuck in 'running' for more than 2 hours as 'failed'.
    """
    stale_cutoff = datetime.utcnow() - timedelta(hours=2)
    cutoff_str   = stale_cutoff.isoformat()

    try:
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE status = 'running' AND created_at < ?",
                (cutoff_str,),
            )
            stale_count = cursor.fetchone()[0]

            if stale_count > 0:
                conn.execute(
                    """
                    UPDATE runs
                    SET
                        status     = 'failed',
                        error      = 'Pipeline was interrupted — server restarted while run '
                                     'was active. Please submit your query again.',
                        updated_at = ?
                    WHERE
                        status     = 'running'
                        AND created_at < ?
                    """,
                    (datetime.utcnow().isoformat(), cutoff_str),
                )
                conn.commit()
                logger.warning("[Startup] Cleaned up %d stale run(s)", stale_count)
            else:
                logger.info("[Startup] No stale runs found")

    except Exception as exc:
        logger.info("[Startup] Stale run cleanup skipped: %s", exc)




@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("[API] Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error":  "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG") == "1" else None,
        },
    )