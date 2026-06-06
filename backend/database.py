"""SQLite persistence for runs, gaps, and ideas."""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("data/nora.db")
DB_PATH.parent.mkdir(exist_ok=True)
_db_lock = threading.Lock()

@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize all tables."""
    with _db_lock:
        with get_db() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                progress TEXT DEFAULT '0',
                result_json TEXT,
                error TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
            CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);

            CREATE TABLE IF NOT EXISTS gaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                paper_id TEXT,
                paper_title TEXT,
                gap_description TEXT NOT NULL,
                gap_type TEXT,
                section_type TEXT,
                evidence_quote TEXT,
                extraction_confidence REAL,
                research_significance REAL,
                source_paper_id TEXT,
                cross_paper_support INTEGER DEFAULT 0,
                domain_alignment REAL,
                gap_quality TEXT,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_gaps_run_id ON gaps(run_id);
            CREATE INDEX IF NOT EXISTS idx_gaps_type ON gaps(gap_type);
            CREATE INDEX IF NOT EXISTS idx_gaps_quality ON gaps(gap_quality);

            CREATE TABLE IF NOT EXISTS ideas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                gap_index INTEGER,
                gap_description TEXT,
                gap_type TEXT,
                title TEXT,
                idea TEXT,
                methodology TEXT,
                why_feasible_now TEXT,
                feasibility REAL,
                novelty_score REAL,
                idea_score REAL,
                approach_track TEXT,
                primary_method TEXT,
                dataset TEXT,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ideas_run_id ON ideas(run_id);
            CREATE INDEX IF NOT EXISTS idx_ideas_novelty ON ideas(novelty_score);
            """)
            conn.commit()
            logger.info("[DB] Schema initialised (WAL mode active)")

def create_run(run_id: str, query: str):
    with _db_lock:
        with get_db() as conn:
            conn.execute(
                """INSERT INTO runs (id, query, status, progress, created_at, updated_at)
                   VALUES (?, ?, 'pending', '0', ?, ?)""",
                (run_id, query, datetime.now().isoformat(), datetime.now().isoformat()),
            )
            conn.commit()
    logger.debug("[DB] Created run %s", run_id)

def update_run_status(run_id: str, status: str, progress: str = None,
                      result_json: str = None, error: str = None):
    with _db_lock:
        with get_db() as conn:
            updates, params = [], []
            if progress is not None: updates.append("progress = ?"); params.append(progress)
            if result_json is not None: updates.append("result_json = ?"); params.append(result_json)
            if error is not None: updates.append("error = ?"); params.append(error)
            updates.append("status = ?"); params.append(status)
            updates.append("updated_at = ?"); params.append(datetime.now().isoformat())
            params.append(run_id)
            conn.execute(f"UPDATE runs SET {', '.join(updates)} WHERE id = ?", params)
            conn.commit()

def save_run_result(run_id: str, gaps: List[Dict], ideas: List[Dict], result_summary: Dict):
    """Write results to relational tables + JSON blob."""
    _validate_gap_fields(gaps, run_id)
    now = datetime.now().isoformat()
    with _db_lock:
        with get_db() as conn:
            # Clear old results for this run_id first (prevents stale data)
            conn.execute("DELETE FROM gaps WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM ideas WHERE run_id = ?", (run_id,))
            
            # Insert gaps
            for g in gaps:
                conn.execute("""
                    INSERT INTO gaps (run_id, paper_id, paper_title, gap_description, gap_type,
                                    section_type, evidence_quote, extraction_confidence,
                                    research_significance, source_paper_id, cross_paper_support,
                                    domain_alignment, gap_quality, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id, g.get("paper_id"), g.get("paper_title"),
                    g.get("gap_description", ""), g.get("gap_type"),
                    g.get("section_type", "unknown"), g.get("evidence_quote", ""),
                    g.get("extraction_confidence", 0), g.get("research_significance", 0),
                    g.get("source_paper_id", ""), g.get("cross_paper_support", 0),
                    g.get("domain_alignment", 0), g.get("gap_quality", "acceptable"), now
                ))
            
            # Insert ideas
            for i in ideas:
                conn.execute("""
                    INSERT INTO ideas (run_id, gap_index, gap_description, gap_type, title, idea,
                                     methodology, why_feasible_now, feasibility, novelty_score,
                                     idea_score, approach_track, primary_method, dataset, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id, i.get("gap_index"), i.get("gap_description"),
                    i.get("gap_type"), i.get("title"), i.get("description"),
                    i.get("methodology"), i.get("why_feasible_now"),
                    i.get("feasibility"), i.get("novelty_score"),
                    i.get("idea_score"), i.get("approach_track"),
                    i.get("primary_method"), i.get("dataset"), now
                ))
            
            # Update run with JSON summary (backward compat)
            conn.execute(
                "UPDATE runs SET result_json = ?, status = 'completed', updated_at = ? WHERE id = ?",
                (json.dumps(result_summary), now, run_id)
            )
            conn.commit()
    logger.info("[DB] Saved run %s | gaps=%d | ideas=%d", run_id, len(gaps), len(ideas))

def get_run(run_id: str) -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row: return None
        result = dict(row)
        if result.get("result_json"):
            try: result["result_json"] = json.loads(result["result_json"])
            except: pass
        return result

def get_run_gaps(run_id: str) -> List[Dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM gaps WHERE run_id = ? ORDER BY research_significance DESC", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

def get_run_ideas(run_id: str) -> List[Dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM ideas WHERE run_id = ? ORDER BY idea_score DESC", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

def _validate_gap_fields(gaps: List[Dict], run_id: str):
    """Warn if gaps use old field names instead of canonical 'gap_description'."""
    if not gaps: return
    for g in gaps:
        if g.get("description") and not g.get("gap_description"):
            logger.warning("[DB] run=%s: gap uses 'description' instead of 'gap_description'", run_id)
        if g.get("gap") and not g.get("gap_description"):
            logger.warning("[DB] run=%s: gap uses 'gap' instead of 'gap_description'", run_id)

def cleanup_old_runs(days: int = 7):
    with _db_lock:
        with get_db() as conn:
            cutoff = datetime.now().timestamp() - days * 86400
            conn.execute("DELETE FROM runs WHERE created_at < datetime(?, 'unixepoch')", (cutoff,))
            conn.commit()
    logger.info("[DB] Cleaned up runs older than %d days", days)