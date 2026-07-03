"""
Structured storage: SQLite database holding submissions (which double as
the audit log) and appeals. Chosen over print()/flat files because the
project requires structured, queryable audit log entries with at least
timestamp, content ID, attribution, confidence, both signal scores, and
appeal status -- SQLite gives us that with minimal setup and no external
dependency beyond Python's stdlib.
"""

import datetime
import sqlite3
import uuid
from contextlib import contextmanager

DB_PATH = "provenance_guard.db"


def init_db():
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS submissions (
                content_id TEXT PRIMARY KEY,
                creator_id TEXT,
                text TEXT,
                timestamp TEXT,
                llm_score REAL,
                llm_reasoning TEXT,
                stylometric_score REAL,
                trained_ml_score REAL,
                confidence REAL,
                attribution TEXT,
                label_text TEXT,
                status TEXT DEFAULT 'classified',
                appeal_reasoning TEXT,
                appeal_timestamp TEXT
            )
            """
        )
        conn.commit()


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def create_submission(
    creator_id: str,
    text: str,
    llm_score: float,
    llm_reasoning: str,
    stylometric_score: float,
    trained_ml_score: float,
    confidence: float,
    attribution: str,
    label_text: str,
) -> str:
    content_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO submissions (
                content_id, creator_id, text, timestamp,
                llm_score, llm_reasoning, stylometric_score, trained_ml_score,
                confidence, attribution, label_text, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'classified')
            """,
            (
                content_id,
                creator_id,
                text,
                timestamp,
                llm_score,
                llm_reasoning,
                stylometric_score,
                trained_ml_score,
                confidence,
                attribution,
                label_text,
            ),
        )
        conn.commit()

    return content_id


def get_submission(content_id: str):
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM submissions WHERE content_id = ?", (content_id,)
        ).fetchone()
        return dict(row) if row else None


def record_appeal(content_id: str, creator_reasoning: str) -> bool:
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE submissions
            SET status = 'under_review', appeal_reasoning = ?, appeal_timestamp = ?
            WHERE content_id = ?
            """,
            (creator_reasoning, timestamp, content_id),
        )
        conn.commit()
        return cur.rowcount > 0


def get_recent_log(limit: int = 20):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM submissions ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
