from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("models/mlops.db")


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS uploaded_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                class_name TEXT NOT NULL,
                status TEXT NOT NULL,
                uploaded_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                latency_ms REAL NOT NULL,
                predicted_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def log_uploaded_file(file_name: str, file_path: str, class_name: str, status: str, uploaded_at: str) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO uploaded_files (file_name, file_path, class_name, status, uploaded_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (file_name, file_path, class_name, status, uploaded_at),
        )
        conn.commit()


def log_prediction(file_name: str, predicted_class: str, confidence: float, latency_ms: float, predicted_at: str) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO prediction_logs (file_name, predicted_class, confidence, latency_ms, predicted_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (file_name, predicted_class, confidence, latency_ms, predicted_at),
        )
        conn.commit()


def mark_uploaded_file_preprocessed(file_path: str) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE uploaded_files
            SET status = 'preprocessed'
            WHERE file_path = ?
            """,
            (file_path,),
        )
        conn.commit()


def mark_uploaded_file_retrained(file_path: str) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE uploaded_files
            SET status = 'retrained'
            WHERE file_path = ?
            """,
            (file_path,),
        )
        conn.commit()
