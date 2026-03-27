import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "history.db")

def _conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                question  TEXT NOT NULL,
                answer    TEXT NOT NULL,
                created_at DATETIME DEFAULT (datetime('now', 'localtime'))
            )
        """)

def save_entry(question: str, answer: str):
    with _conn() as con:
        con.execute(
            "INSERT INTO history (question, answer) VALUES (?, ?)",
            (question, answer)
        )

def load_history() -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT question, answer FROM history ORDER BY id ASC"
        ).fetchall()
    return [{"question": q, "answer": a} for q, a in rows]
