"""SQLite result store (plan 2, stage 8). Rows are written in a fixed order so reruns match byte for byte."""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Iterable, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id INTEGER PRIMARY KEY, domain TEXT, degree INTEGER, seed INTEGER,
    features TEXT, n_rows INTEGER, status TEXT);
CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY, task_id INTEGER, kind TEXT, coeffs TEXT, features TEXT,
    tight TEXT, min_slack TEXT, status TEXT, score REAL, canon TEXT, known TEXT,
    is_new INTEGER, routine INTEGER, removed_by TEXT);
"""


def open_db(path: str) -> sqlite3.Connection:
    """Create (replacing any old file) and return a connection with the schema."""
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def write_task(conn: sqlite3.Connection, task_id: int, task: dict, rels: Iterable[dict]) -> None:
    """Insert one task and its relation rows."""
    conn.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?)",
                 (task_id, task["domain"], task["degree"], task["seed"],
                  json.dumps(task["features"]), task["n_rows"], task["status"]))
    for r in rels:
        conn.execute("INSERT INTO relations (task_id, kind, coeffs, features, tight, min_slack, status, score, "
                     "canon, known, is_new, routine, removed_by) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (task_id, r["kind"], json.dumps(r["coeffs"]), json.dumps(r["features"]),
                      json.dumps(r["tight"]), r["min_slack"], r["status"], r["score"], r["canon"],
                      r["known"], int(r["is_new"]), int(r["routine"]), r["removed_by"]))
    conn.commit()


def dump(conn: sqlite3.Connection) -> Sequence[tuple]:
    """All rows in a fixed order, for comparing two runs."""
    return (conn.execute("SELECT * FROM tasks ORDER BY task_id").fetchall(),
            conn.execute("SELECT * FROM relations ORDER BY id").fetchall())
