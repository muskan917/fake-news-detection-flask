"""Database module (SQLite)."""
import os, sqlite3
from werkzeug.security import generate_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fakenews.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 0 CHECK (is_admin IN (0,1)),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS model_versions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    rows_used INTEGER, accuracy REAL, precision_s REAL, recall_s REAL, f1 REAL,
    trained_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS news_checks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    news_text TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    check_id INTEGER NOT NULL UNIQUE REFERENCES news_checks(id),
    label TEXT NOT NULL CHECK (label IN ('FAKE','REAL')),
    confidence REAL NOT NULL,
    model_version_id INTEGER REFERENCES model_versions(id));
CREATE TABLE IF NOT EXISTS feedback(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    check_id INTEGER NOT NULL REFERENCES news_checks(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    is_correct INTEGER NOT NULL CHECK (is_correct IN (0,1)),
    comment TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
"""


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript(SCHEMA)
    if not conn.execute("SELECT 1 FROM users WHERE is_admin=1").fetchone():
        conn.execute(
            "INSERT INTO users(username,email,password_hash,is_admin) VALUES(?,?,?,1)",
            ("admin", "admin@example.com", generate_password_hash("admin123")))
    conn.commit()
    conn.close()
