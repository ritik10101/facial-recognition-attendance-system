# ============================================================
# test_db.py
# Tests the database connection and table setup
# ============================================================

import os
import pytest
from db import get_conn, test_connection as db_test_connection  # ✅ alias to avoid recursion


def save_db_error_snapshot(message: str):
    """Save DB failure messages to a log file for debugging."""
    os.makedirs("test_artifacts", exist_ok=True)
    path = os.path.join("test_artifacts", "db_connection_error.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(f"⚠️ Saved DB error message to {path}")


def test_db_connection():
    """Verify that MySQL connection works correctly."""
    try:
        ok, msg = db_test_connection()  # ✅ call db.py version
    except Exception as e:
        save_db_error_snapshot(f"DB test raised exception: {e}")
        pytest.fail(f"❌ DB connection test raised exception: {e}")

    if not ok:
        save_db_error_snapshot(f"DB connection failed: {msg}")
    assert ok, f"❌ Database connection failed: {msg or 'Unknown error'}"


def test_get_conn_and_cursor():
    """Ensure that get_conn() returns a working connection and cursor."""
    try:
        conn = get_conn()
        assert conn is not None, "❌ get_conn() returned None"

        cursor = conn.cursor()
        cursor.execute("SELECT DATABASE()")
        dbname = cursor.fetchone()[0]
        print(f"✅ Connected to database: {dbname}")

        cursor.close()
        conn.close()
    except Exception as e:
        save_db_error_snapshot(f"get_conn() failed: {e}")
        pytest.fail(f"❌ get_conn() failed: {e}")
