"""
MySQL connector for Face Recognition Attendance System.

This module:
 - Creates a small MySQL connection pool (if possible)
 - Wraps connection.cursor so returned cursors are buffered by default
 - Provides get_conn() and test_connection()
"""

import os
import mysql.connector
from mysql.connector import pooling
from mysql.connector import Error as MySQLError
from functools import wraps

# ---------------- CONFIG ----------------
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "5233"),   # <-- Change if needed / use env vars
    "database": os.getenv("DB_NAME", "face_recognition_db"),
    # autocommit left False so callers explicitly commit when needed
    "autocommit": False,
}

_POOL = None


def _init_pool():
    """Initialize a small MySQL connection pool (fallback to direct)."""
    global _POOL
    try:
        _POOL = pooling.MySQLConnectionPool(
            pool_name="face_attendance_pool",
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            **DB_CONFIG,
        )
        print("[db.py] ✅ MySQL connection pool initialized.")
    except Exception as e:
        _POOL = None
        # non-fatal: app will fall back to direct connections
        print(f"[db.py] ⚠️ Pool initialization failed, using direct connections: {e}")


_init_pool()


def _wrap_conn_with_buffered_cursor(conn):
    """
    Monkeypatch conn.cursor so that:
      - cursor() by default returns buffered cursors (avoids 'Unread result found')
      - supports cursor(dictionary=True) and other kwargs as usual
    Avoids double-wrapping by setting an attribute flag on the connection.
    """
    if getattr(conn, "_cursor_wrapped_by_dbpy", False):
        return conn

    orig_cursor = conn.cursor

    @wraps(orig_cursor)
    def cursor_wrapper(*args, **kwargs):
        # If code passed buffered explicitly, respect it; otherwise default to True.
        if "buffered" not in kwargs:
            kwargs["buffered"] = True
        return orig_cursor(*args, **kwargs)

    # Replace the cursor attribute only on this connection instance
    try:
        conn.cursor = cursor_wrapper
        conn._cursor_wrapped_by_dbpy = True
    except Exception:
        # if the connection object is implemented in C or is immutable,
        # fall back to returning the original connection unchanged.
        pass

    return conn


def _direct_connect():
    """Create a direct MySQL connection and wrap it for buffered cursors."""
    conn = mysql.connector.connect(**DB_CONFIG)
    return _wrap_conn_with_buffered_cursor(conn)


def get_conn():
    """
    Get a MySQL connection. Uses the pool if available, otherwise creates a direct connection.
    Returned connection's cursor() will be buffered by default.
    Caller must close the connection (conn.close()) when done.
    """
    global _POOL
    if _POOL:
        try:
            conn = _POOL.get_connection()
            return _wrap_conn_with_buffered_cursor(conn)
        except Exception as e:
            # Pool failed -> fallback to direct connect
            print(f"[db.py] ⚠️ Pool get_connection() failed, using direct connect: {e}")
    return _direct_connect()


def test_connection():
    """Simple connection test. Returns (ok: bool, message: str)."""
    try:
        conn = get_conn()
        cur = conn.cursor()  # this cursor is buffered by default now
        cur.execute("SELECT 1")
        _ = cur.fetchone()
        cur.close()
        conn.close()
        return True, "✅ MySQL connection OK"
    except Exception as e:
        return False, f"❌ DB connection failed: {e}"


def close_pool():
    """
    Close pool connections if possible. Useful for graceful shutdowns or tests.
    Note: mysql-connector's pool objects don't expose an explicit close API; this helper
    attempts to close any connections the pool currently holds by borrowing them and closing.
    """
    global _POOL
    if not _POOL:
        return
    try:
        # Try to empty the pool by fetching available connections and closing them.
        # Not all pool implementations expose internals; this is a best-effort attempt.
        while True:
            try:
                conn = _POOL.get_connection()
            except Exception:
                break
            try:
                conn.close()
            except Exception:
                pass
        _POOL = None
        print("[db.py] Pool closed.")
    except Exception as e:
        print(f"[db.py] Failed to close pool cleanly: {e}")
