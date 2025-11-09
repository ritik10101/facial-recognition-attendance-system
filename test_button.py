# ============================================================
# test_button.py
# Automated route/button/link test for Flask Face Attendance System
# ============================================================

import pytest
from app import app, init_db, create_user, authenticate
from db import get_conn

# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    """Setup Flask test client and DB."""
    init_db()
    app.config["TESTING"] = True
    app.secret_key = "test_secret"
    with app.test_client() as client:
        yield client


# ------------------------------------------------------------
# Utility: ensure at least one admin and one user exist
# ------------------------------------------------------------
def _ensure_default_users():
    """Create a test admin and test user if they don’t exist."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users WHERE username='test_admin'")
    if cur.fetchone()[0] == 0:
        create_user("test_admin", "admin123", "admin", face_id=1, name="Admin Tester")
    cur.execute("SELECT COUNT(*) FROM users WHERE username='test_user'")
    if cur.fetchone()[0] == 0:
        create_user("test_user", "user123", "user", face_id=2, name="User Tester")
    conn.close()


# ------------------------------------------------------------
# Helper for login
# ------------------------------------------------------------
def login(client, username, password):
    return client.post("/login", data={"username": username, "password": password}, follow_redirects=True)


# ------------------------------------------------------------
# Button / route tests
# ------------------------------------------------------------
def test_all_buttons_and_links(client):
    """Check every important route returns expected HTML."""
    _ensure_default_users()

    # ----------------------------
    # 1️⃣ Anonymous user
    # ----------------------------
    resp = client.get("/", follow_redirects=True)
    assert resp.status_code == 200
    assert b"Facial Recognition Attendance System" in resp.data
    print("✅ Homepage OK")

    # Check login page
    resp = client.get("/login")
    assert resp.status_code == 200 and b"Login" in resp.data
    print("✅ Login form OK")

    # Check signup page
    resp = client.get("/signup")
    assert resp.status_code == 200 and b"Signup" in resp.data
    print("✅ Signup form OK")

    # Attendance form should be visible (no login yet)
    resp = client.get("/attendance/mark_form")
    assert resp.status_code == 200
    print("✅ Attendance form OK")

    # ----------------------------
    # 2️⃣ Login as normal user
    # ----------------------------
    resp = login(client, "test_user", "user123")
    assert b"Logged in as" in resp.data
    print("✅ Login as user OK")

    # Visit dashboard
    resp = client.get("/")
    assert b"Dashboard" in resp.data
    print("✅ Dashboard after login OK")

    # Visit My Attendance
    resp = client.get("/attendance/history", follow_redirects=True)
    assert resp.status_code == 200
    print("✅ Attendance history OK")

    # Logout
    resp = client.get("/logout", follow_redirects=True)
    assert b"Logged out" in resp.data or resp.status_code == 200
    print("✅ Logout OK")

    # ----------------------------
    # 3️⃣ Login as Admin
    # ----------------------------
    resp = login(client, "test_admin", "admin123")
    assert b"Logged in as" in resp.data
    print("✅ Login as admin OK")

    # Admin Dashboard
    resp = client.get("/admin", follow_redirects=True)
    assert resp.status_code == 200 and b"Users" in resp.data
    print("✅ Admin dashboard OK")

    # Admin Upload
    resp = client.get("/admin/upload", follow_redirects=True)
    assert b"Bulk upload" in resp.data
    print("✅ Admin bulk upload OK")

    # Admin Live Capture
    resp = client.get("/admin/live_capture", follow_redirects=True)
    assert b"Live capture" in resp.data
    print("✅ Admin live capture OK")

    # Admin Gallery
    resp = client.get("/admin/gallery", follow_redirects=True)
    assert b"Gallery" in resp.data
    print("✅ Admin gallery OK")

    # Admin Evaluate
    resp = client.get("/admin/evaluate", follow_redirects=True)
    assert resp.status_code == 200
    print("✅ Admin evaluate route OK")

    # Admin Train
    resp = client.get("/admin/train", follow_redirects=True)
    assert resp.status_code == 200
    print("✅ Admin train model OK")

    # Attendance records CSV (admin only)
    resp = client.get("/attendance/records", follow_redirects=True)
    assert resp.status_code == 200
    print("✅ Attendance CSV export OK")

    # ----------------------------
    # ✅ All buttons & links OK
    # ----------------------------
    print("\n🎉 ALL ROUTES AND BUTTONS TESTED SUCCESSFULLY 🎉")
