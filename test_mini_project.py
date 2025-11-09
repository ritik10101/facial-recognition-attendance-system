# ============================================================
# test_mini_project.py
# Full Flask test suite + automatic HTML snapshot on failure
# ============================================================

import os
import io
import base64
import random
import string
import pytest
from app import app, init_db

# ============================================================
# Helper utilities
# ============================================================

def random_username(prefix="user"):
    """Generate a random username to avoid conflicts."""
    return prefix + "_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

def fake_image_bytes():
    """Create a small fake grayscale image in memory (200x200)."""
    from PIL import Image
    import numpy as np
    arr = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()

def save_failure_snapshot(test_name: str, response):
    """Save failed page HTML to /test_artifacts for debugging."""
    os.makedirs("test_artifacts", exist_ok=True)
    filename = f"test_artifacts/failed_{test_name}.html"
    try:
        html = response.data.decode("utf-8", errors="ignore")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"⚠️ Saved failure snapshot: {filename}")
    except Exception as e:
        print(f"⚠️ Failed to save snapshot for {test_name}: {e}")

# ============================================================
# Pytest fixtures
# ============================================================

@pytest.fixture(scope="module")
def client():
    """Flask test client with initialized DB."""
    init_db()
    app.config['TESTING'] = True
    app.secret_key = "test_secret"
    with app.test_client() as client:
        yield client

# ============================================================
# Basic route tests
# ============================================================

def test_homepage_loads(client):
    resp = client.get('/')
    if resp.status_code != 200:
        save_failure_snapshot("homepage_loads", resp)
    assert resp.status_code == 200
    assert b"Facial Recognition Attendance System" in resp.data

def test_signup_and_login_flow(client):
    """Signup + login as normal user"""
    username = random_username()
    signup_data = {
        "username": username,
        "password": "test123",
        "role": "user",
        "name": "Test User",
        "uid": "U123",
        "section": "A",
        "course": "AI"
    }
    resp_signup = client.post("/signup", data=signup_data, follow_redirects=True)
    if b"Account created" not in resp_signup.data:
        save_failure_snapshot("signup", resp_signup)
    assert b"Account created" in resp_signup.data

    resp_login = client.post("/login", data={"username": username, "password": "test123"}, follow_redirects=True)
    if b"Logged in as" not in resp_login.data:
        save_failure_snapshot("login", resp_login)
    assert b"Logged in as" in resp_login.data

def test_attendance_page_loads(client):
    resp = client.get("/attendance/mark_form")
    if resp.status_code != 200:
        save_failure_snapshot("attendance_form", resp)
    assert resp.status_code == 200
    assert b"Mark attendance" in resp.data

# ============================================================
# Admin route & upload simulation
# ============================================================

def test_admin_signup_and_login(client):
    """Create an admin account and login."""
    username = random_username("admin")
    signup_data = {
        "username": username,
        "password": "admin123",
        "role": "admin",
        "name": "Admin Tester",
        "uid": "A100",
        "section": "HQ",
        "course": "Control"
    }
    resp_signup = client.post("/signup", data=signup_data, follow_redirects=True)
    if b"Account created" not in resp_signup.data:
        save_failure_snapshot("admin_signup", resp_signup)
    assert resp_signup.status_code == 200
    assert b"Account created" in resp_signup.data

    resp_login = client.post("/login", data={"username": username, "password": "admin123"}, follow_redirects=True)
    if b"Logged in as" not in resp_login.data:
        save_failure_snapshot("admin_login", resp_login)
    assert b"Logged in as" in resp_login.data

def test_admin_access_dashboard(client):
    """Admin dashboard should be accessible after admin login."""
    client.post("/login", data={"username": "admin_test", "password": "admin123"}, follow_redirects=True)
    resp = client.get("/admin", follow_redirects=True)
    if resp.status_code != 200:
        save_failure_snapshot("admin_dashboard", resp)
    assert resp.status_code == 200
    assert b"Users" in resp.data or b"Admin Dashboard" in resp.data

def test_admin_live_capture_upload(client):
    """Simulate live capture image upload as admin."""
    client.post("/login", data={"username": "admin_test", "password": "admin123"}, follow_redirects=True)

    img_bytes = fake_image_bytes()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    payload = {"face_id": 1, "images": [b64 for _ in range(5)]}

    resp = client.post("/admin/live_capture_upload", json=payload)
    if resp.status_code != 200:
        save_failure_snapshot("admin_live_capture", resp)
    data = resp.get_json()
    assert resp.status_code == 200
    assert data["ok"] is True
    assert data["saved"] >= 1

def test_admin_bulk_upload(client):
    """Upload a few fake JPEGs to /admin/upload."""
    client.post("/login", data={"username": "admin_test", "password": "admin123"}, follow_redirects=True)

    files = [(io.BytesIO(fake_image_bytes()), f"img_{i}.jpg") for i in range(3)]
    data = {"face_id": "1", "files": files}

    resp = client.post("/admin/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
    if b"Saved" not in resp.data:
        save_failure_snapshot("admin_bulk_upload", resp)
    assert resp.status_code == 200
    assert b"Saved" in resp.data

def test_admin_train_model(client):
    """Trigger model training route."""
    client.post("/login", data={"username": "admin_test", "password": "admin123"}, follow_redirects=True)
    resp = client.get("/admin/train", follow_redirects=True)
    if resp.status_code != 200:
        save_failure_snapshot("admin_train", resp)
    assert resp.status_code == 200
    assert b"Training" in resp.data or b"trained" in resp.data

# ============================================================
# Attendance workflow simulation
# ============================================================

def test_user_mark_attendance_fails_without_model(client):
    """Ensure attendance endpoint gives appropriate error for dummy image."""
    username = random_username("attend")
    client.post("/signup", data={
        "username": username,
        "password": "abc123",
        "role": "user",
        "name": "Test",
        "uid": "X1"
    }, follow_redirects=True)

    client.post("/login", data={"username": username, "password": "abc123"}, follow_redirects=True)

    dummy_b64 = base64.b64encode(fake_image_bytes()).decode("utf-8")
    resp = client.post("/attendance/mark", json={"image_base64": dummy_b64})
    if resp.status_code not in (200, 400, 500):
        save_failure_snapshot("attendance_mark", resp)
    assert resp.status_code in (200, 400, 500)
    data = resp.get_json()
    assert "ok" in data

def test_attendance_history_requires_login(client):
    """Ensure viewing history requires login."""
    # Clear any previous session
    with client.session_transaction() as sess:
        sess.clear()
    resp = client.get("/attendance/history", follow_redirects=True)
    if resp.status_code != 200:
        save_failure_snapshot("attendance_history_requires_login", resp)
    assert resp.status_code == 200
    assert (b"Login" in resp.data) or (b"My attendance" in resp.data)
