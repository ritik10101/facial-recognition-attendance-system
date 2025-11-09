# app.py
# Full upgraded app.py: your original backend logic unchanged,
# with a modern UI injected via an inline BASE template that loads:
#   - static/styles.css
#   - static/app.js
#
# Keep your existing trainer.py, utils.py and db.py files as before.
# After replacing this file, add the two static files:
#   face_attendance_flask/static/styles.css
#   face_attendance_flask/static/app.js
#
# Then run: python app.py

from flask import Flask, render_template_string, request, redirect, url_for, session, send_file, flash, jsonify
import os
import io
import re
import glob
import statistics
import hashlib
import pickle
import base64
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from jinja2 import DictLoader, ChoiceLoader  # in-memory templates

# local imports (assume trainer.py, utils.py and db.py exist)
from trainer import train_model
from utils import ensure_directories
from db import get_conn, test_connection, get_conn as get_mysql_conn

# initialize folders
ensure_directories()

MODEL_PATH = "models/trained_model.yml"
SKLEARN_MODEL_PATH = "models/sklearn_model.pkl"
CONFIDENCE_THRESHOLD = 100
SKLEARN_PROBA_THRESHOLD = 0.80

# ---------------- helper functions ----------------

def _sanitize_for_filename(s: str):
    if not s:
        return ""
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    return s

def pil_to_gray_np(pil_img: Image.Image):
    img = pil_img.convert("L")
    arr = np.array(img)
    return arr

def preprocess_img_np(img_gray, size=(200, 200)):
    if img_gray is None:
        return None
    img = img_gray.astype(np.uint8)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, size)
    return img

def preprocess_face_np(gray_arr, size=(200, 200), pad=0.15):
    if gray_arr is None:
        return None
    h, w = gray_arr.shape[:2]
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_arr, 1.3, 5)
    if len(faces) == 0:
        return None
    faces_sorted = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    x, y, fw, fh = faces_sorted[0]
    pad_w = int(fw * pad)
    pad_h = int(fh * pad)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w, x + fw + pad_w)
    y2 = min(h, y + fh + pad_h)
    face_roi = gray_arr[y1:y2, x1:x2]
    return preprocess_img_np(face_roi, size=size)

def debug_log_prediction(orig_desc, predicted_label, confidence, expected_face_id=None):
    try:
        os.makedirs("debug_logs", exist_ok=True)
        with open(os.path.join("debug_logs", "predictions.log"), "a", encoding="utf-8") as f:
            f.write(
                f"{datetime.now().isoformat()} | {orig_desc} | "
                f"pred={predicted_label} | conf={confidence:.4f} | expected={expected_face_id}\n"
            )
    except Exception:
        pass

# ---------------- DB helpers ----------------

def get_db_name_from_conn(conn):
    try:
        cur = conn.cursor()
        cur.execute("SELECT DATABASE()")
        dbn = cur.fetchone()[0]
        cur.close()
        return dbn
    except Exception:
        return os.getenv("DB_NAME", "face_recognition_db")

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def init_db():
    conn = get_mysql_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            role ENUM('admin','user') NOT NULL DEFAULT 'user',
            face_id INT DEFAULT NULL,
            name VARCHAR(255) DEFAULT NULL,
            uid VARCHAR(100) DEFAULT NULL,
            section VARCHAR(100) DEFAULT NULL,
            course VARCHAR(255) DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS faces (
            face_id INT PRIMARY KEY,
            name VARCHAR(255) NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            face_id INT,
            name VARCHAR(255),
            timestamp DATETIME,
            date DATE,
            UNIQUE KEY uniq_face_date (face_id, date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
    )
    conn.commit()
    cur.close()
    conn.close()

# CRUD wrappers

def create_user(username, password, role, face_id=None, name=None, uid=None, section=None, course=None):
    conn = get_mysql_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, role, face_id, name, uid, section, course) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (username, hash_password(password), role, face_id, name, uid, section, course),
        )
        conn.commit()
        return True, None
    except Exception as e:
        return False, str(e)
    finally:
        cur.close()
        conn.close()

def authenticate(username, password):
    conn = get_mysql_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM users WHERE username = %s", (username,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return False, "User not found"
    if hash_password(password) != row["password_hash"]:
        return False, "Incorrect password"
    return True, row

def get_all_users():
    conn = get_mysql_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, username, role, face_id, name, uid, section, course FROM users ORDER BY id")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def get_user_by_face_id(face_id: int):
    conn = get_mysql_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM users WHERE face_id = %s LIMIT 1", (face_id,))
    r = cur.fetchone()
    cur.close()
    conn.close()
    return r

def update_user_face_id(user_id: int, face_id: int):
    conn = get_mysql_conn()
    cur = conn.cursor()
    cur.execute("UPDATE users SET face_id = %s WHERE id = %s", (face_id, user_id))
    conn.commit()
    cur.close()
    conn.close()

def update_user_details(user_id, name, uid, section, course, role, face_id):
    conn = get_mysql_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE users
        SET name=%s, uid=%s, section=%s, course=%s, role=%s, face_id=%s
        WHERE id=%s
        """,
        (name, uid, section, course, role, face_id, user_id),
    )
    conn.commit()
    cur.close()
    conn.close()

def add_face_mapping(face_id: int, name: str):
    conn = get_mysql_conn()
    cur = conn.cursor()
    cur.execute("REPLACE INTO faces (face_id, name) VALUES (%s, %s)", (face_id, name))
    conn.commit()
    cur.close()
    conn.close()

def get_face_name(face_id: int):
    conn = get_mysql_conn()
    cur = conn.cursor()
    cur.execute("SELECT name FROM faces WHERE face_id = %s", (face_id,))
    r = cur.fetchone()
    cur.close()
    conn.close()
    return r[0] if r else None

def record_attendance(face_id: int, name: str):
    ensure_directories()
    conn = get_mysql_conn()
    cur = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_str = datetime.now().strftime("%Y-%m-%d")
    cur.execute(
        """
        INSERT INTO attendance (face_id, name, timestamp, date)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            name=VALUES(name),
            timestamp=VALUES(timestamp)
        """,
        (face_id, name, ts, date_str)
    )
    conn.commit()
    cur.close()
    conn.close()
    # optional CSV append
    csv_path = os.path.join("attendance", f"{date_str}.csv")
    import csv
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([face_id, name, ts])

def read_attendance(date_str=None):
    conn = get_mysql_conn()
    cur = conn.cursor(dictionary=True)
    if date_str:
        cur.execute("SELECT * FROM attendance WHERE date = %s ORDER BY timestamp DESC", (date_str,))
    else:
        cur.execute("SELECT * FROM attendance ORDER BY timestamp DESC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def clear_attendance(date_str=None):
    conn = get_mysql_conn()
    cur = conn.cursor()
    if date_str:
        cur.execute("DELETE FROM attendance WHERE date = %s", (date_str,))
    else:
        cur.execute("DELETE FROM attendance")
    conn.commit()
    cur.close()
    conn.close()

# ---------------- training image helpers ----------------

def save_training_image_for_face(face_id: int, pil_image: Image.Image):
    arr_gray = pil_image.convert("L")
    arr = np.array(arr_gray)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(arr, 1.3, 5)
    if len(faces) == 0:
        face_img = cv2.resize(arr, (200, 200))
    else:
        faces_sorted = sorted(faces, key=lambda rect: rect[2] * rect[3], reverse=True)
        x, y, w, h = faces_sorted[0]
        face_img = arr[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (200, 200))
    try:
        face_img = face_img.astype(np.uint8)
        face_img = cv2.equalizeHist(face_img)
    except Exception:
        pass
    user = get_user_by_face_id(face_id)
    name_uid = ""
    if user:
        name_part = _sanitize_for_filename(user.get("name") or "")
        uid_part = _sanitize_for_filename(user.get("uid") or "")
        if name_part and uid_part:
            name_uid = f"{name_part}_{uid_part}"
        elif name_part:
            name_uid = name_part
        elif uid_part:
            name_uid = uid_part
    os.makedirs("training_images", exist_ok=True)
    existing = [fn for fn in os.listdir("training_images") if fn.startswith(f"User.{face_id}.")]
    n = len(existing) + 1
    if name_uid:
        fname = os.path.join("training_images", f"User.{face_id}.{name_uid}.{n}.jpg")
    else:
        fname = os.path.join("training_images", f"User.{face_id}.{n}.jpg")
    cv2.imwrite(fname, face_img)
    return fname

def list_training_images(face_id: int):
    folder = "training_images"
    if not os.path.exists(folder):
        return []
    files = sorted([f for f in os.listdir(folder) if f.startswith(f"User.{face_id}.")])
    return [os.path.join(folder, f) for f in files]

def delete_training_image(path: str):
    try:
        os.remove(path)
        return True, None
    except Exception as e:
        return False, str(e)

# ---------------- predictor ----------------

class Predictor:
    def __init__(self, kind, model, proba_threshold=None):
        self.kind = kind
        self.model = model
        self.proba_threshold = proba_threshold

    def predict(self, face_img_200x200_uint8):
        if self.kind == "lbph":
            lbl, conf = self.model.predict(face_img_200x200_uint8)
            return lbl, float(conf), (conf < CONFIDENCE_THRESHOLD)
        else:
            x = face_img_200x200_uint8.reshape(1, -1).astype("float32")
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(x)[0]
                lbl = int(self.model.predict(x)[0])
                max_proba = float(np.max(probs))
                return lbl, max_proba, (max_proba >= (self.proba_threshold or 0.80))
            else:
                lbl = int(self.model.predict(x)[0])
                return lbl, 0.0, False

def ensure_predictor():
    if hasattr(cv2, "face") and os.path.exists(MODEL_PATH):
        try:
            if hasattr(cv2.face, "LBPHFaceRecognizer_create"):
                recognizer = cv2.face.LBPHFaceRecognizer_create()
            elif hasattr(cv2.face, "createLBPHFaceRecognizer"):
                recognizer = cv2.face.createLBPHFaceRecognizer()
            else:
                raise RuntimeError("LBPHFaceRecognizer API not found in cv2.face")
            if hasattr(recognizer, "read"):
                recognizer.read(MODEL_PATH)
            else:
                recognizer.load(MODEL_PATH)
            return Predictor("lbph", recognizer), None
        except Exception as e:
            return None, f"Failed to load LBPH model: {e}"
    if os.path.exists(SKLEARN_MODEL_PATH):
        try:
            with open(SKLEARN_MODEL_PATH, "rb") as f:
                obj = pickle.load(f)
            knn = obj.get("model", obj)
            return Predictor("sklearn", knn, proba_threshold=SKLEARN_PROBA_THRESHOLD), None
        except Exception as e:
            return None, f"Failed to load sklearn model: {e}"
    return None, "No trained model found. Ask admin to train (opencv-contrib preferred)."

# ---------------- Flask app ----------------

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace_this_with_a_real_secret")

# Context processor to provide current year to templates
@app.context_processor
def inject_current_year():
    return {"current_year": datetime.now().year}

# Inline templates (modernized base + small UI JS hooks)
BASE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>{{ title }}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
    <meta name="viewport" content="width=device-width, initial-scale=1">
  </head>
  <body>
    <div class="app-shell">
      <aside class="sidebar">
        <div class="brand">📸 Facial Recognition Attendance System </div>
        <nav>
          <a href="/">Dashboard</a>
          <a href="/attendance/mark_form">Live Attendance</a>
          <a href="/attendance/history">My History</a>
          {% if session.get('user') and session.user.role == 'admin' %}
            <a href="/admin">Admin</a>
          {% endif %}
        </nav>
      </aside>

      <div class="main">
        <header class="topbar">
          <h1>{{ title }}</h1>
          <div class="topbar-right">
            <button class="theme-toggle" title="Toggle theme">🌙</button>

            {% if session.get('user') %}
            <div class="user-menu">
              <div class="user-avatar" id="user-avatar">{{ session.user.username[0]|upper }}</div>
              <div class="dropdown" id="user-dropdown">
                <a href="/profile">👤 Profile</a>
                <a href="/logout">🚪 Logout</a>
              </div>
            </div>
            {% else %}
              <a href="/login" class="btn small">Login</a>
              <a href="/signup" class="btn small">Signup</a>
            {% endif %}
          </div>
        </header>

        <main class="content">
          {% with messages = get_flashed_messages() %}
            {% if messages %}
              <div id="toasts" class="toasts">
                {% for m in messages %}
                  <div class="toast">{{ m }}</div>
                {% endfor %}
              </div>
            {% endif %}
          {% endwith %}
          {% block body %}{% endblock %}
        </main>

        <footer class="app-footer">&copy; {{ current_year }} Face Attendance</footer>
      </div>
    </div>

    <script src="{{ url_for('static', filename='app.js') }}"></script>
  </body>
</html>
"""

# Register in-memory "base" so {% extends "base" %} works
app.jinja_loader = ChoiceLoader([
    DictLoader({"base": BASE}),
    app.jinja_loader,  # keep filesystem loader if you add files later
])

# ensure DB ready
try:
    ok_conn, msg_conn = test_connection()
    if not ok_conn:
        print("DB connection failed:", msg_conn)
except Exception as e:
    print("DB test failed:", e)

init_db()

# --------- routes ---------

# Simple Dashboard (HOME template)
BASE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>{{ title }}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
    <meta name="viewport" content="width=device-width, initial-scale=1">
  </head>
  <body>
    <div class="app-shell">
      <aside class="sidebar">
        <div class="brand">📸 Facial Recognition Attendance System</div>
        <nav>
          <a href="/">🏠 Dashboard</a>
          <a href="/attendance/mark_form">📷 Mark Attendance</a>
          <a href="/attendance/history">🕓 My History</a>
          {% if session.get('user') and session.user.role == 'admin' %}
            <a href="/admin">⚙️ Admin</a>
          {% endif %}
        </nav>
      </aside>

      <div class="main">
        <header class="topbar">
          <h1>{{ title }}</h1>
          <div class="topbar-right">
            <button class="theme-toggle" title="Toggle theme">🌗</button>
            {% if session.get('user') %}
              <div class="user-menu">
                <div class="user-avatar" id="user-avatar">{{ session.user.username[0]|upper }}</div>
                <div class="dropdown" id="user-dropdown">
                  <a href="/profile">👤 Profile</a>
                  <a href="/logout">🚪 Logout</a>
                </div>
              </div>
            {% else %}
              <a href="/login" class="btn small">Login</a>
              <a href="/signup" class="btn small">Signup</a>
            {% endif %}
          </div>
        </header>

        <main class="content">
          {% with messages = get_flashed_messages() %}
            {% if messages %}
              <div id="toasts" class="toasts">
                {% for m in messages %}<div class="toast">{{ m }}</div>{% endfor %}
              </div>
            {% endif %}
          {% endwith %}
          {% block body %}{% endblock %}
        </main>

        <footer class="app-footer">&copy; {{ current_year }} Face Attendance System</footer>
      </div>
    </div>
    <script src="{{ url_for('static', filename='app.js') }}"></script>
  </body>
</html>
"""

app.jinja_loader = ChoiceLoader([
    DictLoader({"base": BASE}),
    app.jinja_loader
])

# ============================================================
# Routes
# ============================================================

HOME = """
{% extends 'base' %}
{% block body %}
<div class="card">
  <h3>Welcome, {{ session.user.name or session.user.username if session.get('user') else 'Guest' }} 👋</h3>
  {% if session.get('user') %}
    <p class="muted">Role: {{ session.user.role }}</p>
    <div class="stack">
      <a class="btn" href="/attendance/mark_form">📸 Mark Attendance</a>
      <a class="btn" href="/attendance/history">🗓 My Attendance</a>
      {% if session.user.role == 'admin' %}
        <a class="btn" href="/admin">⚙️ Admin Dashboard</a>
      {% endif %}
    </div>
  {% else %}
    <p>Please <a href="/login">login</a> or <a href="/signup">signup</a>.</p>
  {% endif %}
</div>
{% endblock %}
"""

@app.route('/')
def index():
    return render_template_string(HOME, title="Dashboard")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    T = """
    {% extends 'base' %}
    {% block body %}
    <div class="card">
      <h3>📝 Create Account</h3>
      <form method="post" class="form-grid">
        <input name="username" placeholder="Username" required>
        <input type="password" name="password" placeholder="Password" required>
        <select name="role"><option>user</option><option>admin</option></select>
        <input name="name" placeholder="Full name">
        <input name="uid" placeholder="UID">
        <input name="section" placeholder="Section">
        <input name="course" placeholder="Course">
        <button class="btn btn-primary" type="submit">Signup</button>
      </form>
    </div>
    {% endblock %}
    """
    if request.method == 'POST':
        ok, err = create_user(
            request.form['username'],
            request.form['password'],
            request.form['role'],
            None,
            request.form.get('name'),
            request.form.get('uid'),
            request.form.get('section'),
            request.form.get('course')
        )
        flash('✅ Account created. Please login.' if ok else f'❌ Signup failed: {err}')
        return redirect(url_for('login' if ok else 'signup'))
    return render_template_string(T, title="Signup")

@app.route('/login', methods=['GET', 'POST'])
def login():
    T = """
    {% extends 'base' %}
    {% block body %}
    <div class="card">
      <h3>🔐 Login</h3>
      <form method="post" class="form-grid">
        <input name="username" placeholder="Username">
        <input type="password" name="password" placeholder="Password">
        <button class="btn btn-primary" type="submit">Login</button>
      </form>
    </div>
    {% endblock %}
    """
    if request.method == 'POST':
        ok, res = authenticate(request.form['username'], request.form['password'])
        if not ok:
            flash(f"❌ {res}")
            return redirect(url_for('login'))
        session['user'] = res
        flash(f"✅ Logged in as {res['username']}")
        return redirect(url_for('index'))
    return render_template_string(T, title="Login")

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("🚪 Logged out successfully.")
    return redirect(url_for('index'))

# ----- admin -----

@app.route('/admin')
def admin_dashboard():
    if not session.get('user') or session['user']['role'] != 'admin':
        flash('Admin only')
        return redirect(url_for('index'))
    users = get_all_users()
    T = """
    {% extends 'base' %}
    {% block body %}
    <div class="card">
      <h3>Users</h3>
      <table class="table">
        <thead><tr><th>id</th><th>username</th><th>role</th><th>face_id</th><th>name</th></tr></thead>
        <tbody>
        {% for u in users %}
          <tr><td>{{u.id}}</td><td>{{u.username}}</td><td>{{u.role}}</td><td>{{u.face_id}}</td><td>{{u.name}}</td></tr>
        {% endfor %}
        </tbody>
      </table>
      <div class="stack" style="margin-top:12px;">
        <a class="btn" href="/admin/train">Train model</a>
        <a class="btn" href="/admin/upload">Bulk upload images</a>
        <a class="btn" href="/admin/live_capture">Live capture</a>
        <a class="btn" href="/admin/gallery">Gallery</a>
        <a class="btn" href="/admin/evaluate">Evaluate</a>
        <a class="btn" href="/attendance/records">Download CSV</a>
      </div>
    </div>
    {% endblock %}
    """
    return render_template_string(T, title="Admin Dashboard", users=users)

@app.route('/admin/clear_attendance', methods=['POST'])
def admin_clear_attendance():
    if not session.get('user') or session['user']['role'] != 'admin':
        flash('Admin only')
        return redirect(url_for('index'))
    d = request.form.get('date')
    try:
        if d:
            _ = datetime.strptime(d, "%Y-%m-%d")
            clear_attendance(date_str=d)
            flash(f'Cleared attendance for {d}')
        else:
            clear_attendance()
            flash('Cleared ALL attendance records')
    except Exception as e:
        flash('Failed to clear: ' + str(e))
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/train')
def admin_train():
    if not session.get('user') or session['user']['role'] != 'admin':
        flash('Admin only')
        return redirect(url_for('index'))
    ok, msg = train_model()
    if ok:
        flash(msg)
    else:
        flash('Training failed: ' + (msg or ''))
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/upload', methods=['GET','POST'])
def admin_upload():
    if not session.get('user') or session['user']['role'] != 'admin':
        flash('Admin only')
        return redirect(url_for('index'))
    T = """
    {% extends 'base' %}
    {% block body %}
    <div class="card">
      <h3>Bulk upload</h3>
      <form method="post" enctype="multipart/form-data">
        <input name="face_id" type="number" placeholder="Face ID"><br>
        <input type="file" name="files" multiple><br>
        <button class="btn" type="submit">Upload</button>
      </form>
    </div>
    {% endblock %}
    """
    if request.method == 'POST':
        face_id = request.form.get('face_id')
        try:
            face_id = int(face_id)
        except Exception:
            flash('Invalid face id')
            return redirect(url_for('admin_upload'))
        files = request.files.getlist('files')
        saved = 0
        errors = []
        for f in files:
            try:
                pil = Image.open(f.stream)
                save_training_image_for_face(face_id, pil)
                saved += 1
            except Exception as e:
                errors.append(str(e))
        flash(f'Saved {saved} files')
        if errors:
            flash('Errors: ' + '; '.join(errors))
        return redirect(url_for('admin_dashboard'))
    return render_template_string(T, title='Bulk upload')

@app.route('/admin/live_capture')
def admin_live_capture():
    if not session.get('user') or session['user']['role'] != 'admin':
        flash('Admin only')
        return redirect(url_for('index'))
    T = """
    {% extends 'base' %}
    {% block body %}
    <div class="card">
      <h3>Live capture & train</h3>
      <label>Face ID: <input id="face_id" type="number" value="1"></label>
      <label>Count: <input id="count" type="number" value="20"></label>
      <div class="camera-wrap" style="margin-top:8px;">
        <video id="video" autoplay muted playsinline></video>
        <canvas id="overlay"></canvas>
      </div>
      <progress id="progress" max="100" value="0" style="width:100%; margin-top:8px;"></progress>
      <div id="preview" class="preview"></div>
      <div style="margin-top:8px;"><button id="startCapture" class="btn btn-primary">Start capture</button></div>
    </div>
    {% endblock %}
    """
    return render_template_string(T, title='Live capture & train')

@app.route('/admin/live_capture_upload', methods=['POST'])
def admin_live_capture_upload():
    if not session.get('user') or session['user']['role'] != 'admin':
        return jsonify({'ok':False,'message':'Admin only'}), 403
    data = request.get_json()
    face_id = data.get('face_id')
    images = data.get('images') or []
    saved = 0
    errors = []
    for idx, b64 in enumerate(images):
        try:
            img_bytes = base64.b64decode(b64)
            pil = Image.open(io.BytesIO(img_bytes)).convert('L')
            save_training_image_for_face(face_id, pil)
            saved += 1
        except Exception as e:
            errors.append(str(e))
    ok, msg = train_model()  # auto-train after capture
    return jsonify({'ok':True,'saved':saved,'errors':errors,'train_ok':ok,'train_msg':msg})

@app.route('/admin/gallery')
def admin_gallery():
    if not session.get('user') or session['user']['role'] != 'admin':
        flash('Admin only')
        return redirect(url_for('index'))
    face_id = request.args.get('face_id', type=int)
    files = list_training_images(face_id) if face_id else []
    T = """
    {% extends 'base' %}
    {% block body %}
    <div class="card">
      <h3>Gallery</h3>
      <form method="get"><input name="face_id" value="{{face_id or ''}}"><button class="btn" type="submit">Show</button></form>
      {% if files %}
        <ul>
        {% for p in files %}
          <li>{{p}} - <a href="/admin/delete_image?path={{p}}">Delete</a></li>
        {% endfor %}
        </ul>
      {% endif %}
    </div>
    {% endblock %}
    """
    return render_template_string(T, title='Gallery', face_id=face_id, files=files)

@app.route('/admin/delete_image')
def admin_delete_image():
    if not session.get('user') or session['user']['role'] != 'admin':
        flash('Admin only')
        return redirect(url_for('index'))
    path = request.args.get('path')
    if not path:
        flash('No path')
        return redirect(url_for('admin_gallery'))
    ok, err = delete_training_image(path)
    if ok:
        flash('Deleted')
    else:
        flash('Delete failed: ' + (err or ''))
    return redirect(url_for('admin_gallery') + f"?face_id={request.args.get('face_id','')}")

@app.route('/admin/evaluate')
def admin_evaluate():
    if not session.get('user') or session['user']['role'] != 'admin':
        flash('Admin only')
        return redirect(url_for('admin_dashboard'))
    rec, rec_err = ensure_predictor()
    if rec is None:
        flash('Model not loaded: ' + (rec_err or ''))
        return redirect(url_for('admin_dashboard'))
    if rec.kind != 'lbph':
        flash('Evaluator reports LBPH confidences only. Use LBPH model to evaluate.')
        return redirect(url_for('admin_dashboard'))
    files = sorted(glob.glob('training_images/*'))
    confs = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        proc = preprocess_img_np(img)
        if proc is None:
            continue
        try:
            _, conf = rec.model.predict(proc)
            confs.append(conf)
        except Exception:
            continue
    if confs:
        flash(f'Training image confidences: count={len(confs)}, min={min(confs):.2f}, median={statistics.median(confs):.2f}, max={max(confs):.2f}')
    else:
        flash('No valid predictions')
    return redirect(url_for('admin_dashboard'))

# ----- attendance (live camera) -----

@app.route('/attendance/mark_form')
def mark_form():
    T = """
    {% extends 'base' %}
    {% block body %}
    <div class="card">
      <h3>Mark attendance — Live camera</h3>
      <div class="camera-wrap">
        <video id="video_mark" autoplay muted playsinline></video>
        <canvas id="overlay_mark"></canvas>
      </div>
      <div class="controls" style="margin-top:8px;">
        <button id="snap" class="btn btn-primary">Capture & Mark Attendance</button>
        <span id="status" class="muted" style="margin-left:12px;"></span>
      </div>
      <hr>
      <h4>Or upload a file</h4>
      <form method="post" action="/attendance/mark" enctype="multipart/form-data">
        Upload photo: <input type="file" name="photo"><br>
        <button class="btn" type="submit">Mark attendance</button>
      </form>
    </div>
    {% endblock %}
    """
    return render_template_string(T, title='Mark attendance')

@app.route('/attendance/mark', methods=['POST'])
def mark_attendance():
    if not session.get('user'):
        return jsonify({'ok':False,'message':'Login required'}), 403
    pil = None
    if request.is_json:
        data = request.get_json()
        b64 = data.get('image_base64')
        if not b64:
            return jsonify({'ok':False,'message':'No image provided'}), 400
        try:
            img_bytes = base64.b64decode(b64)
            pil = Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            return jsonify({'ok':False,'message':'Invalid image: '+str(e)}), 400
    else:
        f = request.files.get('photo')
        if not f:
            return redirect(url_for('mark_form'))
        try:
            pil = Image.open(f.stream)
        except Exception:
            flash('Invalid image')
            return redirect(url_for('mark_form'))

    try:
        gray_np = pil_to_gray_np(pil)
        face_resized = preprocess_face_np(gray_np, size=(200,200), pad=0.15)
        if face_resized is None:
            return jsonify({'ok':False,'message':'No face detected'}) if request.is_json \
                else (flash('No face detected') or redirect(url_for('mark_form')))
        predictor, err = ensure_predictor()
        if predictor is None:
            return jsonify({'ok':False,'message':'Model not loaded: '+(err or '')}) if request.is_json \
                else (flash('Model not loaded') or redirect(url_for('mark_form')))
        label, score, is_confident = predictor.predict(face_resized)
        debug_log_prediction(f"upload:{session['user']['username']}", label, score, session['user'].get('face_id'))
        if predictor.kind == 'lbph':
            info = f"Recognition (LBPH): id={label}, confidence={score:.2f} (require < {CONFIDENCE_THRESHOLD})"
        else:
            info = f"Recognition (KNN): id={label}, probability={score:.2%} (require ≥ {SKLEARN_PROBA_THRESHOLD:.0%})"
        user_face_id = session['user'].get('face_id')
        if not is_confident:
            return jsonify({'ok':False,'message':'Not confident enough to mark attendance','info':info}) if request.is_json \
                else (flash('Not confident enough to mark attendance') or redirect(url_for('mark_form')))
        if user_face_id is None:
            return jsonify({'ok':False,'message':'No face id assigned to your account'}) if request.is_json \
                else (flash('No face id assigned to your account') or redirect(url_for('mark_form')))
        if label != user_face_id:
            return jsonify({'ok':False,'message':f'Recognized as id {label}, which does not match your assigned id {user_face_id}','info':info}) if request.is_json \
                else (flash(f'Recognized as id {label}, which does not match your assigned id {user_face_id}') or redirect(url_for('mark_form')))
        name = get_face_name(label) or f'User{label}'
        record_attendance(label, name)
        msg = f'Recognized as {name} — attendance recorded'
        return jsonify({'ok':True,'message':msg,'info':info}) if request.is_json \
            else (flash(msg) or redirect(url_for('index')))
    except Exception as e:
        return jsonify({'ok':False,'message':'Recognition error: '+str(e)}) if request.is_json \
            else (flash('Recognition error: '+str(e)) or redirect(url_for('mark_form')))

@app.route('/attendance/history')
def attendance_history():
    if not session.get('user'):
        flash('Login required')
        return redirect(url_for('login'))
    rows = read_attendance()
    user_face = session['user'].get('face_id')
    if user_face:
        rows = [r for r in rows if r['face_id'] == user_face]
    T = """
    {% extends 'base' %}
    {% block body %}
    <div class="card">
      <h3>My attendance</h3>
      {% if rows %}
        <table class="table">
          <thead><tr><th>face_id</th><th>name</th><th>timestamp</th><th>date</th></tr></thead>
          <tbody>
            {% for r in rows %}
              <tr><td>{{r.face_id}}</td><td>{{r.name}}</td><td>{{r.timestamp}}</td><td>{{r.date}}</td></tr>
            {% endfor %}
          </tbody>
        </table>
      {% else %}
        <p>No records</p>
      {% endif %}
    </div>
    {% endblock %}
    """
    return render_template_string(T, title='My attendance', rows=rows)

@app.route('/attendance/records')
def attendance_records():
    if not session.get('user') or session['user']['role'] != 'admin':
        flash('Admin only')
        return redirect(url_for('index'))
    d = request.args.get('date')
    rows = read_attendance(date_str=d)
    df = pd.DataFrame(rows)
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    return send_file(io.BytesIO(csv_bytes), mimetype='text/csv', as_attachment=True, download_name=f"attendance_{d or 'all'}.csv")

# -------------- run --------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # default to 0.0.0.0 as requested; override with env or args if needed
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5000")))
    args = parser.parse_args()
    print(f"Starting on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)
