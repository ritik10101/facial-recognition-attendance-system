#!/usr/bin/env python3
"""
verify_env.py — sanity checks for the Face Attendance (Flask) project.

Usage:
  python verify_env.py
  python verify_env.py --image path/to/sample.jpg
"""

import os
import sys
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# ---- optional imports guarded for clearer error messages
def safe_import(name):
    try:
        return __import__(name)
    except Exception as e:
        print(f"[FAIL] cannot import {name}: {e}")
        return None

def status(ok, msg):
    print(("[ OK ] " if ok else "[FAIL] ") + msg)
    return ok

def warn(msg):
    print("[WARN] " + msg)

def die(msg, code=1):
    print("[FATAL] " + msg)
    sys.exit(code)

def check_python():
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 8)
    return status(ok, f"Python {major}.{minor} (>= 3.8 required)")

def check_packages():
    ok_all = True
    needed = [
        "flask",
        "mysql.connector",  # from mysql-connector-python
        "cv2",
        "numpy",
        "pandas",
        "PIL",              # Pillow
        "sklearn"
    ]
    for mod in needed:
        m = safe_import(mod.split('.')[0])
        ok_all &= m is not None
    return ok_all

def show_versions():
    import platform
    print("---- Versions ----")
    print("Platform:", platform.platform())
    print("Python  :", sys.version.replace("\n"," "))
    for name in ["flask", "mysql", "cv2", "numpy", "pandas", "PIL", "sklearn"]:
        try:
            if name == "mysql":
                import mysql.connector as _m
                ver = getattr(_m, "__version__", "unknown")
            elif name == "PIL":
                import PIL as _m
                ver = getattr(_m, "__version__", "unknown")
            else:
                _m = __import__(name)
                ver = getattr(_m, "__version__", "unknown")
            print(f"{name:<7}: {ver}")
        except Exception as e:
            print(f"{name:<7}: (not found) {e}")
    print("------------------")

def check_cv2_face():
    cv2 = safe_import("cv2")
    if cv2 is None:
        return False
    ok = hasattr(cv2, "face")
    msg = "OpenCV contrib present (cv2.face found)" if ok else \
          "OpenCV contrib NOT present — install opencv-contrib-python"
    return status(ok, msg)

def check_db_connection():
    try:
        from db import test_connection
    except Exception as e:
        return status(False, f"db.py not importable: {e}")

    ok, msg = test_connection()
    return status(ok, f"MySQL: {msg}")

def check_directories():
    required = ["models", "training_images", "attendance", "debug_logs"]
    ok_all = True
    for d in required:
        p = Path(d)
        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=True)
                status(True, f"Created missing directory: {d}")
            except Exception as e:
                ok_all &= status(False, f"Cannot create directory {d}: {e}")
                continue
        # write test
        try:
            tmp = p / ".write_test"
            tmp.write_text("ok", encoding="utf-8")
            tmp.unlink(missing_ok=True)
            status(True, f"{d}: writeable")
        except Exception as e:
            ok_all &= status(False, f"{d}: NOT writeable ({e})")
    return ok_all

def check_models():
    ok_all = True
    yml = Path("models/trained_model.yml")
    pkl = Path("models/sklearn_model.pkl")
    if yml.exists():
        try:
            ysz = yml.stat().st_size
            status(True, f"{yml} exists ({ysz} bytes)")
        except Exception as e:
            ok_all &= status(False, f"{yml} unreadable: {e}")
    else:
        warn(f"{yml} not found (LBPH will be unavailable until you train)")

    if pkl.exists():
        try:
            psz = pkl.stat().st_size
            status(True, f"{pkl} exists ({psz} bytes)")
        except Exception as e:
            ok_all &= status(False, f"{pkl} unreadable: {e}")
    else:
        warn(f"{pkl} not found (sklearn fallback will be unavailable until you train)")
    return ok_all

def check_env_vars():
    keys = ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME", "FLASK_SECRET_KEY"]
    for k in keys:
        v = os.getenv(k)
        if v is None:
            warn(f"{k} not set (using defaults). Consider configuring via .env")
        else:
            shown = v if k != "DB_PASSWORD" else "***"
            print(f"[ INFO ] {k} = {shown}")
    return True

def quick_face_detect(sample_path: Path):
    cv2 = safe_import("cv2")
    if cv2 is None:
        return False
    if not sample_path.exists():
        return status(False, f"Sample image not found: {sample_path}")
    try:
        img = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return status(False, "Failed to read sample image (None)")
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(img, 1.3, 5)
        return status(len(faces) > 0, f"Face detection on sample: {len(faces)} face(s) found")
    except Exception as e:
        return status(False, f"Face detection failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Optional sample image path to test face detection")
    args = parser.parse_args()

    print("=== Face Attendance Environment Verification ===")
    print("Started:", datetime.now().isoformat())
    overall_ok = True

    overall_ok &= check_python()
    if not check_packages():
        overall_ok = False

    show_versions()
    overall_ok &= check_cv2_face()
    overall_ok &= check_env_vars()
    overall_ok &= check_directories()
    overall_ok &= check_models()
    overall_ok &= check_db_connection()

    if args.image:
        overall_ok &= quick_face_detect(Path(args.image))

    print("-----------------------------------------------")
    if overall_ok:
        print("All checks passed ✅")
        sys.exit(0)
    else:
        print("One or more checks failed ❌")
        sys.exit(2)

if __name__ == "__main__":
    try:
        main()
    except SystemExit as se:
        raise
    except Exception:
        traceback.print_exc()
        die("Unhandled exception during verification.")
