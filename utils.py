import os
from pathlib import Path

# Directories used by the app
REQUIRED_DIRS = [
    "models",
    "training_images",
    "attendance",
    "debug_logs",
]


def ensure_directories():
    """Create required project directories if they don't exist.

    Returns a list of directories that were created (for logging) and
    silently succeeds if directories already exist.
    """
    created = []
    for d in REQUIRED_DIRS:
        try:
            p = Path(d)
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
                created.append(str(p))
        except Exception:
            # best-effort: ignore failures here so the app can continue
            pass
    return created


def sanitize_filename_component(s: str) -> str:
    """Sanitize a short filename component (no path separators).

    Keeps only ASCII letters, digits and underscore. Collapses whitespace to underscore.
    """
    if not s:
        return ""
    s = str(s).strip().replace(" ", "_")
    # allow only [A-Za-z0-9_]
    return ''.join(ch for ch in s if ch.isalnum() or ch == '_')


def allowed_image_filename(filename: str) -> bool:
    """Basic check for allowed image filename extensions."""
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ('jpg', 'jpeg', 'png')


if __name__ == '__main__':
    print('ensure_directories ->', ensure_directories())
