import os
import glob
import pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Paths expected by the main app
MODEL_PATH = "models/trained_model.yml"
SKLEARN_MODEL_PATH = "models/sklearn_model.pkl"
TRAINING_FOLDER = "training_images"


def _collect_training_images() -> List[Tuple[int, str]]:
    """Return list of (face_id, filepath) for files under training_images.
    Expected filename format: User.<face_id>....jpg
    """
    files = sorted(glob.glob(os.path.join(TRAINING_FOLDER, "*")))
    out = []
    for p in files:
        name = os.path.basename(p)
        parts = name.split('.')
        if len(parts) < 2:
            continue
        if not parts[1].isdigit():
            continue
        try:
            fid = int(parts[1])
        except Exception:
            continue
        out.append((fid, p))
    return out


def _load_images_for_lbph(pairs: List[Tuple[int, str]]):
    imgs = []
    labels = []
    for fid, p in pairs:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Expect trainer images already 200x200 and equalized; but defensively resize & equalize
        try:
            img = img.astype('uint8')
            img = cv2.equalizeHist(img)
            img = cv2.resize(img, (200, 200))
        except Exception:
            pass
        imgs.append(img)
        labels.append(int(fid))
    return imgs, labels


def _load_images_for_sklearn(pairs: List[Tuple[int, str]]):
    X = []
    y = []
    for fid, p in pairs:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        try:
            img = cv2.resize(img, (200, 200))
            img = cv2.equalizeHist(img.astype('uint8'))
        except Exception:
            pass
        X.append(img.flatten().astype('float32'))
        y.append(int(fid))
    if not X:
        return None, None
    X = np.vstack(X)
    y = np.array(y, dtype=np.int32)
    return X, y


def _ensure_models_dir():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SKLEARN_MODEL_PATH), exist_ok=True)


def train_model() -> Tuple[bool, str]:
    """
    Train LBPH (preferred) and fall back to sklearn KNN.
    Returns (ok: bool, message: str).
    - If opencv-contrib is available and training images exist, trains LBPH and writes MODEL_PATH.
    - Also trains a KNN sklearn model and writes SKLEARN_MODEL_PATH as a fallback.
    """
    _ensure_models_dir()
    pairs = _collect_training_images()
    if not pairs:
        return False, "No training images found in training_images/"

    # Try LBPH first (opencv-contrib)
    lbph_ok = False
    lbph_msg = ""
    try:
        if hasattr(cv2, 'face'):
            # prefer modern API name
            if hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
                recognizer = cv2.face.LBPHFaceRecognizer_create()
            elif hasattr(cv2.face, 'createLBPHFaceRecognizer'):
                recognizer = cv2.face.createLBPHFaceRecognizer()
            else:
                recognizer = None

            if recognizer is not None:
                imgs, labels = _load_images_for_lbph(pairs)
                if len(imgs) >= 2:
                    recognizer.train(imgs, np.array(labels, dtype=np.int32))
                    # write model
                    if hasattr(recognizer, 'write'):
                        recognizer.write(MODEL_PATH)
                    else:
                        recognizer.save(MODEL_PATH)
                    lbph_ok = True
                    lbph_msg = f"LBPH model trained and saved to {MODEL_PATH} (samples={len(labels)})"
                else:
                    lbph_msg = "Not enough images for LBPH (need at least 2)."
        else:
            lbph_msg = "cv2.face (opencv-contrib) not available. Skipping LBPH."
    except Exception as e:
        lbph_ok = False
        lbph_msg = f"LBPH training failed: {e}"

    # Train sklearn KNN fallback (always try)
    sk_ok = False
    sk_msg = ""
    try:
        X, y = _load_images_for_sklearn(pairs)
        if X is not None and len(X) >= 2:
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X, y)
            with open(SKLEARN_MODEL_PATH, 'wb') as f:
                # save as dict for compatibility with older code
                pickle.dump({'model': knn}, f)
            sk_ok = True
            sk_msg = f"Sklearn KNN trained and saved to {SKLEARN_MODEL_PATH} (samples={len(y)})"
        else:
            sk_msg = "Not enough images for sklearn training (need at least 2)."
    except Exception as e:
        sk_ok = False
        sk_msg = f"Sklearn training failed: {e}"

    if lbph_ok or sk_ok:
        msgs = []
        if lbph_msg:
            msgs.append(lbph_msg)
        if sk_msg:
            msgs.append(sk_msg)
        return True, " | ".join(msgs)
    else:
        return False, f"Training failed. LBPH: {lbph_msg}; Sklearn: {sk_msg}"


if __name__ == '__main__':
    ok, msg = train_model()
    print(ok, msg)
