# eval_on_train.py
import os
import glob
import cv2
import numpy as np
from collections import defaultdict
import statistics
from trainer import preprocess_img, IMAGE_SIZE

MODEL_PATH = "models/trained_model.yml"

def load_recognizer(model_path):
    if not hasattr(cv2, "face"):
        raise RuntimeError("cv2.face not available. Install opencv-contrib-python")
    if hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        rec = cv2.face.LBPHFaceRecognizer_create()
    elif hasattr(cv2.face, "createLBPHFaceRecognizer"):
        rec = cv2.face.createLBPHFaceRecognizer()
    else:
        raise RuntimeError("LBPHFaceRecognizer API missing from cv2.face")

    # load model
    if os.path.exists(model_path):
        try:
            if hasattr(rec, "read"):
                rec.read(model_path)
            else:
                rec.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return rec

def parse_expected_label(fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    parts = stem.split('.')
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1])
    return None

def main():
    files = sorted(glob.glob(os.path.join('training_images', '*')))
    if not files:
        print('No training_images found.')
        return

    try:
        rec = load_recognizer(MODEL_PATH)
    except Exception as e:
        print('Failed to load recognizer:', e)
        return

    per_id = defaultdict(list)
    all_confs = []
    mismatches = []

    for f in files:
        expected = parse_expected_label(f)
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read {f}")
            continue
        proc = preprocess_img(img, size=IMAGE_SIZE)
        if proc is None:
            print(f"Preprocessing returned None for {f}")
            continue
        try:
            lbl, conf = rec.predict(proc)
        except Exception as e:
            print(f"Predict failed for {f}: {e}")
            continue
        all_confs.append(conf)
        if expected is not None:
            per_id[expected].append(conf)
        ok = (expected == lbl)
        print(f"{os.path.basename(f)} -> predicted={lbl}, conf={conf:.2f}, expected={expected}, ok={ok}")
        if not ok:
            mismatches.append((f, expected, lbl, conf))

    if not all_confs:
        print('No predictions made.')
        return

    print('\nSummary statistics:')
    print(f"Total training images evaluated: {len(all_confs)}")
    print(f"Global confs — min:{min(all_confs):.2f}, median:{statistics.median(all_confs):.2f}, mean:{statistics.mean(all_confs):.2f}, max:{max(all_confs):.2f}")

    print('\nPer-face-id medians:')
    medians = []
    for fid in sorted(per_id.keys()):
        vals = per_id[fid]
        print(f"face_id={fid}: count={len(vals)}, min={min(vals):.2f}, median={statistics.median(vals):.2f}, mean={statistics.mean(vals):.2f}, max={max(vals):.2f}")
        medians.append(statistics.median(vals))

    if medians:
        global_median = statistics.median(medians)
        rec_median = statistics.median(all_confs)
        # Heuristic recommendation: choose threshold somewhat above median of training confs
        recommended = max(rec_median * 1.5, rec_median + 20)
        print('\nRecommended starting CONFIDENCE_THRESHOLD (LBPH, lower is better):')
        print(f"- median of medians: {global_median:.2f}")
        print(f"- median of all training confs: {rec_median:.2f}")
        print(f"- recommended threshold (heuristic): {recommended:.2f}")

    if mismatches:
        print(f"\nNumber of mismatches on training set: {len(mismatches)}")
        for m in mismatches[:50]:
            f, expected, lbl, conf = m
            print(f"MISMATCH: {os.path.basename(f)} expected={expected} predicted={lbl} conf={conf:.2f}")

if __name__ == '__main__':
    main()
