"""
Latenide optional analysis features.

Each detector operates on the BGR thumbnail array that is already decoded from
the RAW embedded JPEG (same pattern used in culler.py). They are intentionally
lightweight and dependency-free beyond OpenCV/numpy so the pipeline stays fast
and easy to install on Windows.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Eye-open / blink detection
# ---------------------------------------------------------------------------
# We use the Haar cascades that ship inside the opencv-python wheel
# (cv2.data.haarcascades) so there is nothing extra to install. Open eyes are
# detectable; a face with no detected eyes is treated as a likely closed/blink.
#
# NOTE: For higher accuracy you could swap this for MediaPipe Face Mesh and a
# true eye-aspect-ratio (EAR) measurement. It is intentionally omitted here to
# avoid the heavier MediaPipe install on Windows.

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


def detect_eyes_open(img):
    """Detect faces and whether at least one open eye is visible.

    Returns a dict: {"faces": int, "eyes_open": bool, "detail": str}.
    When no face is found, eyes_open defaults to True (we only reject portraits
    where a face is clearly present but the eyes appear closed).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return {"faces": 0, "eyes_open": True, "detail": "No face detected"}

    total_eyes = 0
    for (x, y, w, h) in faces:
        # Look in the upper ~60% of the face box where eyes live.
        roi = gray[y:y + int(h * 0.6), x:x + w]
        eyes = _EYE_CASCADE.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=4)
        total_eyes += len(eyes)

    eyes_open = total_eyes > 0
    detail = (
        f"{len(faces)} face(s), {total_eyes} open eye(s)"
        if eyes_open
        else f"{len(faces)} face(s), eyes appear closed"
    )
    return {"faces": int(len(faces)), "eyes_open": bool(eyes_open), "detail": detail}


# ---------------------------------------------------------------------------
# Exposure / histogram check
# ---------------------------------------------------------------------------

def check_exposure(img, low_clip=2.0, high_clip=2.0):
    """Score exposure from the grayscale brightness histogram.

    Returns {"mean", "pct_clipped_low", "pct_clipped_high", "status"}.
    status is one of "ok", "underexposed", "overexposed".
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total = gray.size

    mean = float(gray.mean())
    pct_low = float((gray <= 10).sum()) / total * 100.0
    pct_high = float((gray >= 245).sum()) / total * 100.0

    if mean < 50 or pct_low > 60:
        status = "underexposed"
    elif mean > 205 or pct_high > 60:
        status = "overexposed"
    else:
        status = "ok"

    return {
        "mean": round(mean, 1),
        "pct_clipped_low": round(pct_low, 1),
        "pct_clipped_high": round(pct_high, 1),
        "status": status,
    }


# ---------------------------------------------------------------------------
# Duplicate / burst grouping (perceptual difference hash)
# ---------------------------------------------------------------------------

def compute_phash(img):
    """Compute a 64-bit difference hash (dHash) for the image.

    Resize to 9x8 grayscale, compare adjacent columns, pack the 64 booleans
    into an int. Visually similar frames produce hashes with a small Hamming
    distance.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()

    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def _hamming(a, b):
    return bin(a ^ b).count("1")


def is_duplicate(phash, seen, threshold=5):
    """Check phash against previously seen hashes.

    `seen` is a list of (phash, label) tuples. Returns (is_dup, label_of_match).
    A match is any prior hash within `threshold` Hamming distance.
    """
    for prev_hash, label in seen:
        if _hamming(phash, prev_hash) <= threshold:
            return True, label
    return False, None
