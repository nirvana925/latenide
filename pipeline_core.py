"""
Reusable per-asset processing for Latenide.

Both the headless watcher (pipeline_watcher.py) and the Streamlit UI (app.py)
call process_asset() so the ingestion logic lives in exactly one place. Status
messages are emitted through a `log` callback (defaults to print) so the UI can
capture them into a live log, and a structured result dict is returned for
display.
"""

import os
import csv
import json
import time

import cv2
import numpy as np
import rawpy

import culler
import metadata_engine
import detectors

CSV_HEADERS = [
    "Filename", "Local Path", "Title", "Composition Style", "Focal Criteria",
    "Mood Profile", "Color Palette", "SEO Keywords",
    "Eyes Open", "Exposure", "Duplicate Of",
]


def default_config():
    """Return a fresh default configuration dict."""
    return {
        "threshold": 10.0,
        "enable_eye": True,
        "enable_exposure": True,
        "enable_duplicate": True,
        "reject_closed_eye": False,
        "seen_hashes": [],   # list of (phash, filename) tuples, persisted by caller
    }


def _load_thumbnail(file_path):
    """Decode the RAW embedded JPEG thumbnail into a BGR array (or None)."""
    try:
        with rawpy.imread(file_path) as raw:
            thumb = raw.extract_thumb()
            if thumb.format != rawpy.ThumbFormat.JPEG:
                return None
            img_array = np.frombuffer(thumb.data, dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _wait_for_stable(file_path, log):
    """Block until the file size stops changing (handles in-progress copies)."""
    historical_size = -1
    while True:
        if not os.path.exists(file_path):
            log(" ⚠️ Asset disappeared during stability verification. Ingestion canceled.")
            return False
        try:
            current_size = os.path.getsize(file_path)
            if current_size == historical_size:
                return True
            historical_size = current_size
            time.sleep(0.5)
        except OSError:
            time.sleep(0.5)


def process_asset(file_path, base_dir, config=None, log=print, wait_stable=True):
    """Run the full ingestion pipeline on a single RAW asset.

    Returns a result dict describing what happened. `base_dir` is the directory
    that holds the 01_Sharp_Queue / 02_Rejects_Queue / 03_Cache folders.
    """
    if config is None:
        config = default_config()

    filename = os.path.basename(file_path)
    result = {
        "filename": filename,
        "accepted": False,
        "score": 0.0,
        "eye": None,
        "exposure": None,
        "duplicate": None,
        "metadata": None,
        "preview_path": None,
        "reason": "",
    }

    if wait_stable and not _wait_for_stable(file_path, log):
        result["reason"] = "asset_vanished"
        return result

    # --- Sharpness culling --------------------------------------------------
    is_sharp, score = culler.evaluate_asset(file_path, threshold=config["threshold"])
    result["score"] = score

    # --- Optional detectors (run on the embedded thumbnail) -----------------
    thumb = _load_thumbnail(file_path)
    if thumb is not None:
        if config.get("enable_eye"):
            result["eye"] = detectors.detect_eyes_open(thumb)
        if config.get("enable_exposure"):
            result["exposure"] = detectors.check_exposure(thumb)
        if config.get("enable_duplicate"):
            phash = detectors.compute_phash(thumb)
            is_dup, match = detectors.is_duplicate(phash, config["seen_hashes"])
            result["duplicate"] = {"is_duplicate": is_dup, "match": match}
            config["seen_hashes"].append((phash, filename))

    # --- Accept / reject decision ------------------------------------------
    accepted = is_sharp
    reason = "sharp" if is_sharp else "blurry"
    if accepted and config.get("reject_closed_eye") and result["eye"]:
        if result["eye"]["faces"] > 0 and not result["eye"]["eyes_open"]:
            accepted = False
            reason = "closed_eyes"

    result["accepted"] = accepted
    result["reason"] = reason

    # Route the file to the sharp/rejects queue based on the final decision.
    culler.route_asset(file_path, accepted, score)

    if not accepted:
        log(f" 🚫 [REJECTED]: {filename} ({reason}, score {score:.2f})")
        return result

    # --- Accepted: build preview, call Gemini, persist outputs --------------
    moved_path = os.path.join(base_dir, "01_Sharp_Queue", filename)
    cache_dir = os.path.join(base_dir, "03_Cache")
    os.makedirs(cache_dir, exist_ok=True)

    log(" 🧠 Building optimized web-preview asset...")
    preview_file = metadata_engine.generate_web_preview(moved_path, cache_dir)
    result["preview_path"] = preview_file

    log(" 📡 Streaming token payload to Gemini Vision API...")
    json_payload = metadata_engine.analyze_image_metadata(preview_file)

    if not json_payload:
        log(" ⚠️ No metadata returned (check GEMINI_API_KEY). Asset kept in Sharp Queue.")
        return result

    result["metadata"] = json_payload
    log("🎉 [AI METADATA GENERATED SUCCESSFULLY]")

    # Flatten detector results for logging/CSV.
    eyes_open_str = "" if result["eye"] is None else (
        "Yes" if result["eye"]["eyes_open"] else "No"
    )
    exposure_str = "" if result["exposure"] is None else result["exposure"]["status"]
    dup_str = ""
    if result["duplicate"] and result["duplicate"]["is_duplicate"]:
        dup_str = result["duplicate"]["match"] or "unknown"

    # 4a. Sidecar JSON next to the RAW image.
    sidecar_path = os.path.splitext(moved_path)[0] + ".json"
    sidecar_data = dict(json_payload)
    sidecar_data["latenide_analysis"] = {
        "sharpness_score": round(score, 2),
        "eyes_open": eyes_open_str,
        "exposure": exposure_str,
        "duplicate_of": dup_str,
    }
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_data, f, indent=2)
    log(f" 💾 [SIDECAR SAVED]: {os.path.basename(sidecar_path)}")

    # 4b. Append to the master CSV catalog.
    csv_master_path = os.path.join(base_dir, "photography_catalog.csv")
    file_exists = os.path.exists(csv_master_path)
    with open(csv_master_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADERS)
        writer.writerow([
            filename,
            moved_path,
            json_payload.get("title", ""),
            json_payload.get("composition_style", ""),
            json_payload.get("focal_criteria", ""),
            json_payload.get("mood_profile", ""),
            ", ".join(json_payload.get("color_palette", [])),
            ", ".join(json_payload.get("seo_keywords", [])),
            eyes_open_str,
            exposure_str,
            dup_str,
        ])
    log(" 📊 [MASTER SHEET UPDATED]: photography_catalog.csv")

    return result
