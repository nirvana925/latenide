# CLAUDE.md

Guidance for working in this repository.

## What Latenide is

Latenide is a RAW-photo ingestion pipeline with a Streamlit front-end. It watches a folder
(or accepts uploads) for camera RAW files, culls blurry shots, runs optional detectors
(eye-open, exposure, duplicate/burst), generates creative metadata via Google Gemini Vision,
and writes JSON sidecars + a CSV catalog. It also generates Lightroom-importable color grades
from a reference image — both a `.cube` 3D LUT and a `.xmp` Develop Preset.

Project root: `C:\Users\elahi\Documents\Latenide`
(The project was formerly named "Aura" and lived in `Documents\aura` — that folder is
deprecated and empty.)

## Running

```bash
pip install -r requirements.txt

streamlit run app.py        # the UI (Upload & Analyze, Catalog, Style → LUT, Watcher)
python pipeline_watcher.py  # headless daemon: watches Camera_Ingest_Test/ for new RAWs
python debug_vision.py      # isolated sharpness diagnostics (needs sample sharp.ARW/blurry.ARW)
```

Requires a `.env` with `GEMINI_API_KEY`. Without it, the pipeline still culls/routes and
the LUT math still works — only the Gemini steps (metadata, LUT auto-naming) are skipped
with a graceful fallback.

Optional env vars (read by `pipeline_watcher.build_config_from_env`):
`LATENIDE_THRESHOLD` (float), `LATENIDE_REJECT_CLOSED_EYE` (`1`/`0`).

## Architecture

The pipeline logic lives in **one** place so the UI and the watcher behave identically:

- **`pipeline_core.py`** — `process_asset(file_path, base_dir, config, log, wait_stable)` is
  the single entry point for processing one RAW: stability wait → sharpness cull → enabled
  detectors → accept/reject decision → route to queue → (on accept) preview + Gemini metadata
  → sidecar JSON + CSV row. Returns a structured result dict; emits status via the `log`
  callback (never bare `print`). `default_config()` defines the config dict.
- **`pipeline_watcher.py`** — watchdog `Observer` daemon. `RawImageHandler` delegates to
  `pipeline_core.process_asset`. Also runnable standalone via `__main__`.
- **`app.py`** — Streamlit UI. Four tabs; shared config lives in `st.session_state.config`.
  The watcher runs in a background thread, pushing log lines to a `queue.Queue`.
- **`culler.py`** — `evaluate_asset` (Laplacian-variance sharpness score) and `route_asset`
  (moves file to `01_Sharp_Queue` / `02_Rejects_Queue`).
- **`metadata_engine.py`** — `generate_web_preview` (≤1024px JPEG to keep Gemini tokens cheap)
  and `analyze_image_metadata` (Gemini call with a `PhotoMetadata` Pydantic schema, inline
  bytes — not the Files API).
- **`detectors.py`** — `detect_eyes_open` (OpenCV Haar cascades), `check_exposure` (histogram),
  `compute_phash`/`is_duplicate` (dHash + Hamming distance). All operate on the decoded BGR
  thumbnail; no extra heavy deps.
- **`lut_engine.py`** — reference image → `.cube` LUT + `.xmp` Develop Preset. Hybrid: the math is
  local & deterministic (perceptual LAB mean/std color transfer, aka Reinhard), Gemini only
  names/describes the look (`name_lut`, with filename fallback). The measured reference stats
  (`lab_stats`) feed three paths: `reinhard_transfer` does an accurate per-photo in-app preview
  (source stats from the photo); `build_cube_reinhard` bakes a photo-independent 33³ cube against a
  fixed generic-photo prior (`NEUTRAL_LAB_*`); and `build_xmp_preset` builds a Lightroom Develop
  Preset spanning several panels — a luminance tone curve (`_build_tone_curve`, histogram-matched vs
  the prior), per-channel R/G/B curves for the overall cast (`_rgb_channel_curves`), 3-way Color
  Grading (`_zone_color`, mirrored to legacy SplitToning), Basic Texture/Clarity/Saturation
  (`_basic_panel`), and HSL Sat/Lum (`_hsl_panel`). Each panel owns one visual axis so they don't
  compound (WB Temp/Tint, Basic tone sliders, HSL Hue, and Camera Calibration are pinned to identity).
  All three exports are photo-independent approximations of the preview.

### Data flow / on-disk layout (under `Camera_Ingest_Test/`)
- `01_Sharp_Queue/` — accepted RAWs + their `.json` sidecars
- `02_Rejects_Queue/` — blurry / closed-eye rejects
- `03_Cache/` — transient web previews (deleted by the watcher after analysis)
- `photography_catalog.csv` — master catalog (`pipeline_core.CSV_HEADERS`)
- LUTs are written to `Documents\Latenide\LUTs/`

## Conventions & gotchas

- **All processing goes through `pipeline_core.process_asset`.** Don't reimplement culling or
  metadata in the UI or watcher — extend the core function and pass new flags via `config`.
- **Status output is via the `log` callback**, so the UI can capture it. Avoid bare `print`
  in the core/detector modules.
- The **embedded RAW thumbnail** (`rawpy.extract_thumb()` + `cv2.imdecode`) is reused everywhere
  instead of decoding the full RAW — keeps everything fast and Gemini tokens cheap. Follow this
  pattern for any new image analysis.
- OpenCV channel order is **BGR**; convert to RGB before showing in Streamlit (`_bgr_to_rgb`).
- `seen_hashes` in the config persists across a batch/run for in-batch duplicate detection;
  reuse the same `config` object when processing multiple files.
- **Gemini model:** `gemini-2.5-flash` with `response_schema` Pydantic models for structured JSON.
- The Make.com webhook was removed — do not reintroduce outbound posting without asking.
- No test suite yet. Verify changes by `python -m py_compile *.py`, booting
  `streamlit run app.py --server.headless true`, and (when possible) processing a real RAW.

## Dependencies

opencv-python, numpy, rawpy, google-genai, pydantic, python-dotenv, watchdog, streamlit, Pillow.
No MediaPipe (eye detection uses bundled Haar cascades to avoid Windows install friction).
