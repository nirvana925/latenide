# Architecture

Latenide is a small, modular RAW-photo pipeline. The guiding principle: **all processing flows
through one function**, `pipeline_core.process_asset(...)`, so the Streamlit UI and the headless
watcher behave identically. Detectors and the LUT engine are independent, side-effect-light modules.

## Module map

| Module | Kind | Responsibility |
|--------|------|----------------|
| `app.py` | entry point | Streamlit UI — four tabs (Upload & Analyze, Catalog, Style → LUT, Watcher). Shared config lives in `st.session_state.config`; the watcher runs in a background thread pushing log lines to a `queue.Queue`. |
| `pipeline_core.py` | library | `process_asset(file_path, base_dir, config, log, wait_stable)` — the single processing entry point. `default_config()` defines the config dict. Emits status via a `log` callback (never bare `print`). |
| `pipeline_watcher.py` | entry point + library | Watchdog `Observer` daemon. `RawImageHandler` delegates each new RAW to `pipeline_core.process_asset`. Also runnable standalone via `__main__`. `build_config_from_env()` reads optional env vars. |
| `culler.py` | library | `evaluate_asset` (Laplacian-variance sharpness score) and `route_asset` (moves a file to `01_Sharp_Queue` / `02_Rejects_Queue`). |
| `detectors.py` | library | `detect_eyes_open` (OpenCV Haar cascades), `check_exposure` (histogram), `compute_phash` / `is_duplicate` (dHash + Hamming distance). All operate on the decoded BGR thumbnail. |
| `metadata_engine.py` | library | `generate_web_preview` (≤1024px JPEG to keep Gemini tokens cheap) and `analyze_image_metadata` (Gemini call with a `PhotoMetadata` Pydantic schema). |
| `lut_engine.py` | library | Reference image → `.cube` LUT + `.xmp` Develop Preset via LAB color transfer, plus AI look-naming and manual editing suggestions. See [LUT.md](LUT.md). |
| `debug_vision.py` | entry point | Standalone sharpness diagnostics against sample `sharp.ARW` / `blurry.ARW`. |

## Processing flow (one RAW)

`process_asset` runs these steps in order:

1. **Stability wait** — optionally wait until the file stops growing (for files still copying in).
2. **Sharpness cull** — `culler.evaluate_asset` scores the embedded thumbnail; below threshold → reject.
3. **Detectors (enabled via config)** — eye-open, exposure, duplicate/burst.
4. **Accept / reject decision** → `culler.route_asset` moves the file to the right queue.
5. **On accept** — build a web preview and call Gemini for metadata.
6. **Persist** — write a `.json` sidecar and append a row to the CSV catalog.

It returns a structured result dict and reports progress through the `log` callback.

## Key conventions

- **One source of truth.** Don't reimplement culling or metadata in the UI or watcher — extend
  `process_asset` and pass new flags via `config`.
- **Reuse the embedded RAW thumbnail.** `rawpy.extract_thumb()` + `cv2.imdecode` is used everywhere
  instead of decoding the full RAW — it keeps everything fast and Gemini tokens cheap. Follow this
  pattern for any new image analysis.
- **BGR vs RGB.** OpenCV is BGR; convert to RGB before displaying in Streamlit.
- **Batch state.** `seen_hashes` in the config persists across a run for in-batch duplicate
  detection — reuse the same `config` object when processing multiple files.
- **Status via callback, not `print`.** Core/detector modules emit through `log` so the UI can capture it.

## On-disk layout

Under `Camera_Ingest_Test/`:

```
01_Sharp_Queue/      accepted RAWs + their .json sidecars
02_Rejects_Queue/    blurry / closed-eye rejects
03_Cache/            transient web previews (deleted after analysis)
photography_catalog.csv   master catalog (pipeline_core.CSV_HEADERS)
```

Generated LUTs are written to `LUTs/`. In the repository these folders are kept as empty skeletons
(via `.gitkeep`); your actual photos, sidecars, catalog, and `.cube`/`.xmp` files are git-ignored so
nothing personal is published.

## Configuration & secrets

- **`GEMINI_API_KEY`** (in `.env`) enables all Gemini features. Without it the pipeline still
  culls/routes and builds LUTs; AI steps fall back gracefully.
- Optional env vars read by `pipeline_watcher.build_config_from_env`: `LATENIDE_THRESHOLD` (float),
  `LATENIDE_REJECT_CLOSED_EYE` (`1`/`0`).
- Secrets are only ever read from the environment (`os.environ` / `python-dotenv`) — never hardcoded.

## Dependencies

opencv-python, numpy, rawpy, google-genai, pydantic, python-dotenv, watchdog, streamlit, Pillow.
Eye detection uses bundled OpenCV Haar cascades (no MediaPipe) to avoid Windows install friction.
