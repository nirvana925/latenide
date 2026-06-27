"""
Latenide — Streamlit front-end.

Run with:  streamlit run app.py

Tabs:
  • Upload & Analyze — drop one or more RAW files and run the full pipeline.
  • Catalog / Gallery — browse already-processed photos + their AI metadata.
  • Style → LUT — upload a reference image, generate a Lightroom .cube LUT + .xmp preset.
  • Watcher Control — start/stop the folder watcher and view a live log.

All ingestion work is delegated to pipeline_core.process_asset() so the UI and
the headless watcher share identical logic; LUT work lives in lut_engine.
"""

import os
import csv
import glob
import json
import queue

import cv2
import streamlit as st
from dotenv import load_dotenv
from watchdog.observers import Observer

import pipeline_core
import pipeline_watcher
import lut_engine

load_dotenv()

WATCH_DIRECTORY = os.path.expanduser("~/Documents/Latenide/Camera_Ingest_Test")
SHARP_QUEUE = os.path.join(WATCH_DIRECTORY, "01_Sharp_Queue")
REJECTS_QUEUE = os.path.join(WATCH_DIRECTORY, "02_Rejects_Queue")
CACHE_DIR = os.path.join(WATCH_DIRECTORY, "03_Cache")
CSV_PATH = os.path.join(WATCH_DIRECTORY, "photography_catalog.csv")
LUT_DIR = os.path.expanduser("~/Documents/Latenide/LUTs")
RAW_EXTENSIONS = ["arw", "cr3", "nef", "dng"]

os.makedirs(WATCH_DIRECTORY, exist_ok=True)
os.makedirs(LUT_DIR, exist_ok=True)

st.set_page_config(page_title="Latenide", page_icon="📸", layout="wide")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state():
    if "config" not in st.session_state:
        st.session_state.config = pipeline_core.default_config()
    if "observer" not in st.session_state:
        st.session_state.observer = None
    if "log_queue" not in st.session_state:
        st.session_state.log_queue = queue.Queue()
    if "log_lines" not in st.session_state:
        st.session_state.log_lines = []


_init_state()


def _drain_log():
    """Pull any pending log lines from the background thread into the UI list."""
    q = st.session_state.log_queue
    while not q.empty():
        st.session_state.log_lines.append(q.get())
    st.session_state.log_lines = st.session_state.log_lines[-500:]


def _bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Sidebar — global configuration
# ---------------------------------------------------------------------------
cfg = st.session_state.config

with st.sidebar:
    st.title("📸 Latenide")
    st.caption("RAW ingestion · AI metadata · LUTs")

    api_ok = bool(os.environ.get("GEMINI_API_KEY"))
    if api_ok:
        st.success("GEMINI_API_KEY detected")
    else:
        st.error("GEMINI_API_KEY missing — metadata & LUT naming will fall back")

    st.subheader("Culling")
    cfg["threshold"] = st.slider(
        "Sharpness threshold", min_value=1.0, max_value=100.0,
        value=float(cfg["threshold"]), step=1.0,
        help="Laplacian-variance score below this is rejected as blurry.",
    )

    st.subheader("Features")
    cfg["enable_eye"] = st.toggle("Eye-open detection", value=cfg["enable_eye"])
    cfg["reject_closed_eye"] = st.toggle(
        "Reject closed-eye shots", value=cfg["reject_closed_eye"],
        disabled=not cfg["enable_eye"],
    )
    cfg["enable_exposure"] = st.toggle("Exposure check", value=cfg["enable_exposure"])
    cfg["enable_duplicate"] = st.toggle("Duplicate / burst grouping", value=cfg["enable_duplicate"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _render_detector_results(result):
    cols = st.columns(3)
    eye = result.get("eye")
    with cols[0]:
        if eye is None:
            st.metric("Eyes", "—")
        else:
            st.metric("Eyes", "Open" if eye["eyes_open"] else "Closed")
            st.caption(eye["detail"])
    exp = result.get("exposure")
    with cols[1]:
        if exp is None:
            st.metric("Exposure", "—")
        else:
            st.metric("Exposure", exp["status"])
            st.caption(f"mean {exp['mean']} · low {exp['pct_clipped_low']}% · high {exp['pct_clipped_high']}%")
    dup = result.get("duplicate")
    with cols[2]:
        if dup is None:
            st.metric("Duplicate", "—")
        elif dup["is_duplicate"]:
            st.metric("Duplicate", "Yes")
            st.caption(f"matches {dup['match']}")
        else:
            st.metric("Duplicate", "No")


def _render_metadata(meta):
    st.subheader(meta.get("title", "Untitled"))
    st.write(f"**Composition:** {meta.get('composition_style', '')}")
    st.write(f"**Focal criteria:** {meta.get('focal_criteria', '')}")
    st.write(f"**Mood:** {meta.get('mood_profile', '')}")
    palette = meta.get("color_palette", [])
    if palette:
        swatches = "".join(
            f"<span style='display:inline-block;width:28px;height:28px;"
            f"background:{c};border-radius:4px;margin-right:4px;border:1px solid #ccc'></span>"
            for c in palette
        )
        st.markdown(f"**Palette:** {swatches} {', '.join(palette)}", unsafe_allow_html=True)
    tags = meta.get("seo_keywords", [])
    if tags:
        st.write("**Tags:** " + ", ".join(tags))


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_upload, tab_catalog, tab_lut, tab_watch = st.tabs(
    ["⬆️ Upload & Analyze", "🖼️ Catalog", "🎨 Style → LUT", "👁️ Watcher Control"]
)

# --- Upload & Analyze ------------------------------------------------------
with tab_upload:
    st.header("Upload & Analyze")
    st.write("Drop one or more RAW files to run culling, the enabled detectors, and Gemini metadata.")
    uploaded = st.file_uploader(
        "RAW image(s)", type=RAW_EXTENSIONS, accept_multiple_files=True
    )

    if uploaded and st.button("Analyze", type="primary"):
        accepted_count = 0
        with st.spinner(f"Processing {len(uploaded)} file(s)..."):
            for up in uploaded:
                dest = os.path.join(WATCH_DIRECTORY, up.name)
                with open(dest, "wb") as f:
                    f.write(up.getbuffer())

                logs = []
                # Reuse the same cfg so seen_hashes persists across the batch
                # (in-batch duplicate detection).
                result = pipeline_core.process_asset(
                    dest, WATCH_DIRECTORY, config=cfg, log=logs.append, wait_stable=False
                )
                if result["accepted"]:
                    accepted_count += 1

                verdict = "✅ ACCEPTED" if result["accepted"] else f"🚫 REJECTED ({result['reason']})"
                with st.expander(f"{verdict} — {result['filename']} (score {result['score']:.2f})",
                                 expanded=len(uploaded) == 1):
                    if result.get("preview_path") and os.path.exists(result["preview_path"]):
                        st.image(result["preview_path"], caption=result["filename"], use_container_width=True)
                    _render_detector_results(result)
                    if result.get("metadata"):
                        st.divider()
                        _render_metadata(result["metadata"])
                    elif result["accepted"]:
                        st.info("Metadata skipped (no GEMINI_API_KEY or API error).")
                    with st.popover("Processing log"):
                        st.code("\n".join(logs) or "(no output)")

        st.success(f"Batch complete: {accepted_count}/{len(uploaded)} accepted.")

# --- Catalog ---------------------------------------------------------------
with tab_catalog:
    st.header("Catalog")
    col_a, col_b = st.columns([1, 1])
    col_a.metric("Sharp", len(glob.glob(os.path.join(SHARP_QUEUE, "*.*"))))
    col_b.metric("Rejected", len(glob.glob(os.path.join(REJECTS_QUEUE, "*.*"))))

    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        st.caption(f"{len(rows)} catalogued photo(s)")

        for i in range(0, len(rows), 3):
            cols = st.columns(3)
            for col, row in zip(cols, rows[i:i + 3]):
                with col:
                    sidecar = os.path.splitext(row.get("Local Path", ""))[0] + ".json"
                    st.markdown(f"**{row.get('Title', 'Untitled')}**")
                    st.caption(row.get("Filename", ""))
                    badges = []
                    if row.get("Eyes Open"):
                        badges.append(f"👁️ {row['Eyes Open']}")
                    if row.get("Exposure"):
                        badges.append(f"💡 {row['Exposure']}")
                    if row.get("Duplicate Of"):
                        badges.append(f"♻️ dup of {row['Duplicate Of']}")
                    if badges:
                        st.caption(" · ".join(badges))
                    if row.get("SEO Keywords"):
                        st.caption(row["SEO Keywords"])
                    if os.path.exists(sidecar):
                        with st.expander("Details"):
                            with open(sidecar, encoding="utf-8") as sf:
                                st.json(json.load(sf))
    else:
        st.info("No catalog yet. Process a photo to populate photography_catalog.csv.")

# --- Style → LUT -----------------------------------------------------------
with tab_lut:
    st.header("Style → LUT")
    st.warning(
        "🚧 **Work in progress** — accuracy is still being tuned. Both exports are close "
        "approximations of the reference look; use the suggested **manual adjustments** below to "
        "fine-tune the match in your editor."
    )
    st.write(
        "Upload a **reference image** whose color grade you like. Latenide measures its look and "
        "exports two Lightroom-importable files: a `.cube` 3D LUT (a *Profile*) and a `.xmp` "
        "Develop Preset (adjustable sliders). Apply either to your own photos."
    )

    c1, c2 = st.columns(2)
    ref_file = c1.file_uploader(
        "Reference image (the look you want)", type=RAW_EXTENSIONS + ["jpg", "jpeg", "png"],
        key="lut_ref",
    )
    test_file = c2.file_uploader(
        "Photo to apply the LUT to (optional)", type=RAW_EXTENSIONS + ["jpg", "jpeg", "png"],
        key="lut_test",
    )

    intensity = st.slider("Intensity", 0, 100, 100, step=5,
                          help="Blend toward the original. 100% = full effect.") / 100.0

    if ref_file and st.button("Generate LUT", type="primary"):
        try:
            ref_img = lut_engine.load_image_bytes(ref_file.getvalue(), ref_file.name)
        except Exception as e:
            st.error(f"Could not read reference image: {e}")
            ref_img = None

        if ref_img is not None:
            with st.spinner("Measuring look and building LUT..."):
                lut = lut_engine.generate_lut(
                    ref_img, intensity=intensity, fallback_basename=ref_file.name
                )
                stem = lut["name"].replace(" ", "_")
                cube_path = os.path.join(LUT_DIR, f"{stem}.cube")
                lut_engine.write_cube(lut["grid"], cube_path, title=lut["name"])
                xmp_path = os.path.join(LUT_DIR, f"{stem}.xmp")
                lut_engine.write_xmp(
                    ref_img, xmp_path, lut["name"], lut["description"],
                    ref_mean=lut["ref_mean"], ref_std=lut["ref_std"],
                )

            st.subheader(f"🎨 {lut['name']}")
            st.caption(lut["description"])

            # AI-suggested manual edits to match the look beyond the .cube.
            adjustments = lut.get("adjustments") or []
            if adjustments:
                st.markdown("**✨ Manual adjustments to match the look** "
                            "(Lightroom / Camera Raw)")
                st.caption("Apply these on top of — or instead of — the exported files to refine the match.")
                st.table(adjustments)

            # Reference is measure-only: show it untouched as the look being matched.
            st.markdown("**Reference look** (used to build the LUT)")
            st.image(_bgr_to_rgb(ref_img), caption="Reference", width=320)

            # Apply the LUT to the user's own photo, not the reference.
            if test_file:
                try:
                    test_img = lut_engine.load_image_bytes(test_file.getvalue(), test_file.name)
                    st.markdown("**Applied to your photo — before / after**")
                    tcols = st.columns(2)
                    tcols[0].image(_bgr_to_rgb(test_img), caption="Original", use_container_width=True)
                    tcols[1].image(
                        _bgr_to_rgb(lut_engine.reinhard_transfer(
                            test_img, lut["ref_mean"], lut["ref_std"], intensity=intensity)),
                        caption="LUT applied", use_container_width=True,
                    )
                    st.caption("The downloadable `.cube` and `.xmp` are photo-independent "
                               "approximations of this preview.")
                except Exception as e:
                    st.warning(f"Could not preview on your photo: {e}")
            else:
                st.info("Upload a **photo to apply the LUT to** above to preview the look "
                        "on your own image, or just download the files below.")

            dcols = st.columns(2)
            with open(cube_path, "rb") as f:
                dcols[0].download_button(
                    "⬇️ Download .cube LUT", data=f.read(),
                    file_name=os.path.basename(cube_path), mime="text/plain",
                )
            with open(xmp_path, "rb") as f:
                dcols[1].download_button(
                    "⬇️ Download .xmp preset", data=f.read(),
                    file_name=os.path.basename(xmp_path), mime="application/rdf+xml",
                )
            st.info(
                "**Import into Lightroom Classic:**\n\n"
                "- **`.cube` (Profile):** Develop module → Profile Browser → click **+** → "
                "*Import Profiles* → select the `.cube`, then adjust the **Amount** slider. "
                "(Also works in Photoshop, Premiere, DaVinci, Capture One.)\n"
                "- **`.xmp` (Preset):** Develop module → Presets panel → **+** → "
                "*Import Presets…* → select the `.xmp`. It appears under the **Latenide** group."
            )

# --- Watcher Control -------------------------------------------------------
with tab_watch:
    st.header("Watcher Control")
    st.write(f"Monitoring folder: `{WATCH_DIRECTORY}`")

    running = st.session_state.observer is not None
    w1, w2, w3 = st.columns(3)

    if w1.button("▶️ Start", disabled=running):
        handler = pipeline_watcher.RawImageHandler(
            config=cfg, log=st.session_state.log_queue.put
        )
        observer = Observer()
        observer.schedule(handler, path=WATCH_DIRECTORY, recursive=False)
        observer.start()
        st.session_state.observer = observer
        st.session_state.log_queue.put("🚀 Watcher started.")
        st.rerun()

    if w2.button("⏹️ Stop", disabled=not running):
        st.session_state.observer.stop()
        st.session_state.observer.join(timeout=2)
        st.session_state.observer = None
        st.session_state.log_queue.put("🛑 Watcher stopped.")
        st.rerun()

    if w3.button("🔄 Refresh log"):
        st.rerun()

    st.caption("🟢 Running" if st.session_state.observer else "⚪ Stopped")

    _drain_log()
    st.subheader("Live log")
    st.code("\n".join(st.session_state.log_lines[-100:]) or "(no activity yet)")
