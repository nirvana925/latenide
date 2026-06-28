# Latenide

**A RAW-photo ingestion pipeline with a Streamlit front-end.**

Latenide watches a folder (or accepts uploads) for camera RAW files, culls blurry shots,
runs optional detectors (eye-open, exposure, duplicate/burst), generates creative metadata
via Google Gemini Vision, and writes JSON sidecars + a CSV catalog. It also generates
Lightroom-importable color grades from a reference image — a `.cube` 3D LUT and a `.xmp`
Develop Preset.

> ⚠️ **Status: work in progress.** The core pipeline is functional. The **Style → LUT**
> feature in particular is still being tuned for accuracy — see [docs/LUT.md](docs/LUT.md).

---

## Features

- **Cull** — scores sharpness (Laplacian variance) and routes keepers vs. rejects.
- **Detect (optional)** — eye-open check, exposure histogram check, duplicate/burst detection.
- **Describe** — Google Gemini Vision generates a title, mood, colors, and keywords per photo.
- **Catalog** — writes a `.json` sidecar per photo and appends to a master CSV.
- **Style → LUT** — turn a reference image's color grade into a `.cube` 3D LUT (Profile) and a
  `.xmp` Develop Preset for Lightroom, plus AI-suggested manual adjustments to fine-tune the match.

The UI has four tabs: **Upload & Analyze**, **Catalog**, **Style → LUT**, and **Watcher**.

---

## Requirements

- **Python 3.8+** (developed on 3.11)
- The packages in [`requirements.txt`](requirements.txt): opencv-python, numpy, rawpy,
  google-genai, pydantic, python-dotenv, watchdog, streamlit, Pillow.

```bash
pip install -r requirements.txt
```

---

## Set up your API key

Latenide uses Google Gemini for the AI features. 

1. Get a free key from **[Google AI Studio](https://aistudio.google.com/app/apikey)**.
2. Copy the template to a real `.env`:
   ```bash
   cp .env.example .env          # macOS/Linux
   Copy-Item .env.example .env   # Windows PowerShell
   ```
3. Open `.env` and paste your key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

**No key? It still works.** Without a key, Latenide continues to cull, route, and build LUTs —
only the AI metadata, LUT naming, and editing suggestions are skipped with a graceful fallback.

---

## Running

```bash
streamlit run app.py        # the UI (Upload & Analyze, Catalog, Style → LUT, Watcher)
python pipeline_watcher.py  # headless daemon: watches Camera_Ingest_Test/ for new RAWs
python debug_vision.py      # isolated sharpness diagnostics (needs sample sharp.ARW / blurry.ARW)
```

The first command opens the app in your browser. The second runs a background watcher that
processes any RAW dropped into `Camera_Ingest_Test/`.

---

## How it's organized

All processing flows through a single function, `pipeline_core.process_asset(...)`, so the UI and
the headless watcher behave identically. A quick map:

| File | Role |
|------|------|
| `app.py` | Streamlit UI (four tabs) |
| `pipeline_core.py` | The one processing entry point: cull → detect → route → metadata → sidecar/CSV |
| `pipeline_watcher.py` | Watchdog daemon that delegates to `pipeline_core` |
| `culler.py` | Sharpness scoring + file routing |
| `detectors.py` | Eye-open / exposure / duplicate detectors |
| `metadata_engine.py` | Gemini Vision metadata generation |
| `lut_engine.py` | Reference image → `.cube` LUT + `.xmp` Develop Preset + AI editing suggestions |

For a deeper dive, see:

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — module-by-module breakdown and on-disk data flow.


On-disk layout lives under `Camera_Ingest_Test/` (`01_Sharp_Queue`, `02_Rejects_Queue`,
`03_Cache`) with a `photography_catalog.csv` master catalog; generated LUTs go to `LUTs/`. These
folders are kept in the repo as empty skeletons — your actual photos, catalogs, and `.cube`/`.xmp`
files are git-ignored.

---
## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. You are completely free to use, copy, modify, publish, or distribute this software for personal or commercial workflows.

---

*Maintained by the Latenide contributors.*
