"""
Latenide — reference-image → Lightroom-importable color grade.

Two exportable artifacts are baked from the same measured look:
  • A `.cube` 3D LUT (imports as a Lightroom *Profile* — a per-pixel transform).
  • A `.xmp` Develop Preset (imports into the Presets panel). It carries a luminance
    tone curve + 3-way Color Grading + supporting WB/Vibrance, all measured from the
    reference — a real, adjustable grade, not just sliders. See build_xmp_preset.

Hybrid approach:
  • LUT MATH is measured & deterministic — perceptual color transfer (Reinhard).
    We measure the reference's per-channel mean & standard deviation in CIE-LAB
    (a perceptual space) and re-center / re-scale a target photo's LAB channels to
    match. Mean captures the color cast & exposure of the grade; std captures its
    contrast & saturation. Working in LAB (not RGB) reads as an "editing style"
    rather than a content-driven histogram match.
  • NAMING is the only Gemini touch (name_lut). If GEMINI_API_KEY is missing or
    the call fails we fall back to a name derived from the reference filename, so
    the LUT itself never depends on the API.

Two consumers of the same measured stats:
  • In-app PREVIEW — a true per-photo transfer: we use the target photo's own LAB
    stats as the source, so the result is faithful to that photo.
  • Exported .cube — a 3D LUT must be photo-independent, so we bake the same affine
    LAB transfer against a fixed generic-photo prior (NEUTRAL_LAB_*). This is an
    honest approximation of the per-photo preview, reusable in Lightroom Classic
    (Profile Browser -> Import Profiles).
  • Exported .xmp — a Lightroom Develop Preset that reproduces the grade across several
    panels, each owning one axis: a luminance tone curve (tone shape), per-channel R/G/B
    curves (overall cast), 3-way Color Grading (tonal-zone color), Basic Texture/Clarity/
    Saturation (micro-contrast/chroma), and HSL (per-hue shaping). Fully adjustable.
"""

import math
import os
import uuid
from xml.sax.saxutils import escape

import cv2
import numpy as np
import rawpy

RAW_EXTENSIONS = {".arw", ".cr3", ".nef", ".dng"}


# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------
def load_reference(file_path):
    """Load a reference image as a BGR uint8 array (JPEG/PNG or RAW)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in RAW_EXTENSIONS:
        with rawpy.imread(file_path) as raw:
            thumb = raw.extract_thumb()
            if thumb.format != rawpy.ThumbFormat.JPEG:
                raise ValueError("RAW file has no usable embedded JPEG preview.")
            arr = np.frombuffer(thumb.data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {file_path}")
    return img


def load_image_bytes(data, filename):
    """Load an in-memory uploaded file (bytes) the same way as load_reference."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in RAW_EXTENSIONS:
        # rawpy needs a path-like or file object; wrap the bytes.
        import io
        with rawpy.imread(io.BytesIO(data)) as raw:
            thumb = raw.extract_thumb()
            if thumb.format != rawpy.ThumbFormat.JPEG:
                raise ValueError("RAW file has no usable embedded JPEG preview.")
            arr = np.frombuffer(thumb.data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode uploaded image: {filename}")
    return img


# ---------------------------------------------------------------------------
# Perceptual color transfer (Reinhard, in CIE-LAB)
# ---------------------------------------------------------------------------
# Generic-photo prior used as the *source* stats when baking a photo-independent
# .cube. Units are float-LAB as produced by cv2 on a [0,1] BGR image: L in [0,100],
# a/b roughly in [-128, 127]. These are rough population averages — tune to taste.
NEUTRAL_LAB_MEAN = np.array([55.0, 0.0, 2.0], dtype=np.float64)
NEUTRAL_LAB_STD = np.array([22.0, 8.0, 9.0], dtype=np.float64)

# Clamp the per-channel std scaling so flat / low-variance images don't blow up.
_STD_SCALE_MIN, _STD_SCALE_MAX = 0.5, 2.0
_EPS = 1e-5


def _bgr_to_lab_f(img_bgr):
    """BGR uint8 -> float-LAB (L:0-100, a/b:~-128..127)."""
    f = img_bgr.astype(np.float32) / 255.0
    return cv2.cvtColor(f, cv2.COLOR_BGR2LAB)


def _lab_f_to_bgr(lab):
    """float-LAB -> BGR uint8."""
    bgr = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2BGR)
    return np.clip(bgr * 255.0, 0, 255).astype(np.uint8)


def lab_stats(img_bgr):
    """Return per-channel (mean[3], std[3]) of a BGR image in float-LAB space."""
    lab = _bgr_to_lab_f(img_bgr).reshape(-1, 3).astype(np.float64)
    return lab.mean(axis=0), lab.std(axis=0)


def _scale(ref_std, src_std):
    """Clamped per-channel std ratio ref/src."""
    return np.clip(ref_std / (src_std + _EPS), _STD_SCALE_MIN, _STD_SCALE_MAX)


def reinhard_transfer(target_bgr, ref_mean, ref_std, src_mean=None, src_std=None,
                      intensity=1.0):
    """Re-grade a BGR photo toward the reference's LAB mean/std (Reinhard transfer).

    out_lab[c] = (in_lab[c] - src_mean[c]) * clamp(ref_std/src_std) + ref_mean[c]

    If src_mean/src_std are omitted they are measured from target_bgr itself — a true
    per-photo transfer (the accurate in-app preview path). `intensity` in [0,1] blends
    the result back toward the original (1 = full effect).
    """
    ref_mean = np.asarray(ref_mean, dtype=np.float64)
    ref_std = np.asarray(ref_std, dtype=np.float64)
    lab = _bgr_to_lab_f(target_bgr).astype(np.float64)
    if src_mean is None or src_std is None:
        flat = lab.reshape(-1, 3)
        src_mean, src_std = flat.mean(axis=0), flat.std(axis=0)
    src_mean = np.asarray(src_mean, dtype=np.float64)
    src_std = np.asarray(src_std, dtype=np.float64)

    graded = (lab - src_mean) * _scale(ref_std, src_std) + ref_mean
    out_bgr = _lab_f_to_bgr(graded)

    amount = float(np.clip(intensity, 0.0, 1.0))
    if amount < 1.0:
        out_bgr = np.clip(
            (1.0 - amount) * target_bgr.astype(np.float64) + amount * out_bgr,
            0, 255,
        ).astype(np.uint8)
    return out_bgr


# ---------------------------------------------------------------------------
# 3D .cube export (photo-independent bake of the same LAB transfer)
# ---------------------------------------------------------------------------
def build_cube_reinhard(ref_mean, ref_std, size=33, intensity=1.0,
                        src_mean=NEUTRAL_LAB_MEAN, src_std=NEUTRAL_LAB_STD):
    """Bake the LAB Reinhard transfer into a size^3 RGB grid, values [0,1].

    A .cube must be photo-independent, so the *source* stats default to a fixed
    generic-photo prior (NEUTRAL_LAB_*) rather than any specific photo. Returns a
    float array shaped (size, size, size, 3) in RGB order, R index fastest.
    """
    ref_mean = np.asarray(ref_mean, dtype=np.float64)
    ref_std = np.asarray(ref_std, dtype=np.float64)
    scale = _scale(ref_std, np.asarray(src_std, dtype=np.float64))

    # Build the full RGB grid as one (N,1,3) float32 image and convert in one shot
    # (cv2 wants BGR), instead of size^3 per-pixel cvtColor calls.
    axis = np.linspace(0.0, 1.0, size)
    bb, gg, rr = np.meshgrid(axis, axis, axis, indexing="ij")  # b slowest, r fastest
    rgb = np.stack([rr, gg, bb], axis=-1).reshape(-1, 1, 3).astype(np.float32)
    bgr = rgb[..., ::-1].copy()

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    graded = (lab - np.asarray(src_mean, dtype=np.float64)) * scale + ref_mean

    amount = float(np.clip(intensity, 0.0, 1.0))
    if amount < 1.0:
        graded = (1.0 - amount) * lab + amount * graded

    out_bgr = cv2.cvtColor(graded.astype(np.float32), cv2.COLOR_LAB2BGR)
    out_rgb = np.clip(out_bgr[..., ::-1], 0.0, 1.0)
    return out_rgb.reshape(size, size, size, 3)


def write_cube(grid, path, title="Latenide LUT"):
    """Write an Adobe .cube 3D LUT (R-fastest ordering)."""
    size = grid.shape[0]
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {size}", ""]
    for bi in range(size):
        for gi in range(size):
            for ri in range(size):
                r, g, b = grid[bi, gi, ri]
                lines.append(f"{r:.6f} {g:.6f} {b:.6f}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Lightroom Develop Preset (.xmp) export — a real grade derived from the
# reference. Each visual axis has exactly ONE owner so panels never compound:
#   tone shape -> luminance ToneCurvePV2012   |  overall cast -> R/G/B curves
#   tonal-zone color -> ColorGrade/SplitToning|  chroma -> Vibrance (+Saturation)
#   micro-contrast -> Texture/Clarity         |  per-hue sat/lum -> HSL
# (WB Temp/Tint, Basic tone sliders, HSL Hue, Calibration are pinned to identity.)
# ---------------------------------------------------------------------------
# Fixed input nodes for the point curves (0-255, strictly increasing).
_TONE_ANCHORS = [0, 32, 64, 96, 128, 160, 192, 224, 255]
# crs fields that Lightroom expects as decimals rather than integers.
_FLOAT_CRS_KEYS = {"Exposure2012"}

# Subtle/tasteful tuning: gains map LAB deltas (vs the NEUTRAL_LAB_* prior) onto
# Lightroom's -100..100 sliders; caps keep any single panel from dominating.
_CLARITY_GAIN, _CLARITY_CAP = 1.2, 20
_TEXTURE_GAIN, _TEXTURE_CAP = 0.8, 15
_BASIC_SAT_GAIN, _BASIC_SAT_CAP = 2.0, 25        # half of Vibrance's 4.0
_RGB_CURVE_MAX_SHIFT = 12                          # max |out-in| for cast curves
_HSL_MIN_FRAC = 0.015                              # band must be ≥1.5% of pixels
_HSL_SAT_GAIN, _HSL_SAT_CAP = 1.5, 30
_HSL_LUM_GAIN, _HSL_LUM_CAP = 0.5, 25
# 8 Lightroom HSL bands -> hue-wheel centers in degrees (0-360).
_HSL_BANDS = {
    "Red": 0, "Orange": 30, "Yellow": 60, "Green": 120,
    "Aqua": 180, "Blue": 240, "Purple": 270, "Magenta": 300,
}


def _clamp(v, lo, hi):
    return float(np.clip(v, lo, hi))


def _build_tone_curve(L_flat, strength=0.85):
    """Derive a luminance point curve (list of (in, out) 0-255) from the reference.

    Photo-independent: the *source* tone distribution is the generic-photo prior
    (a Normal(NEUTRAL_LAB_MEAN[L]=55, NEUTRAL_LAB_STD[L]=22) on 0-100), and the
    *target* is the reference's measured L distribution. For each fixed input anchor
    we read the source CDF `q`, then map to the reference's `q`-th L percentile — a
    9-point, heavily smoothed histogram match that captures the look's tone *shape*
    (black lift, contrast, faded/matte) without transferring its content histogram.
    `strength` blends back toward identity (neutral reference -> ~straight line).
    """
    L = np.asarray(L_flat, dtype=np.float64).ravel()
    anchors = np.asarray(_TONE_ANCHORS, dtype=np.float64)
    mu, sd = NEUTRAL_LAB_MEAN[0], NEUTRAL_LAB_STD[0]
    L_in = anchors / 255.0 * 100.0  # input anchors expressed on the 0-100 L scale

    if L.size < 1000 or L.std() < 3.0:
        # Degenerate reference (flat / tiny): no tonal shape to infer, so leave the
        # curve at identity. Any cast is still carried by WB / color grading.
        y_L = L_in.copy()
    else:
        # CDF / percentile match against the neutral prior.
        q = np.array([0.5 * (1.0 + math.erf((x - mu) / (sd * math.sqrt(2.0)))) for x in L_in])
        y_L = np.percentile(L, np.clip(q, 0.0, 1.0) * 100.0)

    ys = y_L / 100.0 * 255.0
    ys = (1.0 - strength) * anchors + strength * ys       # blend toward identity
    ys = np.clip(ys, 0.0, 255.0)
    ys = np.maximum.accumulate(ys)                          # keep Y non-decreasing
    return [(int(x), int(round(y))) for x, y in zip(anchors, ys)]


def _zone_color(lab_flat, global_ab):
    """Measure per-tonal-zone color and map it to 3-way Color Grading crs values.

    Pixels are split by L percentile (shadows ≤33rd, mids, highlights ≥66th). Each
    zone's mean (a, b) has the image-wide cast (`global_ab`, owned by the R/G/B curves)
    subtracted so Color Grading expresses only the *zone-differential* (e.g. teal shadows / warm
    highlights). The residual chroma becomes a hue (0-360, LAB-hue approximation) and a
    capped saturation. Shadow/Highlight are mirrored to legacy SplitToning for
    Lightroom < 10 (which ignores ColorGrade*). Returns a dict of crs fields.
    """
    lab = np.asarray(lab_flat, dtype=np.float64)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    ga, gb = float(global_ab[0]), float(global_ab[1])
    n = L.size
    lo, hi = np.percentile(L, 33), np.percentile(L, 66)
    masks = {"Shadow": L <= lo, "Midtone": (L > lo) & (L < hi), "Highlight": L >= hi}

    out = {}
    for zone, mask in masks.items():
        cnt = int(mask.sum())
        az, bz = (a[mask].mean() - ga, b[mask].mean() - gb) if cnt else (0.0, 0.0)
        chroma = math.hypot(az, bz)
        if cnt < max(1, 0.02 * n) or chroma < 2.0:
            hue, sat = 0, 0
        else:
            hue = int(round((math.degrees(math.atan2(bz, az)) + 360.0) % 360.0))
            sat = int(round(min(chroma * 1.6, 40.0)))
        out[f"ColorGrade{zone}Hue"] = hue
        out[f"ColorGrade{zone}Sat"] = sat
        out[f"ColorGrade{zone}Lum"] = 0

    out["ColorGradeGlobalHue"] = 0
    out["ColorGradeGlobalSat"] = 0
    out["ColorGradeGlobalLum"] = 0
    out["ColorGradeBlending"] = 50   # neutral blend
    out["ColorGradeBalance"] = 0
    # Mirror to legacy SplitToning (no Midtone/Global equivalent).
    out["SplitToningShadowHue"] = out["ColorGradeShadowHue"]
    out["SplitToningShadowSaturation"] = out["ColorGradeShadowSat"]
    out["SplitToningHighlightHue"] = out["ColorGradeHighlightHue"]
    out["SplitToningHighlightSaturation"] = out["ColorGradeHighlightSat"]
    out["SplitToningBalance"] = 0
    return out


def _wb_and_basic(ref_mean, ref_std):
    """Supporting global sliders from the LAB stats. Returns (crs_dict, global_ab).

    Single-owner contract: the tone curve owns brightness/contrast (Exposure2012/
    Contrast2012 pinned to 0) and the per-channel RGB curves own the overall cast, so
    WB Incremental Temperature/Tint are pinned to 0 here (no double cast). Vibrance is
    the one live slider, carrying overall color intensity. `global_ab` (the image's own
    mean a/b) is returned so `_zone_color` can subtract it and isolate the zone color.
    """
    ref_mean = np.asarray(ref_mean, dtype=np.float64)
    ref_std = np.asarray(ref_std, dtype=np.float64)
    _, sa, sb = ref_std - NEUTRAL_LAB_STD

    crs = {
        "IncrementalTemperature": 0,   # cast owned by the R/G/B tone curves
        "IncrementalTint": 0,
        "Vibrance": round(_clamp(((sa + sb) / 2.0) * 4.0, -100, 100)),
        "Exposure2012": 0.0,   # tone curve owns brightness
        "Contrast2012": 0,     # tone curve owns contrast
    }
    return crs, (ref_mean[1], ref_mean[2])


def _fmt_crs(key, value):
    """Format a crs scalar: strings verbatim, exposure as decimal, else integer."""
    if isinstance(value, str):
        return value
    if key in _FLOAT_CRS_KEYS:
        return f"{float(value):.2f}"
    return str(int(round(value)))


def _tone_curve_xml(tag, points):
    """Serialize a point curve as a <crs:TAG><rdf:Seq>…</rdf:Seq></crs:TAG> block."""
    lines = [f"   <crs:{tag}>", "    <rdf:Seq>"]
    lines += [f"     <rdf:li>{int(x)}, {int(y)}</rdf:li>" for x, y in points]
    lines += ["    </rdf:Seq>", f"   </crs:{tag}>"]
    return "\n".join(lines)


def _basic_panel(ref_std):
    """Basic-panel sliders the tone curve can't express. Returns a crs dict.

    The luminance curve owns macro tone, so the tonal sliders (Exposure/Contrast/
    Highlights/Shadows/Whites/Blacks) and Dehaze stay 0. We add only micro-contrast
    (Texture/Clarity, from the reference's luminance spread vs the prior) and a damped
    secondary Saturation (half of Vibrance's weight). Neutral reference -> all 0.
    """
    ref_std = np.asarray(ref_std, dtype=np.float64)
    sL, sa, sb = ref_std - NEUTRAL_LAB_STD
    return {
        # Wider tonal spread than neutral -> punchier; flatter -> gentle softening.
        "Clarity2012": round(_clamp(sL * _CLARITY_GAIN, -_CLARITY_CAP, _CLARITY_CAP)),
        "Texture": round(_clamp(sL * _TEXTURE_GAIN, -_TEXTURE_CAP, _TEXTURE_CAP)),
        "Saturation": round(_clamp(((sa + sb) / 2.0) * _BASIC_SAT_GAIN,
                                   -_BASIC_SAT_CAP, _BASIC_SAT_CAP)),
        "Dehaze": 0,
        "Highlights2012": 0,
        "Shadows2012": 0,
        "Whites2012": 0,
        "Blacks2012": 0,
    }


def _rgb_channel_curves(ref_mean, max_shift=_RGB_CURVE_MAX_SHIFT):
    """Bake the overall color cast into subtle per-channel R/G/B point curves.

    The cast is the reference's mean (a, b) vs the prior. We apply it to a neutral gray
    ramp in LAB and convert back to RGB; the resulting per-channel input->output mapping
    *is* the cast expressed as tone curves (the classic film-curve technique). This is
    the sole owner of the overall cast (WB Incremental is pinned to 0). Shifts are capped
    so the effect stays gentle and the curves are kept monotonic. Zero cast -> identity.
    Returns {"Red"/"Green"/"Blue": [(in, out), …]}.
    """
    ref_mean = np.asarray(ref_mean, dtype=np.float64)
    da, db = ref_mean[1] - NEUTRAL_LAB_MEAN[1], ref_mean[2] - NEUTRAL_LAB_MEAN[2]
    anchors = np.asarray(_TONE_ANCHORS, dtype=np.float64)

    # Take the real gray ramp, measure its (perceptual) LAB, add ONLY the a/b cast, and
    # convert back. Building from actual grays — not a synthesized L — makes zero cast
    # round-trip to exact identity (L is non-linear in the pixel value).
    gray = np.repeat(anchors.reshape(-1, 1, 1), 3, axis=2).astype(np.uint8)  # (N,1,3) BGR
    lab = _bgr_to_lab_f(gray).astype(np.float64)
    lab[..., 1] += da
    lab[..., 2] += db
    bgr = _lab_f_to_bgr(lab).reshape(-1, 3).astype(np.float64)
    rgb = bgr[:, ::-1]  # -> R, G, B (0-255)

    curves = {}
    for i, ch in enumerate(("Red", "Green", "Blue")):
        out = np.clip(rgb[:, i], anchors - max_shift, anchors + max_shift)
        out = np.clip(np.maximum.accumulate(out), 0.0, 255.0)
        curves[ch] = [(int(x), int(round(y))) for x, y in zip(anchors, out)]
    return curves


def _hsl_panel(ref_img, lab_flat):
    """Per-hue Saturation/Luminance shaping (HSL panel). Returns a crs dict.

    Pixels are bucketed into the 8 Lightroom hue bands by HSV hue (near-gray pixels
    ignored). For each band with enough population, SaturationAdjustment is the band's
    LAB chroma vs the image average, and LuminanceAdjustment is the band's mean L vs the
    image average — relative shaping that captures which colors the look pushes or mutes.
    HueAdjustment is left 0 (rotating hues is risky / rarely part of a transferable look).
    Note: Lightroom HSL acts on the *target* photo's colors, so this transfers only
    approximately. Values are smoothed around the band ring. Gray reference -> all 0.
    """
    lab = np.asarray(lab_flat, dtype=np.float64)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    chroma = np.hypot(a, b)
    n = L.size

    hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].reshape(-1).astype(np.float64) * 2.0   # OpenCV H 0-179 -> 0-358
    sat = hsv[:, :, 1].reshape(-1)                            # 0-255
    colored = sat >= 25                                       # ignore near-gray pixels

    L_global = L.mean()
    C_global = chroma[colored].mean() if colored.any() else 0.0
    names = list(_HSL_BANDS)
    centers = np.array([_HSL_BANDS[k] for k in names])

    sat_adj, lum_adj = {}, {}
    for name, center in zip(names, centers):
        diff = np.abs(((hue - center + 180.0) % 360.0) - 180.0)   # circular distance
        mask = colored & (diff <= 22.5)                            # nearest 45° band
        cnt = int(mask.sum())
        if cnt < max(1, _HSL_MIN_FRAC * n):
            sat_adj[name] = lum_adj[name] = 0.0
            continue
        sat_adj[name] = _clamp((chroma[mask].mean() - C_global) * _HSL_SAT_GAIN,
                               -_HSL_SAT_CAP, _HSL_SAT_CAP)
        lum_adj[name] = _clamp((L[mask].mean() - L_global) * _HSL_LUM_GAIN,
                               -_HSL_LUM_CAP, _HSL_LUM_CAP)

    out = {}
    for i, name in enumerate(names):                              # 3-tap ring smoothing
        prev_n, next_n = names[i - 1], names[(i + 1) % len(names)]
        s = (sat_adj[prev_n] + sat_adj[name] + sat_adj[next_n]) / 3.0
        lum = (lum_adj[prev_n] + lum_adj[name] + lum_adj[next_n]) / 3.0
        out[f"SaturationAdjustment{name}"] = round(s)
        out[f"LuminanceAdjustment{name}"] = round(lum)
        out[f"HueAdjustment{name}"] = 0
    return out


def build_xmp_preset(ref_img, name, description="", ref_mean=None, ref_std=None):
    """Build a Lightroom Develop Preset (.xmp) that reproduces the reference grade.

    Populates multiple develop panels, each owning one visual axis so they don't
    compound: a luminance tone curve (`ToneCurvePV2012`) for tone shape; per-channel
    R/G/B tone curves for the overall color cast (WB pinned to 0); 3-way Color Grading
    (+ SplitToning mirror) for tonal-zone color; Basic Texture/Clarity/Saturation for
    micro-contrast and chroma; and HSL Saturation/Luminance for per-hue shaping. All
    measured from `ref_img` (a decoded BGR array); Camera Calibration is left at default.
    Photo-independent and fully adjustable; imports under a "Latenide" group.
    """
    if ref_mean is None or ref_std is None:
        ref_mean, ref_std = lab_stats(ref_img)
    lab_flat = _bgr_to_lab_f(ref_img).reshape(-1, 3).astype(np.float64)

    wb_basic, global_ab = _wb_and_basic(ref_mean, ref_std)
    curve = _build_tone_curve(lab_flat[:, 0])
    zone = _zone_color(lab_flat, global_ab)
    basic = _basic_panel(ref_std)
    hsl = _hsl_panel(ref_img, lab_flat)
    rgb_curves = _rgb_channel_curves(ref_mean)

    attrs = {}
    attrs.update(wb_basic)
    attrs.update(zone)
    attrs.update(basic)
    attrs.update(hsl)
    attrs["ToneCurveName2012"] = "Custom"
    attrs["ToneCurveName"] = "Custom"
    settings = "\n".join(f'    crs:{k}="{_fmt_crs(k, v)}"' for k, v in attrs.items())
    curve_xml = "\n".join([
        _tone_curve_xml("ToneCurvePV2012", curve),
        _tone_curve_xml("ToneCurvePV2012Red", rgb_curves["Red"]),
        _tone_curve_xml("ToneCurvePV2012Green", rgb_curves["Green"]),
        _tone_curve_xml("ToneCurvePV2012Blue", rgb_curves["Blue"]),
    ])

    name_x = escape(name or "Latenide Look")
    desc_x = escape(description or "Color grade derived from the reference image.")
    preset_uuid = uuid.uuid4().hex.upper()

    return f"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    crs:PresetType="Normal"
    crs:Cluster=""
    crs:UUID="{preset_uuid}"
    crs:SupportsAmount="False"
    crs:SupportsColor="True"
    crs:SupportsMonochrome="True"
    crs:SupportsHighDynamicRange="True"
    crs:SupportsNormalDynamicRange="True"
    crs:SupportsSceneReferred="True"
    crs:SupportsOutputReferred="True"
    crs:Version="15.4"
    crs:ProcessVersion="11.0"
    crs:WhiteBalance="As Shot"
{settings}
    crs:HasSettings="True">
{curve_xml}
   <crs:Name>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">{name_x}</rdf:li>
    </rdf:Alt>
   </crs:Name>
   <crs:ShortName>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">{name_x}</rdf:li>
    </rdf:Alt>
   </crs:ShortName>
   <crs:Group>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">Latenide</rdf:li>
    </rdf:Alt>
   </crs:Group>
   <crs:Description>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">{desc_x}</rdf:li>
    </rdf:Alt>
   </crs:Description>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""


def write_xmp(ref_img, path, name, description="", ref_mean=None, ref_std=None):
    """Write a Lightroom Develop Preset (.xmp) to `path`. Returns the path."""
    xmp = build_xmp_preset(ref_img, name, description, ref_mean=ref_mean, ref_std=ref_std)
    with open(path, "w", encoding="utf-8") as f:
        f.write(xmp)
    return path


# ---------------------------------------------------------------------------
# Look naming + manual editing instructions (Gemini, with offline fallback)
# ---------------------------------------------------------------------------
def _stats_adjustments(ref_mean, ref_std):
    """Translate the measured LAB stats into coarse Lightroom-style suggestions.

    Used when Gemini is unavailable. Compares the reference's LAB mean/std against
    the generic-photo prior (NEUTRAL_LAB_*); the sign & size of each delta maps to a
    develop-module move. Returns a list of {setting, suggestion, reason} dicts.
    """
    if ref_mean is None or ref_std is None:
        return []
    ref_mean = np.asarray(ref_mean, dtype=np.float64)
    ref_std = np.asarray(ref_std, dtype=np.float64)
    dL, da, db = ref_mean - NEUTRAL_LAB_MEAN          # L, a(green↔red), b(blue↔yellow)
    sL, sa, sb = ref_std - NEUTRAL_LAB_STD

    def step(delta, lo, hi):
        """Bucket a delta's magnitude into a small/medium/large label."""
        mag = abs(delta)
        return "slightly" if mag < lo else ("strongly" if mag > hi else "moderately")

    out = []
    # b channel ~ warm/cool (yellow vs blue) -> Temperature.
    if abs(db) > 1.5:
        out.append({
            "setting": "Temperature",
            "suggestion": f"{'warmer' if db > 0 else 'cooler'} ({step(db, 4, 10)})",
            "reason": "Reference leans " + ("yellow/warm." if db > 0 else "blue/cool."),
        })
    # a channel ~ magenta vs green -> Tint.
    if abs(da) > 1.5:
        out.append({
            "setting": "Tint",
            "suggestion": f"{'+magenta' if da > 0 else '+green'} ({step(da, 3, 8)})",
            "reason": "Reference has a " + ("magenta/red" if da > 0 else "green") + " cast.",
        })
    # L mean -> Exposure.
    if abs(dL) > 2:
        out.append({
            "setting": "Exposure",
            "suggestion": f"{'raise' if dL > 0 else 'lower'} ({step(dL, 5, 12)})",
            "reason": "Overall " + ("brighter" if dL > 0 else "darker") + " than a neutral frame.",
        })
    # L spread -> Contrast.
    if abs(sL) > 2:
        out.append({
            "setting": "Contrast",
            "suggestion": f"{'increase' if sL > 0 else 'reduce'} ({step(sL, 4, 10)})",
            "reason": ("Wider" if sL > 0 else "flatter") + " tonal range than neutral.",
        })
    # a/b spread -> Saturation / Vibrance.
    sat = (sa + sb) / 2.0
    if abs(sat) > 1.5:
        out.append({
            "setting": "Vibrance / Saturation",
            "suggestion": f"{'increase' if sat > 0 else 'reduce'} ({step(sat, 3, 8)})",
            "reason": ("More" if sat > 0 else "More muted") + " color intensity than neutral.",
        })
    if not out:
        out.append({
            "setting": "—",
            "suggestion": "no strong adjustments",
            "reason": "The reference is already close to a neutral grade.",
        })
    return out


def name_lut(ref_img, fallback_basename="reference", ref_mean=None, ref_std=None):
    """Name/describe the look and suggest manual edits to match it.

    Returns {"name", "description", "adjustments"} where adjustments is a list of
    {setting, suggestion, reason} dicts. Falls back to a filename-derived name and
    stats-derived adjustments if no API key or on any error, so the LUT pipeline
    never hard-depends on the network.
    """
    pretty = os.path.splitext(fallback_basename)[0].replace("_", " ").replace("-", " ").title()
    fallback = {
        "name": f"{pretty} Look",
        "description": "Color grade derived from the reference image.",
        "adjustments": _stats_adjustments(ref_mean, ref_std),
    }
    if not os.environ.get("GEMINI_API_KEY"):
        return fallback

    try:
        from pydantic import BaseModel, Field
        from google import genai
        from google.genai import types

        class Adjustment(BaseModel):
            setting: str = Field(description="Lightroom/Camera Raw control, e.g. Temperature, Contrast, Highlights, Shadows, Vibrance, a specific HSL hue.")
            suggestion: str = Field(description="The move to make, e.g. '+8 warmer', '-20', 'lift to +25'.")
            reason: str = Field(description="Very short why, tied to the look.")

        class LutLook(BaseModel):
            name: str = Field(description="A short, evocative 2-4 word name for this color grade/look.")
            description: str = Field(description="One sentence describing the mood and color treatment.")
            adjustments: list[Adjustment] = Field(
                description="5-8 concrete develop-module adjustments (Temp, Tint, Exposure, "
                "Contrast, Highlights, Shadows, Whites, Blacks, Vibrance, Saturation, key HSL "
                "or tone-curve moves) that recreate this look on a normal photo."
            )

        # Encode the reference as a small JPEG to keep tokens cheap.
        ok, buf = cv2.imencode(".jpg", ref_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return fallback
        image_part = types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")

        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                image_part,
                "Name and describe the color grading / editing style of this photo as a "
                "reusable LUT preset, then list concrete Lightroom/Camera Raw develop "
                "adjustments a user could apply to their own photo to match this look.",
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LutLook,
            ),
        )
        import json
        data = json.loads(response.text)
        adjustments = data.get("adjustments") or fallback["adjustments"]
        return {
            "name": data.get("name") or fallback["name"],
            "description": data.get("description") or fallback["description"],
            "adjustments": adjustments,
        }
    except Exception as e:
        print(f" ⚠️ LUT naming fell back to local ({e}).")
        return fallback


# ---------------------------------------------------------------------------
# One-shot convenience
# ---------------------------------------------------------------------------
def generate_lut(ref_img, intensity=1.0, size=33, gemini_name=True, fallback_basename="reference"):
    """Measure the reference's LAB stats + bake a 3D grid + name the look.

    Returns a dict: {"ref_mean", "ref_std", "grid", "name", "description"}.
    `ref_mean`/`ref_std` drive the accurate per-photo preview (reinhard_transfer);
    `grid` is the photo-independent .cube bake.
    """
    ref_mean, ref_std = lab_stats(ref_img)
    grid = build_cube_reinhard(ref_mean, ref_std, size=size, intensity=intensity)
    if gemini_name:
        look = name_lut(ref_img, fallback_basename, ref_mean=ref_mean, ref_std=ref_std)
    else:
        pretty = os.path.splitext(fallback_basename)[0].replace("_", " ").replace("-", " ").title()
        look = {
            "name": f"{pretty} Look",
            "description": "Color grade derived from the reference image.",
            "adjustments": _stats_adjustments(ref_mean, ref_std),
        }
    return {
        "ref_mean": ref_mean,
        "ref_std": ref_std,
        "grid": grid,
        "name": look["name"],
        "description": look["description"],
        "adjustments": look.get("adjustments", []),
    }
