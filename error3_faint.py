"""error3 SIGNATURE_TOO_FAINT research.

A signature is "too faint" when its strokes are too light relative to the paper —
a low ink-vs-background CONTRAST. Pipeline: locate cell -> remove form lines ->
isolate ink pixels -> measure how dark the stroke pixels are vs the paper.

Metrics per signed cell:
  bg        : paper level = 90th pct of cell grayscale (bright)
  ink_med   : median grayscale of ink-mask pixels (the stroke darkness; high=faint)
  contrast  : bg - ink_med            (absolute; low = faint)
  weber     : contrast / bg           (normalized 0..1; low = faint)
  dark_frac : fraction of ink pixels darker than 0.55*bg (strong ink present?)

Establishes the acceptable range over all signed cells, locates the 'faded'-labeled
samples in that range, and validates with synthetically dimmed signatures.
"""
import os, glob, json
import numpy as np
import cv2
import common as C
from fields_config import FIELDS, DEFAULT_OFFSETS, locate_field, get_fields

TPL = r"C:\Users\25775\Desktop\OCR_research\templates.json"
OUT = r"C:\Users\25775\Desktop\OCR_research\faint_out"
os.makedirs(OUT, exist_ok=True)


def offsets(tpl, form, field, kind):
    return tpl.get(form, {}).get(field, {}).get(kind) or DEFAULT_OFFSETS[form][kind]


def ink_mask(gray):
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 15)
    Hc, Wc = bw.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, Wc // 3), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, Hc // 3)))
    lines = cv2.dilate(cv2.erode(bw, hk), hk) | cv2.dilate(cv2.erode(bw, vk), vk)
    ink = cv2.subtract(bw, lines)
    return cv2.medianBlur(ink, 3)


def faint_metrics(crop):
    if crop.size == 0:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    m = ink_mask(g)
    ink_px = g[m > 0]
    if ink_px.size < 30:
        return None
    bg = float(np.percentile(g, 90))
    ink_med = float(np.median(ink_px))
    contrast = bg - ink_med
    weber = contrast / max(bg, 1.0)
    dark_frac = float((ink_px < 0.55 * bg).mean())
    return {"bg": round(bg, 1), "ink_med": round(ink_med, 1),
            "contrast": round(contrast, 1), "weber": round(weber, 3),
            "dark_frac": round(dark_frac, 3), "n_ink": int(ink_px.size)}


def dim_signature(crop, alpha):
    """Synthesize a fainter signature: blend strokes toward the paper background.
    alpha in (0,1]; smaller = fainter (alpha=1 is original)."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bg = np.percentile(g, 90)
    out = bg - (bg - g.astype(float)) * alpha     # shrink stroke depth toward bg
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def collect(tpl):
    rows = []
    for f in sorted(glob.glob(os.path.join(C.ROOT, "**", "*.pdf"), recursive=True)):
        rel = os.path.relpath(f, C.ROOT)
        cache = C.load_cache(rel)
        if cache is None:
            continue
        items = cache["items"]; form = C.detect_form(items)
        if form not in FIELDS:
            continue
        img = C.render(os.path.join(C.ROOT, rel))
        for fld in get_fields(form, required_only=True):
            sig, dat = locate_field(items, fld)
            if sig is None:
                continue
            box = C.sig_box_clip_date(sig["box"], dat["box"] if dat else None,
                                      offsets(tpl, form, fld["name"], "sig"))
            crop = img[box[1]:box[3], box[0]:box[2]]
            if C.ink_stats(crop)["ink_ratio"] < 0.012:
                continue
            fe = faint_metrics(crop)
            if fe:
                fe["who"] = f"{fld['name']}/{os.path.basename(rel)}"
                fe["crop"] = crop
                rows.append(fe)
    return rows


def dist(rows, k):
    v = np.array([r[k] for r in rows])
    return f"{k:9} min={v.min():.2f} p10={np.percentile(v,10):.2f} med={np.median(v):.2f} p90={np.percentile(v,90):.2f} max={v.max():.2f}"


def main():
    tpl = json.load(open(TPL, encoding="utf-8"))
    rows = collect(tpl)
    print(f"=== {len(rows)} signed signature cells ===")
    for k in ("bg", "ink_med", "contrast", "weber", "dark_frac"):
        print("  " + dist(rows, k))

    print("\n'faded'-labeled samples (where do their signatures fall?):")
    for r in rows:
        if "fade" in r["who"].lower():
            print(f"   {r['who'][:50]:50} contrast={r['contrast']} weber={r['weber']} dark_frac={r['dark_frac']}")

    print("\nLOWEST-contrast signatures in the dataset (faintness ranking):")
    for r in sorted(rows, key=lambda x: x["contrast"])[:8]:
        print(f"   contrast={r['contrast']:5} weber={r['weber']:.3f} dark_frac={r['dark_frac']:.2f}  {r['who'][:48]}")

    # synthetic dimming on a few normal (high-contrast) signatures
    print("\nSYNTHETIC dimming (alpha=1.0 orig -> fainter):")
    strong = sorted(rows, key=lambda x: -x["contrast"])[:3]
    for r in strong:
        line = f"   {r['who'][:30]:30}"
        for a in (1.0, 0.5, 0.3, 0.2, 0.12):
            fe = faint_metrics(dim_signature(r["crop"], a))
            line += f"  a={a}:c{fe['contrast']:.0f}/w{fe['weber']:.2f}"
        print(line)


if __name__ == "__main__":
    main()
