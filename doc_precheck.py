"""Document-level authenticity pre-check for error5's Type-B evasion (paste -> screenshot
-> PDF), which is undetectable as a paste once cleanly flattened. Instead of asking
"is this signature pasted?" (unanswerable), ask "is this a genuine scanned original?".

DECISION signal (the only one clean on this dataset):
  union_cover : union of ALL image bboxes / page area. genuine scans fill the page
                (accepted min 75%, p10 92%, median 100%); a photo/screenshot placed on an
                A4 page leaves margins (the screenshot fake = 58%). FAIL below 0.70.
INFORMATIONAL only (CONTAMINATED on this dataset — reported, never decisive):
  dpi         : genuine scans range 72-600 here (a 72dpi scan was still accepted), so low
                dpi alone is not fraud.
  producer    : the dataset's own curation used pdftk/itext and many genuine scans have an
                empty Producer, so 'edited' / 'empty' metadata is not by itself fraud.

verdict: PASS (fills the page like a scan) | FAIL (not a full-page scan: photo/screenshot/
size problem). Residual gap: a full-page-coverage screenshot would pass coverage, and dpi
is too noisy here to catch it. Standalone — reads only the PDF, no OCR, no model.
"""
import os, glob, sys
import numpy as np
import fitz

SCAN_HINTS = ["scanner", "scan", "quartz", "distiller", "brother", "canon", "epson",
              "camscanner", "kyocera", "ricoh", "xerox", "fujitsu", "scansnap", "hp "]
EDIT_HINTS = ["pdftk", "itext", "ghostscript", "pillow", "reportlab", "word", "wps",
              "powerpoint", "libreoffice", "skia", "cairo", "chromium", "wkhtml"]

COVER_FAIL = 0.70          # accepted scans never drop below 0.75; the screenshot fake = 0.58


def union_cover(pg):
    R = pg.rect
    H, W = max(1, int(round(R.height))), max(1, int(round(R.width)))
    m = np.zeros((H, W), bool)
    for im in pg.get_image_info(xrefs=True):
        if im["width"] <= 4 or im["height"] <= 4:
            continue
        b = im["bbox"]
        x0, y0 = max(0, int(b[0])), max(0, int(b[1]))
        x1, y1 = min(W, int(b[2])), min(H, int(b[3]))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return float(m.mean())


def main_dpi(pg):
    R = pg.rect
    best, area = 0.0, 0.0
    for im in pg.get_image_info(xrefs=True):
        if im["width"] <= 4:
            continue
        b = im["bbox"]; a = (b[2]-b[0]) * (b[3]-b[1])
        if a > area and (b[2]-b[0]) > 1:
            area = a; best = im["width"] / ((b[2]-b[0]) / 72.0)
    return best


def classify_producer(md):
    s = ((md.get("producer") or "") + " " + (md.get("creator") or "")).lower().strip()
    if not s:
        return "empty", s
    if any(h in s for h in EDIT_HINTS):
        return "edit", s
    if any(h in s for h in SCAN_HINTS):
        return "scan", s
    return "unknown", s


def precheck(pdf_path):
    d = fitz.open(pdf_path); pg = d[0]
    cover = union_cover(pg)
    dpi = main_dpi(pg)
    pkind, praw = classify_producer(d.metadata)
    d.close()

    # decision: coverage only (the one signal that cleanly separates here)
    if cover < COVER_FAIL:
        band = "FAIL"
        reasons = [f"image covers only {cover:.0%} of page (not a full-page scan -> photo/screenshot/size)"]
    else:
        band = "PASS"
        reasons = []
    # informational context for a human reviewer (NOT used to decide)
    info = []
    if dpi < 150:
        info.append(f"low {dpi:.0f}dpi")
    if pkind == "edit":
        info.append(f"PDF-edit producer ({praw[:36]})")
    elif pkind in ("empty", "unknown"):
        info.append(f"{pkind} producer")

    return {"band": band, "cover": round(cover, 2), "dpi": round(dpi),
            "producer": pkind, "reasons": reasons, "info": info}


def group(p):
    if "scanned_sign_pages" not in p:
        return "TEST/FAKE"
    return "accepted" if os.sep + "accepted" + os.sep in p else "need_correction"


def main():
    ROOT = r"C:\Users\25775\Desktop\OCR_research\scanned_sign_pages"
    extra = glob.glob(r"C:\Users\25775\Desktop\OCR_research\*.pdf")  # the user's test files at root
    files = sorted(glob.glob(os.path.join(ROOT, "**", "*.pdf"), recursive=True)) + sorted(extra)
    per = {}
    for f in files:
        r = precheck(f); g = group(f)
        per.setdefault(g, {"PASS": 0, "FAIL": 0, "rows": []})
        per[g][r["band"]] += 1
        if r["band"] == "FAIL":
            per[g]["rows"].append((os.path.basename(f)[:46], r))
    for g in ("accepted", "need_correction", "TEST/FAKE"):
        if g not in per:
            continue
        b = per[g]
        fp = " <-- FALSE POSITIVES!" if g == "accepted" and b["FAIL"] else ""
        print(f"\n[{g}] PASS={b['PASS']} FAIL={b['FAIL']}{fp}")
        for name, r in b["rows"]:
            extra = ("  | info: " + ", ".join(r["info"])) if r["info"] else ""
            print(f"   FAIL cover={r['cover']:.0%} {name}{extra}")


if __name__ == "__main__":
    main()
