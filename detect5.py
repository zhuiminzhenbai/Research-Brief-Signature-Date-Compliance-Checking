"""error5 PASTED_IMAGE_SIGNATURE detector.

Two complementary layers (pixel forensics was tested and FAILS on clean scans, so it is
NOT used):

  Layer-1 (structural)  : read the PDF's image objects (no render); a signature pasted on
                          a phone as a separate image object sits in the signature field as
                          a non-page-covering, non-tiling image -> PASTED. Catches the
                          common phone-paste cheat deterministically.
  doc-precheck          : a paste that was screenshotted/flattened leaves no separate object
                          (Layer-1 blind), but the document then fails to look like a real
                          full-page scan -> doc_precheck FAIL. (see doc_precheck.py)

verdict per file:
  PASTED_IMAGE_SIGNATURE   Layer-1 found a pasted image object in a signature field
  DOC_NOT_SCAN             not a genuine full-page scan (screenshot/photo) -> manual review
  OK                       neither fired
"""
import os, glob
from collections import Counter
import fitz
import common as C
from fields_config import locate_field, get_fields, FIELDS
from detect4 import offsets
from doc_precheck import precheck
import json

TPL = json.load(open(r"C:\Users\25775\Desktop\OCR_research\templates.json", encoding="utf-8"))
PAGE_COVER = 0.6     # an image covering >=60% of the page is the scan itself, not a paste
TILE_REPEAT = 4      # a (w,h) size appearing >=4 times = a tiling grid (scan background)
FIELD_PAD = 15       # pt; a pasted signature may sit slightly outside the tight field line
MIN_W_PT, MIN_H_PT = 25, 6   # a pasted signature is wider than a speck/dot/glyph artifact
MAX_COVER_PASTE = 0.12       # bigger than this = a page region (half-page photo), not a signature
DPI_TOL = 0.15               # a real paste's resolution differs from the page scan's (came elsewhere)
DPI_FAMILY_MIN = 3           # >=3 sub-images sharing one dpi = a systematic OCR/e-sign overlay layer,
                             # not a lone paste (the real validated paste is a single odd-dpi outlier)


def field_boxes_pt(pdf_path, rel_for_cache):
    """Signature-field boxes in PDF points, from the OCR cache (no model load)."""
    cache = C.load_cache(rel_for_cache)
    if cache is None:
        return None, None
    items = cache["items"]; form = C.detect_form(items)
    if form not in FIELDS:
        return form, {}
    R = fitz.open(pdf_path)[0].rect
    scale = 2200 / max(R.width, R.height)
    boxes = {}
    for fld in get_fields(form, required_only=True):
        sig, dat = locate_field(items, fld)
        if sig is None:
            continue
        abox = C.sig_box_clip_date(sig["box"], dat["box"] if dat else None,
                                   offsets(TPL, form, fld["name"], "sig"))
        boxes[fld["name"]] = [v / scale for v in abox]
    return form, boxes


def _img_dpi(im):
    w = im["bbox"][2] - im["bbox"][0]
    return im["width"] / (w / 72) if w else 0.0


def detect_paste(pdf_path, boxes_pt):
    """Layer-1: a pasted signature = a separate image object in a signature field that is
    signature-sized, NOT page-covering / tiling, and bears a structural paste tell:
      (A) ALPHA — the image has a soft mask / transparency. Cut-out signatures keep a
          transparent background so they don't occlude the form; genuine scans are opaque.
          This is the strongest deterministic tell (fires even when dpi matches the page).
      (B) DPI MISMATCH — an opaque image whose resolution differs from the page scan and is
          NOT part of a same-dpi overlay FAMILY (>=3, = OCR/e-sign assembly layer, not a paste).
    Both are read from the PDF object table; no rendering."""
    doc = fitz.open(pdf_path); pg = doc[0]; R = pg.rect
    page_area = R.width * R.height
    smask_of = {im[0]: im[1] for im in pg.get_images(full=True)}   # xref -> soft-mask xref (0 = opaque)
    imgs = [im for im in pg.get_image_info(xrefs=True) if im["width"] > 1 and im["height"] > 1]
    if not imgs:
        doc.close(); return []
    page_dpi = _img_dpi(max(imgs, key=lambda im: (im["bbox"][2]-im["bbox"][0])*(im["bbox"][3]-im["bbox"][1])))
    sizes = Counter((round(im["bbox"][2]-im["bbox"][0]), round(im["bbox"][3]-im["bbox"][1])) for im in imgs)
    dpi_hist = Counter()                                       # dpi family of the non-page sub-images
    for im in imgs:
        b = im["bbox"]
        if ((b[2]-b[0])*(b[3]-b[1]))/page_area < PAGE_COVER:
            dpi_hist[round(_img_dpi(im)/25)*25] += 1
    hits = []
    for im in imgs:
        b = im["bbox"]; w, h = b[2]-b[0], b[3]-b[1]
        cov = (w*h) / page_area
        dpi = _img_dpi(im)
        if cov >= PAGE_COVER:                                  # the page scan itself
            continue
        if sizes[(round(w), round(h))] >= TILE_REPEAT:         # tiling grid (CamScanner)
            continue
        if w < MIN_W_PT or h < MIN_H_PT:                       # speck / dot / glyph artifact
            continue
        if cov >= MAX_COVER_PASTE:                             # a page region, not a signature
            continue
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        field = next((k for k, f in boxes_pt.items()
                      if f[0]-FIELD_PAD <= cx <= f[2]+FIELD_PAD and f[1]-FIELD_PAD <= cy <= f[3]+FIELD_PAD), None)
        if not field:
            continue
        has_alpha = smask_of.get(im["xref"], 0) != 0
        dpi_off = bool(page_dpi) and abs(dpi - page_dpi) / page_dpi >= DPI_TOL
        family = dpi_hist[round(dpi/25)*25] >= DPI_FAMILY_MIN
        if has_alpha:
            tell = "alpha/transparency (overlay, not a scan)"
        elif dpi_off and not family:
            tell = f"dpi {round(dpi)} != page {round(page_dpi)}"
        else:
            continue
        hits.append({"xref": im["xref"], "field": field, "dpi": round(dpi),
                     "place": f"{w:.0f}x{h:.0f}pt", "cover": round(cov, 3),
                     "page_dpi": round(page_dpi), "smask": smask_of.get(im["xref"], 0),
                     "tell": tell})
    doc.close()
    return hits


def error5(pdf_path, rel_for_cache):
    # DOC_NOT_SCAN (doc-level authenticity) was dropped: it is not a target error type and
    # over-fired on native/segmented PDFs. error5 now reports only the pasted-image signature.
    form, boxes = field_boxes_pt(pdf_path, rel_for_cache)
    hits = detect_paste(pdf_path, boxes) if boxes else []
    if hits:
        return {"verdict": "PASTED_IMAGE_SIGNATURE", "hits": hits}
    return {"verdict": "OK", "hits": []}


def main():
    ROOT = C.ROOT
    originals = sorted(glob.glob(os.path.join(ROOT, "**", "*.pdf"), recursive=True))
    print(f"=== regression on {len(originals)} ORIGINAL files (expect 0 PASTED) ===")
    counts = Counter()
    for f in originals:
        rel = os.path.relpath(f, ROOT)
        r = error5(f, rel)
        counts[r["verdict"]] += 1
        if r["verdict"] == "PASTED_IMAGE_SIGNATURE":
            print(f"  !! PASTED {os.path.basename(f)[:40]} -> {r['hits']}")
    print("  " + "  ".join(f"{k}={v}" for k, v in counts.items()))

    print("\n=== user test files ===")
    PROJ = r"C:\Users\25775\Desktop\OCR_research"
    # phone-paste (Type A): use the twin original's cache for field geometry
    paste = glob.glob(os.path.join(PROJ, "*G-28 Page 3_NG (not signed)_*.pdf"))
    twin = r"need_correction\G-28 Page 3\100394_Form G-28 Page 3_NG (not signed).pdf"
    if paste:
        r = error5(paste[0], twin)
        print(f"  phone-paste (A): {r['verdict']}  hits={r['hits']}")


if __name__ == "__main__":
    main()
