"""error4 (TYPED_SIGNATURE) detector — assembled from the research pieces.

Runs only on SIGNED required cells (empty cells are error1's job). For each:
  1. crop the signature region (tight OCR box if present, else anchor cell with
     printed labels painted white);
  2. cheap signals: Tesseract readability (print-only OCR) + structural features
     (cc_h_cv / band_conc on the v2 full-span-only line-removed mask);
  3. obvious handwriting (Tesseract reads nothing AND high glyph-height CV) is
     auto-cleared as SIGNED — no VLM call;
  4. everything typed-leaning goes to the pluggable VLM (vlm_classify): typed ->
     TYPED_SIGNATURE, handwritten -> SIGNED, uncertain (or no VLM configured) -> REVIEW.
Typed "N/A" is an intentional not-applicable mark, reported as NA (not error4).

No labeled typed positives exist in the dataset, so the eval reports the band
breakdown over the 94 genuine handwritten cells: ideal = mostly SIGNED, 0 TYPED
false-positives, the rest REVIEW (deferred). A real VLM backend resolves REVIEWs.
"""
import os, glob, json, re
import numpy as np
import cv2
import fitz
import common as C
from fields_config import FIELDS, DEFAULT_OFFSETS, locate_field, get_fields
from error4_mask2 import ink_mask_v2, feats
from error4_fontmatch import font_score as _font_score, TF as _FONT_TF
import vlm_classify

OUT = r"C:\Users\25775\Desktop\OCR_research\detect4_out"
CROPS = os.path.join(OUT, "crops")
os.makedirs(CROPS, exist_ok=True)
TPL = r"C:\Users\25775\Desktop\OCR_research\templates.json"

SIG_INK = 0.012            # cell considered signed (same as detect12)
PD_TYPED = 0.85            # below this OCR conf = clear handwriting -> auto-pass
PD_TYPED_HARD = 0.99       # >= this = standard-font typed -> local TYPED verdict, no VLM
NATIVE_PAGE_MAX = 50       # Layer-1 only when whole page has <= this many native words:
                           # a scan + a few OVERLAID typed words (real samples=4) is a typed
                           # paste; a full-page text layer (searchable/native PDF, 278-343
                           # words here) is NOT separable by "field has text" -> use pixels.
CCHCV_TYPED = 0.55         # glyph-height CV (recorded only; no longer routes)
VEC_GLYPH_MIN = 8          # >= this many glyph-sized VECTOR fills in a field (on a scan-like
                           # page) = text drawn as outlines/curves = typed/outlined signature.
                           # 0 on every genuine scan AND on the 2 typed-as-text samples tested
                           # (those are text objects, caught by Layer-1 text). No outlined-text
                           # positive exists yet -> recall UNVALIDATED, but FP-safe (scans = 0);
                           # this is the deterministic catch IF an outlined-text evasion appears.
# Routing. The DETERMINISTIC typed signal is Layer-1 (a native PDF text object in the
# field, handled above) — that is the ONLY thing that yields a local TYPED verdict.
# OCR confidence does NOT separate typed from handwriting on real data: the pd>=0.99
# heuristic (calibrated on 2 typed + 94 handwritten samples) FALSE-fired on legible
# handwritten names in the 400+ test set (neat cursive names also read 0.99+),
# so it was removed. cc_h_cv and stroke-width CV are likewise non-separable. Pixel/OCR
# routing therefore only AUTO-CLEARS clear handwriting; anything legible goes to the VLM:
#   pd_score < 0.85    -> handwriting -> SIGNED, no VLM call
#   pd_score >= 0.85   -> legible but ambiguous -> VLM (REVIEW if no VLM configured)
NA_RE = re.compile(r"^\s*n\.?/?\s*a\.?\s*$", re.I)
ALPHA = re.compile(r"[A-Za-z]")
LABEL_WORDS = re.compile(
    r"signat|\bdate\b|\bpart\b|\bpage\b|inform|petition|interpret|represent|document|"
    r"contact|certif|consent|number|address|email|family|given|\bname\b|telephone|"
    r"addition|authoriz|signatory|client|attorney|\blaw\b|student|employer|declarat|"
    r"foreign|worker|business|organiz|middle|daytime|mobile|group|firm|\bof\b|\bthe\b|"
    r"mm/dd|yyyy|edition|form i-|eta-", re.I)


def offsets(tpl, form, field, kind):
    return tpl.get(form, {}).get(field, {}).get(kind) or DEFAULT_OFFSETS[form][kind]


def page_words(pdf_path):
    """Native PDF text words + the render-px scale (long side 2200).
    A pure scan returns []; a digital/typed PDF returns its text objects.
    This is error4's deterministic Layer-1 (cf. error5 structural detection)."""
    doc = fitz.open(pdf_path); pg = doc[0]; R = pg.rect
    words = pg.get_text("words")           # (x0,y0,x1,y1,word,block,line,wordno) in PDF pt
    doc.close()
    return words, 2200 / max(R.width, R.height)


def native_in_field(words, scale, abox_px):
    """Native text words whose center falls inside the signature field (abox in render px)."""
    pt = [c / scale for c in abox_px]
    return [w[4] for w in words
            if pt[0] <= (w[0] + w[2]) / 2 <= pt[2] and pt[1] <= (w[1] + w[3]) / 2 <= pt[3]]


def page_glyph_fills(pdf_path):
    """Glyph-sized VECTOR fill rectangles (PDF pt) — text drawn as outlines/curves, which
    get_text() can't see. Excludes large shapes (form boxes/lines). A raster scan has none."""
    doc = fitz.open(pdf_path); pg = doc[0]
    rects = []
    for d in pg.get_drawings():
        if d.get("fill") is None:                  # only filled paths (glyph bodies), not strokes
            continue
        r = d["rect"]; w, h = r.width, r.height
        if 1 <= w <= 40 and 1 <= h <= 40:          # glyph-sized (not the field box / page rules)
            rects.append((r.x0, r.y0, r.x1, r.y1))
    doc.close()
    return rects


def vector_glyphs_in_field(rects, scale, abox_px):
    """Count glyph-sized vector fills whose center is inside the signature field."""
    pt = [c / scale for c in abox_px]
    return sum(1 for x0, y0, x1, y1 in rects
               if pt[0] <= (x0 + x1) / 2 <= pt[2] and pt[1] <= (y0 + y1) / 2 <= pt[3])


def inside(b, B, frac=0.45):
    b = [int(v) for v in b]
    ba = (b[2]-b[0]) * (b[3]-b[1])
    ix = max(0, min(b[2], B[2]) - max(b[0], B[0]))
    iy = max(0, min(b[3], B[3]) - max(b[1], B[1]))
    return ba and ix*iy >= frac * ba


def sig_crop(img, items, abox):
    """Tight OCR box for the signature if present, else the cell with labels whited."""
    content = [it for it in items if inside(it["box"], abox)
               and not LABEL_WORDS.search(it["text"]) and ALPHA.search(it["text"])]
    if content:
        best = max(content, key=lambda it: it.get("score", 0))
        ob = [int(v) for v in best["box"]]; pad = 8
        crop = img[max(0, ob[1]-pad):ob[3]+pad, max(0, ob[0]-pad):ob[2]+pad].copy()
        return crop, best["text"], best.get("score", 0)
    crop = img[abox[1]:abox[3], abox[0]:abox[2]].copy()
    for it in items:
        if LABEL_WORDS.search(it["text"]):
            b = [int(v) for v in it["box"]]
            x0, y0 = max(0, b[0]-abox[0]), max(0, b[1]-abox[1])
            x1, y1 = min(crop.shape[1], b[2]-abox[0]), min(crop.shape[0], b[3]-abox[1])
            if x1 > x0 and y1 > y0:
                crop[y0:y1, x0:x1] = 255
    return crop, "", -1.0


def process(rel, tpl, use_vlm=True):
    cache = C.load_cache(rel)
    if cache is None:
        return None
    items = cache["items"]; form = C.detect_form(items)
    if form not in FIELDS:
        return {"rel": rel, "form": form, "fields": {}, "note": "no_config"}
    pdf_path = os.path.join(C.ROOT, rel)
    img = C.render(pdf_path)
    nwords, nscale = page_words(pdf_path)          # Layer-1a: native PDF text objects
    use_native = len(nwords) <= NATIVE_PAGE_MAX    # else searchable/native PDF -> not usable
    glyph_fills = page_glyph_fills(pdf_path)       # Layer-1b: text drawn as vector outlines
    base = os.path.basename(rel)
    res = {"rel": rel, "form": form, "fields": {}}
    for fld in get_fields(form, required_only=True):
        sig, dat = locate_field(items, fld)
        if sig is None:
            res["fields"][fld["name"]] = {"status": "REVIEW", "reason": "anchor_not_found"}
            continue
        abox = C.sig_box_clip_date(sig["box"], dat["box"] if dat else None,
                                   offsets(tpl, form, fld["name"], "sig"))
        native = native_in_field(nwords, nscale, abox) if use_native else []
        if native:                                  # overlaid digital text in a scan = typed
            res["fields"][fld["name"]] = {
                "status": "TYPED_SIGNATURE", "layer": "pdf-text",
                "reason": "native PDF text object in signature field",
                "native_text": " ".join(native)}
            continue
        nglyph = vector_glyphs_in_field(glyph_fills, nscale, abox) if use_native else 0
        if nglyph >= VEC_GLYPH_MIN:                 # text drawn as vector outlines = typed
            res["fields"][fld["name"]] = {
                "status": "TYPED_SIGNATURE", "layer": "pdf-vector",
                "reason": f"{nglyph} glyph-sized vector fills in field (outlined typed text)"}
            continue
        if C.ink_stats(img[abox[1]:abox[3], abox[0]:abox[2]])["ink_ratio"] < SIG_INK:
            continue  # empty -> error1's domain, not error4
        crop, pdtext, pdscore = sig_crop(img, items, abox)
        if NA_RE.match(pdtext):
            res["fields"][fld["name"]] = {"status": "NA", "reason": "typed N/A (not-applicable)",
                                          "pd": pdtext}
            continue
        fe = feats(crop, ink_mask_v2) or {"cc_h_cv": 1.0, "band_conc": 0.0}
        info = {"pd_score": round(pdscore, 3), "pd": pdtext,
                "cc_h_cv": fe["cc_h_cv"], "band_conc": fe["band_conc"]}
        if pdscore < PD_TYPED:
            info["status"] = "SIGNED"; info["reason"] = "handwriting (low ocr conf)"
        else:
            # Layer-2 PREFILTER: font-template match selects typed-candidates. It cannot
            # decide alone (neat hand-printing scores in the typed band too), only narrows
            # who needs the VLM terminal verdict.
            fs = _font_score(crop, pdtext)
            info["font_score"] = round(fs, 3)
            if fs < _FONT_TF:
                info["status"] = "SIGNED"; info["reason"] = f"handwriting (font={fs:.2f})"
            elif use_vlm:
                cp = os.path.join(CROPS, f"{base[:24]}__{fld['name']}.png".replace(" ", "_"))
                cv2.imwrite(cp, crop)
                v = vlm_classify.classify(cp)
                info["vlm"] = v["label"]; info["vlm_backend"] = v["backend"]
                info["status"] = {"typed": "TYPED_SIGNATURE", "handwritten": "SIGNED"}.get(
                    v["label"], "REVIEW")
                info["reason"] = f"vlm={v['label']} (font={fs:.2f})"
            else:
                # typed-candidate but no VLM -> defer to human REVIEW (font-match alone
                # cannot separate neat hand-printing from typed).
                info["status"] = "REVIEW"; info["reason"] = f"typed-candidate, no-vlm (font={fs:.2f})"
        res["fields"][fld["name"]] = info
    return res


def main():
    import sys
    use_vlm = "--novlm" not in sys.argv
    tpl = json.load(open(TPL, encoding="utf-8"))
    files = sorted(glob.glob(os.path.join(C.ROOT, "**", "*.pdf"), recursive=True))
    rows, tally = [], {}
    for f in files:
        r = process(os.path.relpath(f, C.ROOT), tpl, use_vlm)
        if r is None or "note" in r:
            continue
        rows.append(r)
        for name, info in r["fields"].items():
            tally[info["status"]] = tally.get(info["status"], 0) + 1
    json.dump(rows, open(os.path.join(OUT, "detect4.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("=" * 64)
    print("error4 status tally over all required signed cells:")
    for k in sorted(tally):
        print(f"   {k:16} {tally[k]}")
    typed = [(r["rel"], n, i) for r in rows for n, i in r["fields"].items()
             if i["status"] == "TYPED_SIGNATURE"]
    rev = [(r["rel"], n, i) for r in rows for n, i in r["fields"].items()
           if i["status"] == "REVIEW" and "anchor" not in i.get("reason", "")]
    print(f"\nTYPED_SIGNATURE flags ({len(typed)}):")
    for rel, n, i in typed:
        print(f"   {os.path.basename(rel)[:34]} :: {n}  pd='{i.get('pd')}' pd_score={i.get('pd_score')} cc_h_cv={i.get('cc_h_cv')}")
    print(f"\nsent to VLM / REVIEW (typed-candidates) ({len(rev)}):")
    for rel, n, i in rev[:20]:
        print(f"   {os.path.basename(rel)[:34]} :: {n}  pd='{i.get('pd')}' pd_score={i.get('pd_score')} cc_h_cv={i.get('cc_h_cv')} vlm={i.get('vlm','-')}")


if __name__ == "__main__":
    main()
