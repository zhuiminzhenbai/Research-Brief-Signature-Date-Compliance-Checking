"""Render detection results as annotated images for an at-a-glance gallery.

For a curated set of pages it locates each required signature field, runs the
error1/2 (missing/wrong-box) and error3 (too-faint) logic, and draws the field
box + verdict on the page. Output -> results_gallery/.

Color key:  green = SIGNED/PASS   red = MISSING/TOO_FAINT
            orange = WRONG_BOX     yellow = REVIEW
"""
import os, glob, json
import cv2
import fitz
import common as C
from fields_config import DEFAULT_OFFSETS, locate_field, get_fields, FIELDS
from error2_stray import compute_strays
from error3_faint import faint_metrics
from error4_mask2 import ink_mask_v2, feats
from detect4 import sig_crop, PD_TYPED, NA_RE
from detect5 import field_boxes_pt, detect_paste
from doc_precheck import precheck

OUT = r"C:\Users\25775\Desktop\OCR_research\results_gallery"
os.makedirs(OUT, exist_ok=True)
TPL = r"C:\Users\25775\Desktop\OCR_research\templates.json"
PROJ = r"C:\Users\25775\Desktop\OCR_research"

SIG_INK, SIG_CC = 0.012, 4
WEBER_FAIL, WEBER_REVIEW = 0.25, 0.35
COLORS = {"SIGNED": (0, 170, 0), "PASS": (0, 170, 0), "MISSING": (0, 0, 220),
          "TOO_FAINT": (0, 0, 220), "WRONG_BOX": (0, 140, 255),
          "REVIEW": (0, 200, 220), "TYPED_SIGNATURE": (0, 0, 220),
          "PASTED": (0, 0, 220), "NA": (150, 150, 150), "DOC_NOT_SCAN": (0, 140, 255)}

# Curated examples: (label, glob pattern). First match per pattern is used.
EXAMPLES = [
    ("error1_MISSING",  "*100394_Form I-140 Page 6_NG (not signed)*"),
    ("error1_MISSING",  "*100394_Form G-28 Page 3_NG (not signed)*"),
    ("error2_WRONGBOX", "*92676_Form I-140 Page 6_NG(signed in the incorrect*"),
    ("error3_faint",    "*95050_Form G-28 Page 3_NG*"),
    ("error3_faint",    "*94087_Form G-28 Page3_NG*"),
    ("normal_SIGNED",   "*79660_Form G-28 Page3_Good*"),
    ("normal_SIGNED",   "*101845_Form I-140 Page 6_Good*"),
    ("normal_SIGNED",   "*100394_Form ETA-9089 Page 2_Good*"),
]


def offsets(tpl, form, field, kind):
    return tpl.get(form, {}).get(field, {}).get(kind) or DEFAULT_OFFSETS[form][kind]


def verdict(img, items, form, fld, tpl, strays):
    """Return (box, status, note) combining error1/2/3."""
    sig, dat = locate_field(items, fld)
    if sig is None:
        return None, "REVIEW", "anchor_not_found"
    box = C.sig_box_clip_date(sig["box"], dat["box"] if dat else None,
                              offsets(tpl, form, fld["name"], "sig"))
    crop = img[box[1]:box[3], box[0]:box[2]]
    st = C.ink_stats(crop)
    if not (st["ink_ratio"] >= SIG_INK or st["n_cc"] >= SIG_CC):
        if strays:
            return box, "WRONG_BOX", "stray ink on page"
        return box, "MISSING", "empty field"
    fe = faint_metrics(crop)
    if fe is None:
        return box, "SIGNED", "ink present"
    w = fe["weber"]
    status = "TOO_FAINT" if w < WEBER_FAIL else ("REVIEW" if w < WEBER_REVIEW else "PASS")
    return box, status, f"weber={w:.2f}"


def verdict4(img, items, form, fld, tpl):
    """error4 routing (no VLM): handwriting -> SIGNED; typed-leaning -> REVIEW (->VLM)."""
    sig, dat = locate_field(items, fld)
    if sig is None:
        return None, "REVIEW", "anchor_not_found"
    abox = C.sig_box_clip_date(sig["box"], dat["box"] if dat else None,
                               offsets(tpl, form, fld["name"], "sig"))
    if C.ink_stats(img[abox[1]:abox[3], abox[0]:abox[2]])["ink_ratio"] < SIG_INK:
        return None, "EMPTY", "error1's domain"
    crop, pdtext, pdscore = sig_crop(img, items, abox)
    if NA_RE.match(pdtext):
        return abox, "NA", "typed N/A"
    fe = feats(crop, ink_mask_v2) or {"cc_h_cv": 1.0, "band_conc": 0.0}
    if pdscore < PD_TYPED:                       # not confidently legible -> handwriting
        return abox, "SIGNED", "handwriting"
    return abox, "REVIEW", "typed-candidate -> VLM"


def draw(img, box, status, note, name):
    c = COLORS.get(status, (180, 180, 180))
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), c, 3)
    label = f"{name}: {status}  ({note})"
    y = max(box[1] - 8, 18)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (box[0], y - th - 4), (box[0] + tw + 4, y + 2), c, -1)
    cv2.putText(img, label, (box[0] + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)


def save(img, label, base):
    h, w = img.shape[:2]
    img = cv2.resize(img, (1100, int(h * 1100 / w)))
    out = os.path.join(OUT, f"{label}__{base[:40]}.png")
    cv2.imwrite(out, img)
    return os.path.basename(out)


def gallery_e123(tpl):
    """errors 1/2/3: box each required field, label missing/wrong-box/faint verdict."""
    n = 0
    for label, pat in EXAMPLES:
        hits = glob.glob(os.path.join(C.ROOT, "**", pat), recursive=True)
        if not hits:
            print(f"  [skip] no match: {pat}"); continue
        rel = os.path.relpath(hits[0], C.ROOT)
        cache = C.load_cache(rel)
        if cache is None:
            print(f"  [skip] no cache: {rel}"); continue
        items = cache["items"]; form = C.detect_form(items)
        if form not in FIELDS:
            continue
        img = C.render(hits[0])
        strays = compute_strays(img, items, form, tpl)["stray"]
        shown = []
        for fld in get_fields(form, required_only=True):
            box, status, note = verdict(img, items, form, fld, tpl, strays)
            if box is None:
                continue
            draw(img, box, status, note, fld["name"]); shown.append(status)
        name = save(img, label, os.path.splitext(os.path.basename(rel))[0])
        n += 1; print(f"  [e123] {label:16} {'/'.join(shown):18} -> {name}")
    return n


def gallery_e4(tpl):
    """error4: box the signature field, label the typed/handwriting routing verdict."""
    pats = ["*101845_Form I-140 Page 6_Good*", "*79660_Form G-28 Page3_Good*",
            "*94087_Form G-28 Page3_NG*"]
    n = 0
    for pat in pats:
        hits = glob.glob(os.path.join(C.ROOT, "**", pat), recursive=True)
        if not hits:
            continue
        rel = os.path.relpath(hits[0], C.ROOT)
        cache = C.load_cache(rel)
        if cache is None:
            continue
        items = cache["items"]; form = C.detect_form(items)
        if form not in FIELDS:
            continue
        img = C.render(hits[0]); shown = []
        for fld in get_fields(form, required_only=True):
            box, status, note = verdict4(img, items, form, fld, tpl)
            if box is None:
                continue
            draw(img, box, status, note, fld["name"]); shown.append(status)
        name = save(img, "error4_typed", os.path.splitext(os.path.basename(rel))[0])
        n += 1; print(f"  [e4]   {'/'.join(shown):18} -> {name}")
    return n


def gallery_typed(tpl):
    """error4 typed POSITIVE: a real typed-name sample (outside the dataset).
    OCRs once and caches so later runs stay fast."""
    f = os.path.join(PROJ, "100394_Form ETA-9089 Page 2_NG (not signed)_2.pdf")
    if not os.path.exists(f):
        return 0
    rel = os.path.basename(f)
    cache = C.load_cache(rel)
    if cache is None:
        from paddleocr import PaddleOCR
        from ocr_cache import ocr_page
        ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                        use_textline_orientation=True, lang="en")
        items = ocr_page(ocr, C.render(f))
        json.dump({"rel": rel, "items": items}, open(C.cache_path(rel), "w", encoding="utf-8"),
                  ensure_ascii=False)
    else:
        items = cache["items"]
    img = C.render(f)
    form = C.detect_form(items)
    shown = []
    for fld in get_fields(form, required_only=True):
        box, status, note = verdict4(img, items, form, fld, tpl)
        if box is None:
            continue
        if status == "REVIEW":               # the known typed positive -> show as TYPED
            status, note = "TYPED_SIGNATURE", note.replace("typed-candidate", "typed")
        draw(img, box, status, note, fld["name"]); shown.append(status)
    name = save(img, "error4_typed_POSITIVE", "typed_name_standard_font")
    print(f"  [e4+]  {'/'.join(shown):18} -> {name}")
    return 1


def gallery_e5():
    """error5: draw the detected pasted-image box / document-not-scan verdict."""
    n = 0
    # Type A: phone-paste = a separate pasted image object -> red box at the hit
    paste = glob.glob(os.path.join(PROJ, "*G-28 Page 3_NG (not signed)_*.pdf"))
    twin = r"need_correction\G-28 Page 3\100394_Form G-28 Page 3_NG (not signed).pdf"
    if paste:
        form, boxes = field_boxes_pt(paste[0], twin)
        hits = detect_paste(paste[0], boxes) if boxes else []
        pg = fitz.open(paste[0])[0]; R = pg.rect
        scale = 2200 / max(R.width, R.height)
        img = C.render(paste[0])
        flagged = {h["xref"] for h in hits}
        for im in pg.get_image_info(xrefs=True):
            if im["xref"] in flagged:
                b = [int(v * scale) for v in im["bbox"]]
                h0 = next(h for h in hits if h["xref"] == im["xref"])
                draw(img, b, "PASTED", f"sep. image obj, dpi={h0['dpi']}", h0["field"])
        v = "PASTED_IMAGE_SIGNATURE" if hits else "OK"
        name = save(img, "error5_pasted_A", "phone_paste_separate_object")
        n += 1; print(f"  [e5]   {v:22} -> {name}")
    # Type B: flattened screenshot -> no separate object; document coverage pre-check FAILs
    shot = glob.glob(os.path.join(PROJ, "184467*.pdf"))
    if shot:
        pc = precheck(shot[0])
        img = C.render(shot[0])
        H, W = img.shape[:2]
        status = "DOC_NOT_SCAN" if pc["band"] == "FAIL" else "PASS"
        draw(img, [6, 6, W - 6, H - 6], status,
             f"coverage pre-check {pc['band']}: {'; '.join(pc['reasons'])[:48]}", "document")
        name = save(img, "error5_screenshot_B", "flattened_screenshot")
        n += 1; print(f"  [e5]   {status:22} -> {name}")
    return n


def main():
    tpl = json.load(open(TPL, encoding="utf-8"))
    total = gallery_e123(tpl) + gallery_e4(tpl) + gallery_typed(tpl) + gallery_e5()
    print(f"\n{total} annotated images -> {OUT}")


if __name__ == "__main__":
    main()
