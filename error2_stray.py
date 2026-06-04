"""error2 (SIGNATURE_WRONG_BOX) via page-level stray-ink detection.

Strategy (no per-monitor calibration needed):
  1. Compute a page-wide ink mask (adaptive threshold + form-line removal).
  2. Subtract every OCR text box (printed labels + typed/filled data are recognized
     by OCR; handwriting generally is not) -> remaining ink = handwriting candidates.
  3. Keep signature-like connected components (size/shape filter).
  4. "Allowed" zones = all required sig boxes + required date boxes (+ tuned optional).
     Any signature-like cluster OUTSIDE all allowed zones = stray (misplaced) signature.
  5. WRONG_BOX when a required sig box is empty but a stray signature cluster exists.
"""
import os, sys, json, re
import numpy as np
import cv2
import common as C
from fields_config import FIELDS, DEFAULT_OFFSETS, locate_field, get_fields

OUT = r"C:\Users\25775\Desktop\OCR_research\stray_out"
os.makedirs(OUT, exist_ok=True)
TPL = r"C:\Users\25775\Desktop\OCR_research\templates.json"

# Mask mode: "field" = remove ink EXPLAINED by a field (printed labels + typed data
# outside signature cells; "N/A" inside a cell); keep unexplained ink (incl. neat
# signatures OCR happens to read). "conf" = legacy: remove any OCR text >= score_thr.
MASK_MODE = "field"
LABEL_RE = re.compile(
    r"signat|\bdate\b|\bpart\b|\bpage\b|inform|petition|interpret|represent|document|"
    r"contact|certif|consent|number|address|email|family|given|\bname\b|telephone|"
    r"addition|authoriz|signatory|client|attorney|\blaw\b|student|employer|declarat|"
    r"foreign|worker|business|organiz|middle|daytime|mobile|group|firm|\bof\b|\bthe\b|"
    r"mm/dd|yyyy|edition|form i-|eta-|u.s.|department|labor|approv|expirat|omb", re.I)
NA_RE = re.compile(r"^\s*n\.?\s*/?\s*a\.?\s*$", re.I)


def _inside_any(b, regions, frac=0.4):
    b = [int(v) for v in b]
    ba = (b[2]-b[0]) * (b[3]-b[1])
    for R in regions:
        ix = max(0, min(b[2], R[2]) - max(b[0], R[0]))
        iy = max(0, min(b[3], R[3]) - max(b[1], R[1]))
        if ba and ix*iy >= frac * ba:
            return True
    return False


def offsets(tpl, form, field, kind):
    return tpl.get(form, {}).get(field, {}).get(kind) or DEFAULT_OFFSETS[form][kind]


def page_handwriting_mask(img, items, protected=(), score_thr=0.42):
    """Full-page ink with form lines and CONFIDENT OCR text removed.

    Only high-confidence OCR boxes are treated as printed/typed text and masked;
    handwriting that OCR misreads tends to score low, so it survives the mask.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 15)
    H, W = bw.shape
    # remove long horizontal/vertical rules (form boxes/underlines)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, W // 50), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, H // 50)))
    lines = cv2.dilate(cv2.erode(bw, hk), hk) | cv2.dilate(cv2.erode(bw, vk), vk)
    ink = cv2.subtract(bw, lines)
    # remove ink that is EXPLAINED by a non-signature element. pad scales with box
    # height so bold strokes bleeding past the tight box are covered.
    txt = np.zeros((H, W), np.uint8)
    for it in items:
        b = it["box"]; text = it.get("text", "")
        sc = it.get("score", 1.0)
        if MASK_MODE == "field":
            is_label = bool(LABEL_RE.search(text))
            in_cell = _inside_any(b, protected)
            if is_label:
                drop = True                              # printed form label/instruction
            elif in_cell:
                drop = bool(NA_RE.match(text.strip()))   # keep handwriting; drop typed "N/A"
            else:
                drop = sc >= score_thr                   # typed/printed data outside cells
        else:  # legacy confidence mode
            drop = sc >= score_thr
        if not drop:
            continue
        pad = max(2, int(0.18 * (b[3] - b[1])))
        x0, y0 = max(0, int(b[0]) - pad), max(0, int(b[1]) - pad)
        x1, y1 = min(W, int(b[2]) + pad), min(H, int(b[3]) + pad)
        txt[y0:y1, x0:x1] = 255
    hand = cv2.subtract(ink, txt)
    hand = cv2.medianBlur(hand, 3)
    return hand, ink


def clusters(hand, img_shape, y_lo, y_hi, med_lh):
    """Signature-like connected components within the field band [y_lo, y_hi].

    y_hi is the bottom of the lowest located field (+margin); the page-bottom
    barcode/footer (structurally below the fields) is thus usually excluded. On
    forms whose signature sits near the bottom (I-140 P8) the barcode can fall
    inside the band, so it is also rejected by its texture: wide + dense + low.
    """
    H, W = img_shape[:2]
    page_area = H * W
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
    closed = cv2.dilate(hand, k)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats((closed > 0).astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cyc = y + h / 2
        ink_px = int((hand[y:y+h, x:x+w] > 0).sum())   # real ink before closing
        fill = ink_px / area if area else 0
        if w < 40 or ink_px < page_area * 0.00018:      # too small -> noise/dots
            continue
        if w > 12 * max(h, 1) and h < 22:               # very wide & thin -> rule/band
            continue
        if w > 6 * med_lh and fill > 0.35 and cyc > 0.85 * H:   # dense wide strip low -> barcode
            continue
        if cyc < y_lo or cyc > y_hi:                     # outside the field band
            continue
        out.append({"box": [int(x), int(y), int(x+w), int(y+h)],
                    "area": int(area), "ink_px": ink_px,
                    "fill": round(fill, 4)})
    return out


def overlaps(box, zones, pad=0, frac=0.35):
    """True if box overlaps any zone (zones grown by `pad`) by >= frac of box area.
    Padding absorbs a field's own signature/date ink that spills past its box."""
    bx = (box[2]-box[0]) * (box[3]-box[1])
    for z in zones:
        zx0, zy0, zx1, zy1 = z[0]-pad, z[1]-pad, z[2]+pad, z[3]+pad
        ix = max(0, min(box[2], zx1) - max(box[0], zx0))
        iy = max(0, min(box[3], zy1) - max(box[1], zy0))
        if bx and ix*iy >= frac * bx:
            return True
    return False


def compute_strays(img, items, form, tpl):
    """Page-level stray-signature clusters (handwriting-like ink outside every
    required sig/date box). Reusable by detect12 as the WRONG_BOX arbiter.

    Returns dict: stray (list), clusters (all), allowed (zones), req_boxes (name->box).
    """
    # allowed zones = required sig + date boxes; field band spans ALL located
    # fields (required + optional) so the page-bottom barcode/footer is excluded.
    allowed = []
    req_boxes = {}
    protected = []          # ALL signature cells (req+opt): handwriting here is kept
    tops, bots, lhs = [], [], []
    for fld in FIELDS[form]:
        sig, dat = locate_field(items, fld)
        if sig is None:
            if fld["required"]:
                req_boxes[fld["name"]] = None
            continue
        sb = C.sig_box_clip_date(sig["box"], dat["box"] if dat else None,
                                 offsets(tpl, form, fld["name"], "sig"))
        tops.append(sig["box"][1]); bots.append(sb[3]); lhs.append(C.h(sig["box"]))
        protected.append(sb)
        db = C.date_box(dat["box"], fld.get("date_dir", "right")) if dat else None
        if db is not None:
            bots.append(db[3])
        if fld["required"]:
            req_boxes[fld["name"]] = sb
            allowed.append(sb)
            if db is not None:
                allowed.append(db)

    H = img.shape[0]
    med_lh = np.median(lhs) if lhs else 30
    m = 1.2 * med_lh
    y_lo = (min(tops) - m) if tops else 0.06 * H
    y_hi = (max(bots) + m) if bots else 0.95 * H

    hand, _ = page_handwriting_mask(img, items, protected=protected)
    cls = clusters(hand, img.shape, y_lo, y_hi, med_lh)
    zone_pad = int(1.2 * med_lh)
    stray = [c for c in cls if not overlaps(c["box"], allowed, pad=zone_pad)]
    return {"stray": stray, "clusters": cls, "allowed": allowed, "req_boxes": req_boxes}


def analyze(rel, tpl, save=True):
    cache = C.load_cache(rel)
    if cache is None:
        return None
    items = cache["items"]
    form = C.detect_form(items)
    if form not in FIELDS:
        return {"form": form, "empty": [], "n_cls": 0, "n_stray": 0,
                "stray": [], "wrong_box": False}
    img = C.render(os.path.join(C.ROOT, rel))

    sr = compute_strays(img, items, form, tpl)
    stray, cls, allowed, req_boxes = sr["stray"], sr["clusters"], sr["allowed"], sr["req_boxes"]

    # which required boxes are empty?
    empty = [n for n, b in req_boxes.items()
             if b is not None and not _signed(img, b)]

    wrong_box = len(empty) > 0 and len(stray) > 0
    if save:
        vis = img.copy()
        for z in allowed:
            cv2.rectangle(vis, (z[0], z[1]), (z[2], z[3]), (0, 170, 0), 2)
        for c in cls:
            b = c["box"]
            col = (0, 0, 230) if c in stray else (180, 180, 180)
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), col, 2)
        for n in empty:
            b = req_boxes[n]
            cv2.putText(vis, f"{n} EMPTY", (b[0], max(20, b[1]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 230), 2)
        out = os.path.join(OUT, "stray_" + os.path.basename(rel).replace(".pdf", ".png"))
        cv2.imwrite(out, vis)
        print(f"[{form:9}] {os.path.basename(rel)}")
        print(f"   empty required: {empty}")
        print(f"   clusters={len(cls)} stray={len(stray)}")
        for c in stray:
            print(f"     stray @ {c['box']}  area={c['area']} ink={c['ink_px']} fill={c['fill']}")
        print(f"   -> {out}")
    return {"form": form, "empty": empty, "n_cls": len(cls), "n_stray": len(stray),
            "stray": stray, "wrong_box": wrong_box}


def _signed(img, box, ink_thr=0.012, cc_thr=4):
    st = C.ink_stats(img[box[1]:box[3], box[0]:box[2]])
    return st["ink_ratio"] >= ink_thr or st["n_cc"] >= cc_thr


def batch(tpl):
    import glob
    files = sorted(glob.glob(os.path.join(C.ROOT, "**", "*.pdf"), recursive=True))
    flagged = []
    for f in files:
        rel = os.path.relpath(f, C.ROOT)
        r = analyze(rel, tpl, save=False)
        if r is None:
            continue
        tag = "WRONG_BOX" if r["wrong_box"] else ("stray" if r["n_stray"] else "")
        if r["n_stray"] or r["empty"]:
            print(f"[{r['form']:9}] empty={r['empty']!s:30} stray={r['n_stray']} {tag}  {os.path.basename(rel)}")
        if r["wrong_box"]:
            flagged.append(rel)
    print(f"\nWRONG_BOX flagged: {len(flagged)}")
    for x in flagged:
        print("  ", x)


def main():
    tpl = json.load(open(TPL, encoding="utf-8"))
    if len(sys.argv) > 1:
        for rel in sys.argv[1:]:
            analyze(rel, tpl)
    else:
        batch(tpl)


if __name__ == "__main__":
    main()
