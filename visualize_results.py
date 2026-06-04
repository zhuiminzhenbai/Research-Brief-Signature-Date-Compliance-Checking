"""Render detection results as annotated images for an at-a-glance gallery.

For a curated set of pages it locates each required signature field, runs the
error1/2 (missing/wrong-box) and error3 (too-faint) logic, and draws the field
box + verdict on the page. Output -> results_gallery/.

Color key:  green = SIGNED/PASS   red = MISSING/TOO_FAINT
            orange = WRONG_BOX     yellow = REVIEW
"""
import os, glob, json
import cv2
import common as C
from fields_config import DEFAULT_OFFSETS, locate_field, get_fields, FIELDS
from error2_stray import compute_strays
from error3_faint import faint_metrics

OUT = r"C:\Users\25775\Desktop\OCR_research\results_gallery"
os.makedirs(OUT, exist_ok=True)
TPL = r"C:\Users\25775\Desktop\OCR_research\templates.json"

SIG_INK, SIG_CC = 0.012, 4
WEBER_FAIL, WEBER_REVIEW = 0.25, 0.35
COLORS = {"SIGNED": (0, 170, 0), "PASS": (0, 170, 0), "MISSING": (0, 0, 220),
          "TOO_FAINT": (0, 0, 220), "WRONG_BOX": (0, 140, 255),
          "REVIEW": (0, 200, 220)}

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


def draw(img, box, status, note, name):
    c = COLORS.get(status, (180, 180, 180))
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), c, 3)
    label = f"{name}: {status}  ({note})"
    y = max(box[1] - 8, 18)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (box[0], y - th - 4), (box[0] + tw + 4, y + 2), c, -1)
    cv2.putText(img, label, (box[0] + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)


def main():
    tpl = json.load(open(TPL, encoding="utf-8"))
    made = 0
    for label, pat in EXAMPLES:
        hits = glob.glob(os.path.join(C.ROOT, "**", pat), recursive=True)
        if not hits:
            print(f"  [skip] no match: {pat}")
            continue
        rel = os.path.relpath(hits[0], C.ROOT)
        cache = C.load_cache(rel)
        if cache is None:
            print(f"  [skip] no cache: {rel}")
            continue
        items = cache["items"]
        form = C.detect_form(items)
        if form not in FIELDS:
            continue
        img = C.render(hits[0])
        strays = compute_strays(img, items, form, tpl)["stray"]
        shown = []
        for fld in get_fields(form, required_only=True):
            box, status, note = verdict(img, items, form, fld, tpl, strays)
            if box is None:
                continue
            draw(img, box, status, note, fld["name"])
            shown.append(status)
        # downscale for a compact gallery image
        h, w = img.shape[:2]
        scale = 1100 / w
        img = cv2.resize(img, (1100, int(h * scale)))
        base = os.path.splitext(os.path.basename(rel))[0][:40]
        out = os.path.join(OUT, f"{label}__{base}.png")
        cv2.imwrite(out, img)
        made += 1
        print(f"  [{made}] {label:16} {'/'.join(shown):20} -> {os.path.basename(out)}")
    print(f"\n{made} annotated images -> {OUT}")


if __name__ == "__main__":
    main()
