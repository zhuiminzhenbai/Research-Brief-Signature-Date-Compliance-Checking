"""error3 (SIGNATURE_TOO_FAINT) detector.

Runs on SIGNED required cells (empty = error1). Isolates stroke pixels (adaptive
threshold + line removal) and measures their darkness vs the paper background.
Decision band on Weber contrast = (paper - stroke_median) / paper:
  >= 0.35  -> PASS  (acceptable)
  0.25..35 -> REVIEW (borderline faint -> human)
  <  0.25  -> FAIL  (TOO_FAINT)
Thresholds calibrated so all 94 real signatures pass/REVIEW (min weber 0.34) while
synthetically dimmed (visually illegible) signatures fall below 0.25.
"""
import os, glob, json
import numpy as np
import cv2
import common as C
from fields_config import FIELDS, DEFAULT_OFFSETS, locate_field, get_fields
from error3_faint import faint_metrics

OUT = r"C:\Users\25775\Desktop\OCR_research\detect3_out"
os.makedirs(OUT, exist_ok=True)
TPL = r"C:\Users\25775\Desktop\OCR_research\templates.json"

WEBER_FAIL = 0.25
WEBER_REVIEW = 0.35
SIG_INK = 0.012


def offsets(tpl, form, field, kind):
    return tpl.get(form, {}).get(field, {}).get(kind) or DEFAULT_OFFSETS[form][kind]


def band(weber):
    if weber < WEBER_FAIL:
        return "TOO_FAINT"
    if weber < WEBER_REVIEW:
        return "REVIEW"
    return "PASS"


def process(rel, tpl):
    cache = C.load_cache(rel)
    if cache is None:
        return None
    items = cache["items"]; form = C.detect_form(items)
    if form not in FIELDS:
        return {"rel": rel, "form": form, "fields": {}, "note": "no_config"}
    img = C.render(os.path.join(C.ROOT, rel))
    res = {"rel": rel, "form": form, "fields": {}}
    for fld in get_fields(form, required_only=True):
        sig, dat = locate_field(items, fld)
        if sig is None:
            res["fields"][fld["name"]] = {"status": "REVIEW", "reason": "anchor_not_found"}
            continue
        box = C.sig_box_clip_date(sig["box"], dat["box"] if dat else None,
                                  offsets(tpl, form, fld["name"], "sig"))
        crop = img[box[1]:box[3], box[0]:box[2]]
        if C.ink_stats(crop)["ink_ratio"] < SIG_INK:
            continue  # empty -> error1
        fe = faint_metrics(crop)
        if fe is None:
            res["fields"][fld["name"]] = {"status": "REVIEW", "reason": "no_ink_pixels"}
            continue
        res["fields"][fld["name"]] = {"status": band(fe["weber"]), "weber": fe["weber"],
                                      "contrast": fe["contrast"], "dark_frac": fe["dark_frac"]}
    return res


def main():
    tpl = json.load(open(TPL, encoding="utf-8"))
    files = sorted(glob.glob(os.path.join(C.ROOT, "**", "*.pdf"), recursive=True))
    rows, tally = [], {}
    for f in files:
        r = process(os.path.relpath(f, C.ROOT), tpl)
        if r is None or "note" in r:
            continue
        rows.append(r)
        for n, i in r["fields"].items():
            tally[i["status"]] = tally.get(i["status"], 0) + 1
    json.dump(rows, open(os.path.join(OUT, "detect3.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("=" * 60)
    print("error3 status tally over required signed cells:")
    for k in sorted(tally):
        print(f"   {k:12} {tally[k]}")
    for label in ("TOO_FAINT", "REVIEW"):
        hits = [(r["rel"], n, i) for r in rows for n, i in r["fields"].items()
                if i["status"] == label and "anchor" not in i.get("reason", "")]
        print(f"\n{label} ({len(hits)}):")
        for rel, n, i in sorted(hits, key=lambda x: x[2].get("weber", 9)):
            print(f"   weber={i.get('weber')} contrast={i.get('contrast')} dark_frac={i.get('dark_frac')}"
                  f"  {n}/{os.path.basename(rel)[:34]}")


if __name__ == "__main__":
    main()
