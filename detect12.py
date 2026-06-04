"""error1 (SIGNATURE_MISSING) + error2 (SIGNATURE_WRONG_BOX) detector + full eval.

Per required field: locate sig box (anchor-relative template), measure ink.
  - signed          -> SIGNED
  - empty + a page-level stray signature exists (handwriting-like ink outside
    every required sig/date box) -> WRONG_BOX
  - empty otherwise -> MISSING
The WRONG_BOX arbiter is page-level stray detection (error2_stray.compute_strays),
so no per-monitor box calibration is needed. Compares to ground_truth.json
(default: every required field SIGNED).
"""
import os, glob, json, re
import cv2
import common as C
from fields_config import FIELDS, DEFAULT_OFFSETS, locate_field, get_fields
from error2_stray import compute_strays

OUT = r"C:\Users\25775\Desktop\OCR_research\detect12_out"
os.makedirs(OUT, exist_ok=True)
TPL = r"C:\Users\25775\Desktop\OCR_research\templates.json"
GT = r"C:\Users\25775\Desktop\OCR_research\ground_truth.json"

# signature decision
SIG_INK = 0.012     # ink_ratio >= this OR
SIG_CC = 4          # n_cc >= this  => signed


def offsets(tpl, form, field, kind):
    return tpl.get(form, {}).get(field, {}).get(kind) or DEFAULT_OFFSETS[form][kind]


def sig_ink(img, items, form, fld, tpl):
    sig, dat = locate_field(items, fld)
    if sig is None:
        return None
    box = C.sig_box_clip_date(sig["box"], dat["box"] if dat else None,
                              offsets(tpl, form, fld["name"], "sig"))
    crop = img[box[1]:box[3], box[0]:box[2]]
    return C.ink_stats(crop)


def is_signed(st):
    return st is not None and (st["ink_ratio"] >= SIG_INK or st["n_cc"] >= SIG_CC)


def process(rel, tpl):
    cache = C.load_cache(rel)
    if cache is None:
        return None
    items = cache["items"]
    form = C.detect_form(items)
    if form not in FIELDS:
        return {"rel": rel, "form": form, "fields": {}, "note": "no_config"}
    img = C.render(os.path.join(C.ROOT, rel))

    # page-level stray-signature clusters arbitrate WRONG_BOX vs MISSING
    strays = compute_strays(img, items, form, tpl)["stray"]

    res = {"rel": rel, "form": form, "fields": {}, "n_stray": len(strays)}
    for fld in get_fields(form, required_only=True):
        name = fld["name"]
        st = sig_ink(img, items, form, fld, tpl)
        if st is None:
            res["fields"][name] = {"status": "REVIEW", "reason": "anchor_not_found"}
            continue
        if is_signed(st):
            status, reason = "SIGNED", "ink_present"
        elif strays:
            box = strays[0]["box"]
            status, reason = "WRONG_BOX", f"stray_sig@{box[:2]}"
        else:
            status, reason = "MISSING", "empty"
        res["fields"][name] = {"status": status, "reason": reason,
                               "ink": st["ink_ratio"], "cc": st["n_cc"]}
    return res


def main():
    tpl = json.load(open(TPL, encoding="utf-8")) if os.path.exists(TPL) else {}
    gt = json.load(open(GT, encoding="utf-8"))
    files = sorted(glob.glob(os.path.join(C.ROOT, "**", "*.pdf"), recursive=True))

    rows = []
    # confusion: per-field expected vs predicted
    conf = {"TP_miss": [], "TP_wrong": [], "FN": [], "FP": [], "REVIEW": [], "OK_neg": 0}
    for f in files:
        rel = os.path.relpath(f, C.ROOT)
        r = process(rel, tpl)
        if r is None or "note" in r:
            continue
        rows.append(r)
        base = os.path.basename(rel)
        gt_fields = {k: v for k, v in gt.get(base, {}).items() if not k.startswith("_")}
        for name, info in r["fields"].items():
            pred = info["status"]
            exp = gt_fields.get(name, "SIGNED")
            tag = f"{base} :: {name}  exp={exp} pred={pred} (ink={info.get('ink')},cc={info.get('cc')})"
            if pred == "REVIEW":
                conf["REVIEW"].append(tag)
            elif exp == "SIGNED" and pred == "SIGNED":
                conf["OK_neg"] += 1
            elif exp == "MISSING" and pred == "MISSING":
                conf["TP_miss"].append(tag)
            elif exp == "WRONG_BOX" and pred == "WRONG_BOX":
                conf["TP_wrong"].append(tag)
            elif exp != "SIGNED" and pred == "SIGNED":
                conf["FN"].append(tag)
            elif exp == "SIGNED" and pred in ("MISSING", "WRONG_BOX"):
                conf["FP"].append(tag)
            else:  # mismatch between MISSING/WRONG_BOX
                conf["FN"].append(tag + "  [type-mismatch]")

    json.dump(rows, open(os.path.join(OUT, "detect12.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("=" * 70)
    print("POSITIVES (expected MISSING / WRONG_BOX):")
    for k in ("TP_miss", "TP_wrong", "FN"):
        for t in conf[k]:
            mark = {"TP_miss": "[OK miss]", "TP_wrong": "[OK wrong]", "FN": "[MISS!! ]"}[k]
            print(f"  {mark} {t}")
    print(f"\nFALSE POSITIVES on SIGNED fields ({len(conf['FP'])}):")
    for t in conf["FP"]:
        print(f"  [FP] {t}")
    print(f"\nREVIEW (anchor/loc uncertain) ({len(conf['REVIEW'])}):")
    for t in conf["REVIEW"]:
        print(f"  {t}")
    npos = len(conf["TP_miss"]) + len(conf["TP_wrong"]) + len(conf["FN"])
    print("\n" + "=" * 70)
    print(f"Positives caught: miss {len(conf['TP_miss'])}, wrong {len(conf['TP_wrong'])}, "
          f"missed {len(conf['FN'])} / total positive-fields {npos}")
    print(f"True-negative SIGNED fields correct: {conf['OK_neg']}   "
          f"False positives: {len(conf['FP'])}   Reviews: {len(conf['REVIEW'])}")


if __name__ == "__main__":
    main()
