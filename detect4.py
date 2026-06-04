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
import common as C
from fields_config import FIELDS, DEFAULT_OFFSETS, locate_field, get_fields
from error4_mask2 import ink_mask_v2, feats
import vlm_classify

OUT = r"C:\Users\25775\Desktop\OCR_research\detect4_out"
CROPS = os.path.join(OUT, "crops")
os.makedirs(CROPS, exist_ok=True)
TPL = r"C:\Users\25775\Desktop\OCR_research\templates.json"

SIG_INK = 0.012            # cell considered signed (same as detect12)
PD_TYPED = 0.85            # PaddleOCR read cell content at >= this conf -> typed-leaning
CCHCV_TYPED = 0.55         # glyph-height CV (recorded only; no longer routes - see below)
# Routing = OCR confidence alone. If PaddleOCR did NOT read confident content
# (pd_score < PD_TYPED) the cell is clear handwriting -> auto-pass, no VLM call.
# Everything legible (typed OR neat handwriting reads confidently) goes to the VLM.
# (The glyph-height-uniformity signal was dropped: a real typed sample was MORE
# varied than genuine handwriting, so it only caused false REVIEWs.)
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
    img = C.render(os.path.join(C.ROOT, rel))
    base = os.path.basename(rel)
    res = {"rel": rel, "form": form, "fields": {}}
    for fld in get_fields(form, required_only=True):
        sig, dat = locate_field(items, fld)
        if sig is None:
            res["fields"][fld["name"]] = {"status": "REVIEW", "reason": "anchor_not_found"}
            continue
        abox = C.sig_box_clip_date(sig["box"], dat["box"] if dat else None,
                                   offsets(tpl, form, fld["name"], "sig"))
        if C.ink_stats(img[abox[1]:abox[3], abox[0]:abox[2]])["ink_ratio"] < SIG_INK:
            continue  # empty -> error1's domain, not error4
        crop, pdtext, pdscore = sig_crop(img, items, abox)
        if NA_RE.match(pdtext):
            res["fields"][fld["name"]] = {"status": "NA", "reason": "typed N/A (not-applicable)",
                                          "pd": pdtext}
            continue
        fe = feats(crop, ink_mask_v2) or {"cc_h_cv": 1.0, "band_conc": 0.0}
        # cc_h_cv (stroke-height uniformity) was dropped as a routing signal: a real
        # standard-font typed sample measured cc_h_cv=0.75 -- MORE varied than much
        # genuine handwriting (0.16-0.49) -- so the uniformity rule never separated
        # typed from handwriting and only produced false REVIEWs. Standard-font typed
        # text is caught reliably by high OCR confidence alone. (cc_h_cv still recorded.)
        typed_evidence = pdscore >= PD_TYPED
        info = {"pd_score": round(pdscore, 3), "pd": pdtext,
                "cc_h_cv": fe["cc_h_cv"], "band_conc": fe["band_conc"]}
        if not typed_evidence:
            info["status"] = "SIGNED"; info["reason"] = "handwriting (no typed evidence)"
        elif use_vlm:
            cp = os.path.join(CROPS, f"{base[:24]}__{fld['name']}.png".replace(" ", "_"))
            cv2.imwrite(cp, crop)
            v = vlm_classify.classify(cp)
            info["vlm"] = v["label"]; info["vlm_backend"] = v["backend"]
            info["status"] = {"typed": "TYPED_SIGNATURE", "handwritten": "SIGNED"}.get(
                v["label"], "REVIEW")
            info["reason"] = f"vlm={v['label']}"
        else:
            info["status"] = "REVIEW"; info["reason"] = "typed-candidate (no vlm)"
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
