"""Full error1-5 detection pass over the unlabeled test_f set, with annotated-image
export of every page that gets flagged, for human review.

Pipeline per PDF (PAGE 1 only — mirrors the detectors' doc[0] behavior):
  OCR (cached, resumable) -> detect_form -> if known form:
    e1/e2  (missing / wrong-box)   via visualize_results.verdict + page strays
    e3     (too-faint)             folded into verdict's Weber band
    e4     (typed)                 via verdict4 (Layer-1 native + OCR-conf routing)
    e5     (pasted/doc-not-scan)   structural image object + coverage pre-check

Outputs:
  testf_out/ocr/<md5>.json   per-file OCR cache (so re-runs are fast / resumable)
  testf_out/review.json      every file's per-field verdicts + e5
  testf_review/error/*.png   pages with a HARD error (missing/wrong/faint/typed/pasted/not-scan)
  testf_review/review/*.png  pages only needing REVIEW (anchor/faint-borderline/e4 mid-band)
All outputs are gitignored (contain real PII renders) — local review only.
"""
import os, json, hashlib, time, glob
from collections import Counter
import fitz
import cv2
import common as C
from ocr_cache import ocr_page
from fields_config import locate_field, get_fields, FIELDS
from error2_stray import compute_strays
from visualize_results import verdict, verdict4, draw, offsets
from doc_precheck import precheck
from detect5 import detect_paste

ROOT = r"C:\Users\25775\Desktop\OCR_research\test_f\test_f"
OUT = r"C:\Users\25775\Desktop\OCR_research\testf_out"
OCR_DIR = os.path.join(OUT, "ocr")
REV = r"C:\Users\25775\Desktop\OCR_research\testf_review"
TPL_PATH = r"C:\Users\25775\Desktop\OCR_research\templates.json"
for d in (OUT, OCR_DIR, os.path.join(REV, "error"), os.path.join(REV, "review")):
    os.makedirs(d, exist_ok=True)

HARD = {"MISSING", "WRONG_BOX", "TOO_FAINT", "TYPED_SIGNATURE",
        "PASTED_IMAGE_SIGNATURE", "DOC_NOT_SCAN"}


def all_pdfs(root):
    out = []
    for r, _, files in os.walk(root):
        for n in files:
            if n.lower().endswith(".pdf"):
                out.append(os.path.join(r, n))
    return sorted(out)


def cached_ocr(ocr, pdf_path, rel):
    cp = os.path.join(OCR_DIR, hashlib.md5(rel.encode("utf-8")).hexdigest()[:10] + ".json")
    if os.path.exists(cp):
        return json.load(open(cp, encoding="utf-8")), C.render(pdf_path)
    img = C.render(pdf_path)
    items = ocr_page(ocr, img)
    json.dump(items, open(cp, "w", encoding="utf-8"), ensure_ascii=False)
    return items, img


def boxes_pt(pdf_path, items, form, tpl):
    R = fitz.open(pdf_path)[0].rect
    scale = 2200 / max(R.width, R.height)
    bx = {}
    for fld in get_fields(form, required_only=True):
        sig, dat = locate_field(items, fld)
        if sig is None:
            continue
        abox = C.sig_box_clip_date(sig["box"], dat["box"] if dat else None,
                                   offsets(tpl, form, fld["name"], "sig"))
        bx[fld["name"]] = [v / scale for v in abox]
    return bx, scale


def field_verdict(img, items, form, fld, tpl, strays, pdf_path):
    """Merge e1/2/3 (verdict) with e4 (verdict4) for one field."""
    box, s, note = verdict(img, items, form, fld, tpl, strays)
    if box is None:
        return None, s, note
    if s in ("MISSING", "WRONG_BOX", "TOO_FAINT"):
        return box, s, note
    b4, s4, n4 = verdict4(img, items, form, fld, tpl, pdf_path)
    if s4 == "TYPED_SIGNATURE":
        return box, "TYPED_SIGNATURE", n4
    if s == "REVIEW":
        return box, "REVIEW", note
    if s4 == "REVIEW":
        return box, "REVIEW", n4
    return box, ("SIGNED" if s == "PASS" else s), note


def process(ocr, pdf_path, tpl):
    rel = os.path.relpath(pdf_path, ROOT)
    doc = fitz.open(pdf_path); npg = doc.page_count; doc.close()
    items, img = cached_ocr(ocr, pdf_path, rel)
    form = C.detect_form(items)
    rec = {"file": rel, "pages": npg, "form": form, "fields": {}, "e5": None}
    if form not in FIELDS:
        rec["note"] = "no_config"
        return rec, img, []

    strays = compute_strays(img, items, form, tpl)["stray"]
    draws = []  # (box, status, note, name)
    for fld in get_fields(form, required_only=True):
        box, st, note = field_verdict(img, items, form, fld, tpl, strays, pdf_path)
        rec["fields"][fld["name"]] = {"status": st, "note": note}
        if box is not None:
            draws.append((box, st, note, fld["name"]))

    # e5 structural (page-level)
    try:
        bx, scale = boxes_pt(pdf_path, items, form, tpl)
        hits = detect_paste(pdf_path, bx) if bx else []
        pc = precheck(pdf_path)
        if hits:
            rec["e5"] = {"verdict": "PASTED_IMAGE_SIGNATURE", "hits": hits}
            pg = fitz.open(pdf_path)[0]
            for h in hits:
                im = next((i for i in pg.get_image_info(xrefs=True) if i["xref"] == h["xref"]), None)
                if im:
                    b = [int(v * scale) for v in im["bbox"]]
                    draws.append((b, "PASTED_IMAGE_SIGNATURE", f"dpi={h['dpi']}", h["field"]))
        elif pc["band"] == "FAIL":
            rec["e5"] = {"verdict": "DOC_NOT_SCAN", "why": pc["reasons"]}
            H, W = img.shape[:2]
            draws.append(([6, 6, W - 6, H - 6], "DOC_NOT_SCAN",
                          f"cover {pc['cover']:.0%}", "document"))
    except Exception as e:
        rec["e5_err"] = str(e)[:60]
    return rec, img, draws


def save(img, sub, prefix, base):
    h, w = img.shape[:2]
    out = cv2.resize(img, (1100, int(h * 1100 / w)))
    path = os.path.join(REV, sub, f"{prefix}__{base[:48]}.png")
    cv2.imwrite(path, out)


def main():
    tpl = json.load(open(TPL_PATH, encoding="utf-8"))
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=True, lang="en")
    pdfs = all_pdfs(ROOT)
    print(f"reviewing {len(pdfs)} PDFs (page-1, error1-5)...", flush=True)

    rows, t0 = [], time.time()
    tally = Counter()
    for i, p in enumerate(pdfs, 1):
        try:
            rec, img, draws = process(ocr, p, tpl)
        except Exception as e:
            rows.append({"file": os.path.relpath(p, ROOT), "form": "ERR", "err": str(e)[:80]})
            tally["ERR"] += 1
            if i % 25 == 0:
                print(f"  {i}/{len(pdfs)} ({time.time()-t0:.0f}s)", flush=True)
            continue
        rows.append(rec)

        statuses = {d[1] for d in draws}
        if rec.get("e5"):
            statuses.add(rec["e5"]["verdict"])
        hard = statuses & HARD
        has_review = "REVIEW" in statuses
        for s in statuses:
            tally[s] += 1

        if draws and (hard or has_review):
            for box, st, note, name in draws:
                draw(img, box, st, note, name)
            base = rec["file"].replace(os.sep, "_").replace(".pdf", "")
            if hard:
                prefix = "+".join(sorted(hard))[:30]
                save(img, "error", prefix, base)
            else:
                save(img, "review", "REVIEW", base)

        if i % 25 == 0:
            print(f"  {i}/{len(pdfs)} ({time.time()-t0:.0f}s)  flagged so far: "
                  f"err={sum(tally[k] for k in HARD)}", flush=True)

    json.dump(rows, open(os.path.join(OUT, "review.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print(f"DONE {len(rows)} files in {time.time()-t0:.0f}s")
    print(f"OCR cache -> {OCR_DIR}   verdicts -> {OUT}\\review.json")
    print(f"flagged images -> {REV}\\error  and  {REV}\\review")
    print("\n=== verdict tally (file-level occurrences) ===")
    for k, v in tally.most_common():
        mark = " [ERROR]" if k in HARD else (" [review]" if k == "REVIEW" else "")
        print(f"  {k:24} {v}{mark}")
    n_err_files = sum(1 for r in rows if (r.get("e5") and r["e5"]["verdict"] in HARD)
                      or any(f["status"] in HARD for f in r.get("fields", {}).values()))
    nocfg = sum(1 for r in rows if r.get("note") == "no_config")
    print(f"\nfiles with >=1 HARD error: {n_err_files}")
    print(f"files skipped (form not in the 4 configured): {nocfg}")


if __name__ == "__main__":
    main()
