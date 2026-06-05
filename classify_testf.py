"""Content-based form classification over the unlabeled test_f set (read content,
NOT filenames). For each PDF: page count + detect_form on page 1. Uses the PDF text
layer when present (free), else OCRs page 1 with the same PaddleOCR config as
ocr_cache. Writes testf_out/classify.json and prints a summary + the '?' failures.

This deliberately classifies on PAGE 1 ONLY (mirroring the detectors' current
doc[0] behavior) so multi-page bundles surface as a limitation.
"""
import os, json, time
import fitz
import common as C
from ocr_cache import ocr_page

OUT = r"C:\Users\25775\Desktop\OCR_research\testf_out"
os.makedirs(OUT, exist_ok=True)
ROOT = r"C:\Users\25775\Desktop\OCR_research\test_f\test_f"


def all_pdfs(root):
    out = []
    for r, _, files in os.walk(root):
        for n in files:
            if n.lower().endswith(".pdf"):
                out.append(os.path.join(r, n))
    return sorted(out)


def main():
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=True, lang="en")
    pdfs = all_pdfs(ROOT)
    print(f"classifying {len(pdfs)} PDFs (page-1 content)...", flush=True)

    rows = []
    t0 = time.time()
    for i, p in enumerate(pdfs, 1):
        rec = {"file": os.path.relpath(p, ROOT)}
        try:
            doc = fitz.open(p)
            rec["pages"] = doc.page_count
            txt = doc[0].get_text()
            doc.close()
        except Exception as e:
            rec["pages"] = 0; rec["form"] = "ERR"; rec["src"] = str(e)[:40]
            rows.append(rec); continue
        if len(txt.strip()) >= 20:
            items = [{"text": txt, "score": 1.0, "box": [0, 0, 0, 0]}]
            rec["src"] = "textlayer"
        else:
            items = ocr_page(ocr, C.render(p))
            rec["src"] = "ocr"
        rec["form"] = C.detect_form(items)
        rows.append(rec)
        if i % 25 == 0:
            print(f"  {i}/{len(pdfs)}  ({time.time()-t0:.0f}s)", flush=True)

    json.dump(rows, open(os.path.join(OUT, "classify.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ---- summary ----
    from collections import Counter
    byform = Counter(r["form"] for r in rows)
    bysrc = Counter(r["src"] for r in rows if "src" in r)
    print("\n" + "=" * 60)
    print(f"DONE {len(rows)} files in {time.time()-t0:.0f}s -> {OUT}\\classify.json")
    print("\n=== form distribution (content-based) ===")
    for k, v in byform.most_common():
        print(f"  {k:12} {v}")
    print("\n=== source ===")
    for k, v in bysrc.most_common():
        print(f"  {k:12} {v}")
    fails = [r for r in rows if r["form"] in ("?", "ERR")]
    print(f"\n=== detect_form FAILURES ({len(fails)}) ===")
    for r in fails:
        print(f"  [{r['form']}] {r['pages']}p src={r.get('src')}  {r['file'][:60]}")
    multi = [r for r in rows if r["pages"] > 1]
    print(f"\n=== multi-page files ({len(multi)}) — page-1-only detects only the first form ===")
    for r in sorted(multi, key=lambda x: -x["pages"]):
        print(f"  {r['pages']}p  form(p1)={r['form']:10} {r['file'][:55]}")


if __name__ == "__main__":
    main()
