"""Evidence images for the doubted PASTED / TYPED flags, so a human can compare
the ORIGINAL page against what triggered the detector.

PASTED : render the page, box the flagged image-object region, and place a 6x zoom
         of that region + the embedded object's structural facts (xref, size in pt,
         dpi vs page dpi, coverage) beside it.
TYPED  : crop the signature field, zoom it, label the OCR confidence that triggered
         TYPED. Lets you see whether it's actually typed or just legible handwriting.

Output -> testf_evidence/{pasted,typed}/   (gitignored, local only)
"""
import os, json, hashlib
import numpy as np
import cv2
import fitz
import common as C
from fields_config import locate_field, get_fields, FIELDS
from visualize_results import offsets

ROOT = r"C:\Users\25775\Desktop\OCR_research\test_f\test_f"
OCR_DIR = r"C:\Users\25775\Desktop\OCR_research\testf_out\ocr"
OUT = r"C:\Users\25775\Desktop\OCR_research\testf_evidence"
TPL = json.load(open(r"C:\Users\25775\Desktop\OCR_research\templates.json", encoding="utf-8"))
for s in ("pasted", "typed"):
    os.makedirs(os.path.join(OUT, s), exist_ok=True)


def load_items(rel):
    cp = os.path.join(OCR_DIR, hashlib.md5(rel.encode("utf-8")).hexdigest()[:10] + ".json")
    return json.load(open(cp, encoding="utf-8")) if os.path.exists(cp) else None


def label_bar(text, w, h=34):
    bar = np.full((h, w, 3), 40, np.uint8)
    cv2.putText(bar, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def pasted_evidence(rel, hits):
    pdf = os.path.join(ROOT, rel)
    pg = fitz.open(pdf)[0]; R = pg.rect
    scale = 2200 / max(R.width, R.height)
    page_dpi = None
    infos = {im["xref"]: im for im in pg.get_image_info(xrefs=True)}
    # estimate page dpi from the largest image
    big = max(infos.values(), key=lambda im: (im["bbox"][2]-im["bbox"][0])*(im["bbox"][3]-im["bbox"][1]), default=None)
    if big:
        page_dpi = round(big["width"] / max((big["bbox"][2]-big["bbox"][0])/72.0, 1e-3))
    img = C.render(pdf)
    H, W = img.shape[:2]
    for h in hits:
        im = infos.get(h["xref"])
        if not im:
            continue
        b = [int(v*scale) for v in im["bbox"]]
        page = img.copy()
        cv2.rectangle(page, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 4)
        # zoom region (pad)
        pad = 60
        x0, y0 = max(0, b[0]-pad), max(0, b[1]-pad)
        x1, y1 = min(W, b[2]+pad), min(H, b[3]+pad)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            crop = np.full((80, 200, 3), 220, np.uint8)
        zoom = cv2.resize(crop, (min(700, crop.shape[1]*6), min(700, crop.shape[0]*6)),
                          interpolation=cv2.INTER_NEAREST)
        pageR = cv2.resize(page, (700, int(H*700/W)))
        # stack zoom under a fact bar
        wpt = im["bbox"][2]-im["bbox"][0]; hpt = im["bbox"][3]-im["bbox"][1]
        dpi = round(im["width"]/max(wpt/72.0, 1e-3))
        facts = f"xref{h['xref']} {wpt:.0f}x{hpt:.0f}pt dpi={dpi} vs page~{page_dpi} cover={h['cover']} field={h['field']}"
        zb = np.vstack([label_bar(facts, zoom.shape[1]), zoom])
        # pad to same height then hconcat
        Hh = max(pageR.shape[0], zb.shape[0])
        def padh(a):
            if a.shape[0] < Hh:
                a = np.vstack([a, np.full((Hh-a.shape[0], a.shape[1], 3), 30, np.uint8)])
            return a
        combo = np.hstack([padh(pageR), np.full((Hh, 12, 3), 80, np.uint8), padh(zb)])
        name = f"{rel.replace(os.sep,'_').replace('.pdf','')}__xref{h['xref']}.png"
        cv2.imwrite(os.path.join(OUT, "pasted", name), combo)
        print(f"  [pasted] {name}  ({facts})")


def typed_evidence(rel, field_name, conf):
    pdf = os.path.join(ROOT, rel)
    items = load_items(rel)
    if items is None:
        print(f"  [typed] no cache for {rel}"); return
    form = C.detect_form(items)
    img = C.render(pdf)
    for fld in get_fields(form, required_only=True):
        if fld["name"] != field_name:
            continue
        sig, dat = locate_field(items, fld)
        if sig is None:
            print(f"  [typed] {rel} {field_name} anchor lost"); return
        box = C.sig_box_clip_date(sig["box"], dat["box"] if dat else None,
                                  offsets(TPL, form, fld["name"], "sig"))
        crop = img[box[1]:box[3], box[0]:box[2]]
        if crop.size == 0:
            return
        z = cv2.resize(crop, (min(900, crop.shape[1]*3), min(300, crop.shape[0]*3)),
                       interpolation=cv2.INTER_CUBIC)
        out = np.vstack([label_bar(f"{field_name}  judged TYPED because ocr_conf={conf} (>=0.99)  -- typed or just legible handwriting?", z.shape[1]), z])
        name = f"{rel.replace(os.sep,'_').replace('.pdf','')}__{field_name}.png"
        cv2.imwrite(os.path.join(OUT, "typed", name), out)
        print(f"  [typed] {name}  conf={conf}")
        return


def main():
    rows = json.load(open(r"C:\Users\25775\Desktop\OCR_research\testf_out\review.json", encoding="utf-8"))
    print("=== PASTED evidence ===")
    for r in rows:
        if r.get("e5") and r["e5"]["verdict"] == "PASTED_IMAGE_SIGNATURE":
            pasted_evidence(r["file"], r["e5"]["hits"])
    print("=== TYPED evidence ===")
    for r in rows:
        for fn, i in r.get("fields", {}).items():
            if i["status"] == "TYPED_SIGNATURE":
                conf = i.get("note", "").replace("ocr_conf=", "")
                typed_evidence(r["file"], fn, conf)
    print(f"\n-> {OUT}\\pasted  and  {OUT}\\typed")


if __name__ == "__main__":
    main()
