"""Measure error4 features (pd_score, cc_h_cv) on an arbitrary PDF outside the
dataset cache. OCRs the file once with the same PaddleOCR config as ocr_cache,
locates each required signature field, and reports the typed/handwriting routing.

Usage:  python test_error4_file.py "<path-to.pdf>"
"""
import sys, os
import common as C
from ocr_cache import render, ocr_page
from fields_config import FIELDS, DEFAULT_OFFSETS, locate_field, get_fields
from detect4 import sig_crop, offsets, SIG_INK, PD_TYPED, NA_RE
from error4_mask2 import ink_mask_v2, feats

TPL_PATH = r"C:\Users\25775\Desktop\OCR_research\templates.json"
import json
TPL = json.load(open(TPL_PATH, encoding="utf-8"))


def analyze(pdf_path, ocr):
    img = render(pdf_path)
    items = ocr_page(ocr, img)
    form = C.detect_form(items)
    print(f"\n### {os.path.basename(pdf_path)}   form={form}")
    if form not in FIELDS:
        print("   (no field config for this form)"); return
    for fld in get_fields(form, required_only=True):
        sig, dat = locate_field(items, fld)
        if sig is None:
            print(f"   {fld['name']:14} anchor_not_found"); continue
        abox = C.sig_box_clip_date(sig["box"], dat["box"] if dat else None,
                                   offsets(TPL, form, fld["name"], "sig"))
        ink = C.ink_stats(img[abox[1]:abox[3], abox[0]:abox[2]])["ink_ratio"]
        if ink < SIG_INK:
            print(f"   {fld['name']:14} EMPTY (ink={ink:.4f})"); continue
        crop, pdtext, pdscore = sig_crop(img, items, abox)
        fe = feats(crop, ink_mask_v2) or {"cc_h_cv": 1.0, "band_conc": 0.0}
        c1 = pdscore >= PD_TYPED                 # routing = OCR confidence only
        status = "REVIEW->VLM" if c1 else "SIGNED"
        print(f"   {fld['name']:14} {status:11} pd_score={pdscore:.3f} pd='{pdtext}' "
              f"cc_h_cv={fe['cc_h_cv']:.3f} (recorded)  trigger=[{'conf' if c1 else 'none'}]")


def main():
    paths = sys.argv[1:] or [
        r"C:\Users\25775\Desktop\OCR_research\100394_Form ETA-9089 Page 2_NG (not signed)_2.pdf"]
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=True, lang="en")
    print(f"routing threshold: PD_TYPED={PD_TYPED} (OCR confidence only)")
    for p in paths:
        analyze(p, ocr)


if __name__ == "__main__":
    main()
