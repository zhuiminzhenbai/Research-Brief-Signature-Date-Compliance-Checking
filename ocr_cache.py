"""OCR every page once, cache text+score+box so all detectors can reuse it."""
import os, sys, glob, json, hashlib, re
import numpy as np
import fitz

ROOT = r"C:\Users\25775\Desktop\OCR_research\scanned_sign_pages\scanned_sign_pages"
CACHE = r"C:\Users\25775\Desktop\OCR_research\ocr_cache"
os.makedirs(CACHE, exist_ok=True)
TARGET_LONG = 2200


def render(path, target_long=TARGET_LONG):
    doc = fitz.open(path); page = doc[0]; r = page.rect
    scale = target_long / max(r.width, r.height)
    pm = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.h, pm.w, pm.n)
    if pm.n == 4: img = img[:, :, :3]
    doc.close()
    return np.ascontiguousarray(img[:, :, ::-1])  # BGR


def poly_to_box(p):
    p = np.array(p).reshape(-1, 2)
    return [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]


def ocr_page(ocr, img):
    r = ocr.predict(img)[0]
    boxes = r.get("rec_boxes", None)
    if boxes is None or len(boxes) == 0:
        boxes = [poly_to_box(p) for p in r["rec_polys"]]
    else:
        boxes = [[float(v) for v in b] for b in np.array(boxes).tolist()]
    return [{"text": t, "score": round(float(s), 4), "box": [round(v, 1) for v in b]}
            for t, s, b in zip(r["rec_texts"], r["rec_scores"], boxes)]


def rotate(img, ang):
    """Rotate a rendered image by 0/90/180/270 degrees (clockwise)."""
    import cv2
    if ang == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if ang == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if ang == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _sideways(items):
    """True if most multi-char text lines are taller than wide (page content rotated 90/270)."""
    multi = [it for it in items if len(it["text"]) >= 4]
    if not multi:
        return False
    vert = sum((it["box"][3]-it["box"][1]) > (it["box"][2]-it["box"][0]) for it in multi)
    return vert / len(multi) > 0.5


def _ori_score(items):
    multi = [it for it in items if len(it["text"]) >= 4]
    if not multi:
        return 0.0
    horiz = sum((it["box"][2]-it["box"][0]) >= (it["box"][3]-it["box"][1]) for it in multi)
    mean_s = sum(it["score"] for it in multi) / len(multi)
    return horiz / len(multi) + mean_s          # prefer horizontal text that also reads confidently


# Footer tokens common to all 3 form families; these always sit at the page BOTTOM
# ("Page N of M", the "Form <code>" edition line, and the long barcode digit string).
_FOOT_RE = re.compile(r"page\s*\d+\s*of\s*\d+|form\s+(?:g-28|i-140|eta-9089)|^\d{10,}", re.I)


def footer_at_top(items, top_frac=0.25):
    """True if every detected footer token sits in the top `top_frac` of the page.
    0deg and 180deg both produce horizontal text (so OCR confidence can't tell them
    apart); the footer's vertical position is the deterministic 180-flip signal."""
    if not items:
        return False
    H = max(it["box"][3] for it in items)
    foot = [(it["box"][1] + it["box"][3]) / 2 for it in items if _FOOT_RE.search(it.get("text", ""))]
    return bool(foot) and max(foot) < top_frac * H


def ocr_page_upright(ocr, img0):
    """OCR with orientation correction. OCRs as-rendered; if the page content is sideways
    (vertical text lines), re-OCRs at 90/270 and keeps the orientation that reads upright;
    then, if the footer is at the top (page is 180 off in the horizontal axis), re-OCRs at
    +180. Returns (items, angle) where angle is the clockwise rotation applied to img0.
    Detectors must render the page and apply the SAME rotation so boxes/pixels stay aligned."""
    items = ocr_page(ocr, img0)
    angle = 0
    if _sideways(items):
        best = (items, 0, _ori_score(items))
        for ang in (90, 270):
            it = ocr_page(ocr, rotate(img0, ang))
            sc = _ori_score(it)
            if sc > best[2]:
                best = (it, ang, sc)
        items, angle = best[0], best[1]
    if footer_at_top(items):                     # footer landed on top -> page is 180 flipped
        angle = (angle + 180) % 360
        items = ocr_page(ocr, rotate(img0, angle))
    return items, angle


def cache_path(rel):
    key = hashlib.md5(rel.encode("utf-8")).hexdigest()[:10]
    return os.path.join(CACHE, key + ".json")


def main():
    from paddleocr import PaddleOCR
    files = sorted(glob.glob(os.path.join(ROOT, "**", "*.pdf"), recursive=True))
    force = "--force" in sys.argv
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=True, lang="en")
    done = 0
    for f in files:
        rel = os.path.relpath(f, ROOT)
        cp = cache_path(rel)
        if os.path.exists(cp) and not force:
            print(f"skip (cached) {rel}"); continue
        img = render(f)
        items = ocr_page(ocr, img)
        with open(cp, "w", encoding="utf-8") as fp:
            json.dump({"rel": rel, "target_long": TARGET_LONG,
                       "img_wh": [img.shape[1], img.shape[0]], "items": items}, fp,
                      ensure_ascii=False)
        done += 1
        print(f"[{done}] {len(items):>3} items -> {rel}")
    print(f"\nCached {done} new / {len(files)} total -> {CACHE}")


if __name__ == "__main__":
    main()
