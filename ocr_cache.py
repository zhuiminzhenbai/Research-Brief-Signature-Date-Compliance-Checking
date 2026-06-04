"""OCR every page once, cache text+score+box so all detectors can reuse it."""
import os, sys, glob, json, hashlib
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
