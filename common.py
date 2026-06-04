"""Shared pipeline: cache loading, render, geometry, ink stats, form config."""
import os, re, json, hashlib
import numpy as np
import cv2
import fitz

ROOT = r"C:\Users\25775\Desktop\OCR_research\scanned_sign_pages\scanned_sign_pages"
CACHE = r"C:\Users\25775\Desktop\OCR_research\ocr_cache"
TARGET_LONG = 2200

RE_DATE_LABEL = re.compile(r"date of signature|date signed|^date[:\s]|^date$", re.I)


# ---- geometry helpers ----
def h(b):  return b[3] - b[1]
def w(b):  return b[2] - b[0]
def cx(b): return (b[0] + b[2]) / 2
def cy(b): return (b[1] + b[3]) / 2
def x_overlap(a, b): return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))


def render(path, target_long=TARGET_LONG):
    doc = fitz.open(path); page = doc[0]; r = page.rect
    scale = target_long / max(r.width, r.height)
    pm = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.h, pm.w, pm.n)
    if pm.n == 4: img = img[:, :, :3]
    doc.close()
    return np.ascontiguousarray(img[:, :, ::-1])  # BGR


def cache_path(rel):
    key = hashlib.md5(rel.encode("utf-8")).hexdigest()[:10]
    return os.path.join(CACHE, key + ".json")


def load_cache(rel):
    cp = cache_path(rel)
    if not os.path.exists(cp):
        return None
    with open(cp, encoding="utf-8") as fp:
        return json.load(fp)


def detect_form(items):
    txt = " ".join(it["text"] for it in items).lower()
    page8 = "part 11" in txt or "additional information" in txt
    if "i-140" in txt or "i140" in txt:
        return "I-140 P8" if page8 else "I-140 P6"
    if "g-28" in txt or "g28" in txt:   return "G-28 P3"
    if "9089" in txt:                   return "ETA-9089"
    return "?"


# ---- required signature field config per form ----
# label_re: matches the signature field label; exclude_re: drop matches; dir: where ink lives.
FORM_CONFIG = {
    "I-140 P6": [
        {"name": "petitioner", "required": True,
         "label_re": re.compile(r"(Petitioner|Authorized Signatory).{0,45}Signature", re.I),
         "exclude_re": re.compile(r"interpreter", re.I), "dir": "below"},
    ],
    "I-140 P8": [
        {"name": "signatory", "required": True,
         "label_re": re.compile(r"^Signature\b|\bSignature\s*$", re.I),
         "exclude_re": None, "dir": "right"},
    ],
    "G-28 P3": [
        {"name": "attorney", "required": True,
         "label_re": re.compile(r"Signature of Attorney", re.I),
         "exclude_re": None, "dir": "below"},
        {"name": "client", "required": True,
         "label_re": re.compile(r"Signature of Client", re.I),
         "exclude_re": None, "dir": "below"},
    ],
    "ETA-9089": [
        {"name": "foreign_worker", "required": True,
         "label_re": re.compile(r"^1\.?\s*Signature", re.I),
         "exclude_re": None, "dir": "below"},
    ],
}


def find_label(items, cfg):
    for it in items:
        if cfg["label_re"].search(it["text"]):
            if cfg["exclude_re"] and cfg["exclude_re"].search(it["text"]):
                continue
            return it
    return None


def signature_interior(sig_label_box, items, img_shape, direction="below"):
    """Box where a handwritten signature would sit, relative to its label."""
    H, W = img_shape[:2]
    slb = sig_label_box
    lh = h(slb)
    date_labels = [it for it in items if RE_DATE_LABEL.search(it["text"])]

    if direction == "right":
        x0 = slb[2] + 0.3 * lh
        y0 = slb[1] - 0.3 * lh
        x1 = slb[2] + 14 * lh
        y1 = slb[3] + 0.5 * lh
        # bound by a date label to the right on same row
        for dl in date_labels:
            b = dl["box"]
            if b[0] > slb[2] and abs(cy(b) - cy(slb)) < 1.5 * lh:
                x1 = min(x1, b[0] - 0.3 * lh)
    else:  # below
        x0 = slb[0]
        x1 = min(W, slb[0] + 18 * lh)
        y0 = slb[3] + 0.2 * lh
        y1 = slb[3] + 4.0 * lh
        # clip to a date label that sits below (box bottom), but ignore too-close ones
        for dl in date_labels:
            b = dl["box"]
            dy = b[1] - slb[3]
            if 1.3 * lh < dy < 5 * lh and x_overlap(b, [x0, 0, x1, 0]) > 0.15 * (x1 - x0):
                y1 = min(y1, b[1] - 0.2 * lh)
    box = [max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))]
    return box, lh


def box_from_anchor(anchor, params):
    """Box from anchor top-left + offsets [a,b,c,d] in units of anchor height lh."""
    ax0, ay0, ax1, ay1 = anchor
    lh = ay1 - ay0
    a, b, c, d = params
    return [max(0, int(ax0 + a*lh)), max(0, int(ay0 + b*lh)),
            int(ax0 + c*lh), int(ay0 + d*lh)]


def sig_box_clip_date(sig_anchor, date_anchor, params):
    """Signature box; if the date label sits in a right-side cell (same row), clip the
    box's right edge to the cell divider (date anchor left). Scan-invariant: avoids
    amplifying anchor-height noise via a large horizontal multiplier."""
    box = box_from_anchor(sig_anchor, params)
    if date_anchor is not None:
        dax0 = date_anchor[0]
        if dax0 > sig_anchor[2] and box[2] > dax0:   # date is to the right, same row
            box[2] = int(dax0 - 0.2 * (sig_anchor[3] - sig_anchor[1]))
    return box


def date_box(label_anchor, date_dir):
    """Date-value search box derived from the date LABEL's edge (scan-stable):
    'right' -> just right of the label (value on same row);
    'below' -> just under the label (value in the cell below). Avoids the large
    left-origin multiplier that amplified anchor-height noise."""
    lx0, ly0, lx1, ly1 = label_anchor
    lh = ly1 - ly0
    if date_dir == "below":
        return [int(lx0), int(ly1 + 0.2*lh), int(lx0 + 7.0*lh), int(ly1 + 3.0*lh)]
    # default: right of the label
    return [int(lx1 + 0.2*lh), int(ly0 - 0.3*lh), int(lx1 + 9.0*lh), int(ly1 + 0.4*lh)]


def ink_stats(crop):
    if crop.size == 0:
        return {"ink_ratio": 0.0, "ink_px": 0, "n_cc": 0, "area": 0}
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 15)
    Hc, Wc = bw.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, Wc // 3), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, Hc // 3)))
    lines = cv2.dilate(cv2.erode(bw, hk), hk) | cv2.dilate(cv2.erode(bw, vk), vk)
    ink = cv2.subtract(bw, lines)
    ink = cv2.medianBlur(ink, 3)
    area = Hc * Wc
    ink_px = int((ink > 0).sum())
    n, _, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= max(8, area * 0.0002)]
    return {"ink_ratio": round(ink_px / area, 5) if area else 0.0,
            "ink_px": ink_px, "n_cc": len(big), "area": area}
