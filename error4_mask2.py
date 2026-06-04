"""Gentler line removal: only strip rows/cols that are ~full-span rules/borders
(projection coverage based), so typed letters' vertical stems survive. Re-measure
typed-vs-handwritten separation with TIGHT OCR boxes for both classes.
"""
import os, glob, json, re
import numpy as np
import cv2
import common as C
from fields_config import FIELDS, DEFAULT_OFFSETS, locate_field, get_fields

OUT = r"C:\Users\25775\Desktop\OCR_research\typed_out"
TPL = r"C:\Users\25775\Desktop\OCR_research\templates.json"
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


def is_name_like(text):
    t = text.strip()
    if any(ch.isdigit() for ch in t) or LABEL_WORDS.search(t):
        return False
    return 1 <= len(t.split()) <= 3 and 4 <= len(ALPHA.findall(t)) <= 22


def ink_mask_v2(crop, cov=0.78):
    """Binarize, then remove only full-span rules: a connected line component whose
    bbox spans >=cov of width (horizontal rule) or >=cov of height (vertical border).
    Letter stems (partial height) survive."""
    if crop.size == 0:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 15)
    H, W = bw.shape
    # candidate line pixels via thin long kernels
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(0.5*W)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(0.5*H))))
    hor = cv2.dilate(cv2.erode(bw, hk), hk)
    ver = cv2.dilate(cv2.erode(bw, vk), vk)
    lines = np.zeros_like(bw)
    for comp in (hor, ver):
        n, _, stats, _ = cv2.connectedComponentsWithStats((comp > 0).astype(np.uint8), 8)
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if w >= cov*W or h >= cov*H:
                lines[y:y+h, x:x+w] |= comp[y:y+h, x:x+w]
    ink = cv2.subtract(bw, lines)
    return cv2.medianBlur(ink, 3)


def feats(crop, mask_fn):
    m = mask_fn(crop)
    if m is None or (m > 0).sum() < 40:
        return None
    n, _, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    hs = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 10]
    cc_h_cv = float(np.std(hs) / (np.mean(hs) + 1e-6)) if len(hs) >= 3 else 1.0
    H = m.shape[0]
    rowsum = m.sum(axis=1).astype(float)
    bandh = max(1, int(0.40 * H))
    band_conc = float(np.convolve(rowsum, np.ones(bandh), "valid").max() / (rowsum.sum() + 1e-6))
    return {"cc_h_cv": round(cc_h_cv, 3), "band_conc": round(band_conc, 3)}


def main():
    from error4_features import ink_mask as ink_mask_v1
    tpl = json.load(open(TPL, encoding="utf-8"))
    files = sorted(glob.glob(os.path.join(C.ROOT, "**", "*.pdf"), recursive=True))
    hand1, hand2, typ1, typ2 = [], [], [], []
    typed_mask_vis = []
    for f in files:
        rel = os.path.relpath(f, C.ROOT)
        cache = C.load_cache(rel)
        if cache is None:
            continue
        items = cache["items"]; form = C.detect_form(items)
        if form not in FIELDS:
            continue
        img = C.render(os.path.join(C.ROOT, rel))
        for fld in get_fields(form, required_only=True):
            sig, dat = locate_field(items, fld)
            if sig is None:
                continue
            abox = C.sig_box_clip_date(sig["box"], dat["box"] if dat else None,
                                       offsets(tpl, form, fld["name"], "sig"))
            if C.ink_stats(img[abox[1]:abox[3], abox[0]:abox[2]])["ink_ratio"] < 0.012:
                continue
            content = [it for it in items if inside(it["box"], abox)
                       and not LABEL_WORDS.search(it["text"]) and ALPHA.search(it["text"])]
            if not content:
                continue
            ob = [int(v) for v in max(content, key=lambda it: it.get("score", 0))["box"]]
            pad = 6
            crop = img[max(0, ob[1]-pad):ob[3]+pad, max(0, ob[0]-pad):ob[2]+pad]
            a, b = feats(crop, ink_mask_v1), feats(crop, ink_mask_v2)
            if a and b:
                hand1.append(a); hand2.append(b)
        for it in items:
            if is_name_like(it["text"]) and it.get("score", 0) >= 0.92 \
               and 36 <= (it["box"][3]-it["box"][1]) <= 130:
                bx = [int(v) for v in it["box"]]; pad = 6
                crop = img[max(0, bx[1]-pad):bx[3]+pad, max(0, bx[0]-pad):bx[2]+pad]
                a, b = feats(crop, ink_mask_v1), feats(crop, ink_mask_v2)
                if a and b:
                    typ1.append(a); typ2.append(b)
                    if len(typed_mask_vis) < 6:
                        typed_mask_vis.append((crop, ink_mask_v2(crop)))

    def sweep(typed, hand, label):
        print(f"\n[{label}]  typed n={len(typed)}  hand n={len(hand)}")
        for T in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            rec = np.mean([s["cc_h_cv"] <= T for s in typed])
            fp = np.mean([s["cc_h_cv"] <= T for s in hand])
            print(f"   cc_h_cv<=T={T:.2f}  recall(typed)={rec:.1%}  FP(hand)={fp:.1%}")

    sweep(typ1, hand1, "v1 mask (aggressive line removal), both tight OCR box")
    sweep(typ2, hand2, "v2 mask (full-span-only line removal), both tight OCR box")

    # show v2 masks on typed crops
    rows = []
    for crop, m in typed_mask_vis:
        mrgb = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        def fit(im, w=460, h=80):
            hh, ww = im.shape[:2]; sc = min((w-8)/max(1, ww), (h-8)/max(1, hh))
            return cv2.resize(im, (max(1, int(ww*sc)), max(1, int(hh*sc))))
        a, b = fit(crop), fit(mrgb)
        row = np.full((80, 920, 3), 255, np.uint8)
        row[4:4+a.shape[0], 4:4+a.shape[1]] = a
        row[4:4+b.shape[0], 464:464+b.shape[1]] = b
        rows.append(row)
    canvas = np.full((sum(r.shape[0] for r in rows)+30, 920, 3), 255, np.uint8)
    cv2.putText(canvas, "TYPED with v2 mask  [ original | mask ]", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y = 28
    for r in rows:
        canvas[y:y+r.shape[0]] = r; y += r.shape[0]
    p = os.path.join(OUT, "linecheck_typed_v2.png")
    cv2.imwrite(p, canvas); print(f"\n-> {p}")


if __name__ == "__main__":
    main()
