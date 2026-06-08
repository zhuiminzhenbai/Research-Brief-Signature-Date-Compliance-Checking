"""error4 Layer-2: local typed-vs-handwritten via TWO independent signals.

Signal 1 (font_score): how well the crop ink matches a standard-font rendering of
  the OCR'd string (mean-subtracted spatial correlation, max over fonts). Typed ink
  matches some font; handwriting matches none.
Signal 2 (tess_combined): does a SECOND, print-specialised OCR (Tesseract) read the
  same string AND agree with PaddleOCR?  combined = (tess_conf/100) * str_agreement.

AND-rule with short-circuit gating: a typed verdict needs BOTH signals. Tesseract
runs ONLY when font_score has already cleared TF -- it cannot flip a font-negative
verdict, so computing it there would be wasted. The two signals fail on DIFFERENT
samples (neat cursive scores high on font but low on OCR-agreement, and vice-versa),
so requiring both is much harder to fool than either alone.

Validated on scanned_sign_pages: 0 FP / 57 handwritten, 2/2 typed at TF=0.34, TC=0.40.
CAVEAT: only 2 typed positives exist, so the thresholds are a calibrated estimate; the
DISAGREE zone (font-high, OCR-unconfirmed) is meant for the VLM tier, not a hard verdict.
"""
import os, re, difflib
import numpy as np
import cv2

TF = 0.34          # font_score gate (typed candidate)
TC = 0.40          # tess_combined confirmation
H = 48             # normalized glyph height
SHIFT = 6          # 2D alignment slack
FONTDIR = r"C:\Windows\Fonts"
FONTS = ["arial.ttf", "times.ttf", "cour.ttf", "calibri.ttf", "verdana.ttf",
         "tahoma.ttf", "georgia.ttf", "ariblk.ttf", "segoeui.ttf", "consola.ttf"]

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False
try:
    import pytesseract
    from pytesseract import Output
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    _HAVE_TESS = True
except Exception:
    _HAVE_TESS = False

_ALPHA = re.compile(r"[A-Za-z]")
_FONT_CACHE = {}


def _font(path):
    if path not in _FONT_CACHE:
        try:
            _FONT_CACHE[path] = ImageFont.truetype(path, 64)
        except Exception:
            _FONT_CACHE[path] = None
    return _FONT_CACHE[path]


def _ink_mask(gray):
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 15)
    Hc, Wc = bw.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(Wc * 0.7)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(Hc * 0.7))))
    lines = cv2.dilate(cv2.erode(bw, hk), hk) | cv2.dilate(cv2.erode(bw, vk), vk)
    return cv2.subtract(bw, lines)


def _normalize(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return None
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    return cv2.resize(m, (max(1, int(round(w * H / h))), H), interpolation=cv2.INTER_AREA)


def _render(text, font):
    img = Image.new("L", (1400, 140), 255)
    ImageDraw.Draw(img).text((10, 20), text, fill=0, font=font)
    a = np.array(img)
    return _normalize((a < 128).astype(np.uint8) * 255)


def font_score(crop, text):
    """Max over standard fonts of TM_CCOEFF_NORMED between crop ink and the font
    rendering of the SAME string. Typed -> high; handwriting -> low."""
    if not _HAVE_PIL or crop is None or crop.size == 0 or not text.strip():
        return 0.0
    cg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    cm = _normalize(_ink_mask(cg))
    if cm is None:
        return 0.0
    cf = cv2.GaussianBlur(cm.astype(np.float32), (3, 3), 0)
    padded = cv2.copyMakeBorder(cf, SHIFT, SHIFT, SHIFT, SHIFT, cv2.BORDER_CONSTANT, value=0)
    best = 0.0
    for fn in FONTS:
        f = _font(os.path.join(FONTDIR, fn))
        if f is None:
            continue
        tm = _render(text, f)
        if tm is None:
            continue
        tf = cv2.GaussianBlur(cv2.resize(tm, (cm.shape[1], H)).astype(np.float32), (3, 3), 0)
        sc = float(cv2.matchTemplate(padded, tf, cv2.TM_CCOEFF_NORMED).max())
        if sc > best:
            best = sc
    return best


def tess_combined(crop, paddle_text):
    """(tess_conf/100) * string-agreement(paddle, tesseract). High only when a second,
    print-specialised OCR reads the SAME string confidently."""
    if not _HAVE_TESS or crop is None or crop.size == 0 or not paddle_text.strip():
        return 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    g = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    try:
        d = pytesseract.image_to_data(g, config="--psm 7 --oem 1", lang="eng",
                                      output_type=Output.DICT)
    except Exception:
        return 0.0
    bc, bt = -1.0, ""
    for txt, conf in zip(d["text"], d["conf"]):
        c = float(conf)
        if txt.strip() and _ALPHA.search(txt) and c > bc:
            bc, bt = c, txt.strip()
    if bc < 0:
        return 0.0
    agree = difflib.SequenceMatcher(None, paddle_text.lower().strip(), bt.lower().strip()).ratio()
    return (bc / 100.0) * agree


def layer2(crop, ocr_text):
    """Return (verdict, info) with verdict in {TYPED, SIGNED, DISAGREE}.
    Short-circuits Tesseract when font_score < TF (cannot change the AND outcome)."""
    fs = font_score(crop, ocr_text)
    info = {"font_score": round(fs, 3)}
    if fs < TF:
        return "SIGNED", info                       # not a standard font -> handwriting
    tc = tess_combined(crop, ocr_text)
    info["tess_combined"] = round(tc, 3)
    if tc >= TC:
        return "TYPED", info                         # both signals agree -> typed
    return "DISAGREE", info                           # font-high, OCR-unconfirmed -> VLM tier
