# Signature & Date Compliance Detection for Scanned Immigration Forms

An OCR-driven pipeline that flags **six signature/date compliance errors** on scanned
US immigration form pages (I-140 Page 6, I-140 Page 8, G-28 Page 3, ETA-9089 Page 2).

> ⚠️ **Data / PII notice.** The system processes scanned immigration documents
> containing real signatures and personal data. The source PDFs, the OCR cache derived
> from them, and any live API keys are **excluded from this repository** (see
> `.gitignore`). Only code and test-result artifacts are committed. Keep this repository
> **private**.

---

## The six error types

| # | Code | Meaning |
|---|------|---------|
| 1 | `SIGNATURE_MISSING` | A required signature field is empty |
| 2 | `SIGNATURE_WRONG_BOX` | Signed in the wrong field / stray ink outside the box |
| 3 | `SIGNATURE_TOO_FAINT` | Signature ink contrast too low to be legible |
| 4 | `TYPED_SIGNATURE` | A typed name instead of a handwritten signature |
| 5 | `PASTED_IMAGE_SIGNATURE` | A signature image pasted/screenshotted into the form |
| 6 | `DATE_UNREADABLE` | Signature-date value missing or illegible |

(Full requirements: `research-brief-0601.md`.)

---

## Architecture

The pipeline is **decoupled from the OCR model**: only `ocr_cache.py` touches PaddleOCR.
Every page is OCR'd once into an md5-keyed JSON cache (`{text, score, box}`), and all
detectors consume that cache. Swapping the OCR engine means changing one file.

```
PDF ──▶ ocr_cache.py (PaddleOCR PP-OCRv5, render long-side 2200px)
            │   writes ocr_cache/<md5>.json  = {text, score, box:[x0,y0,x1,y1]}
            ▼
       common.py  ── shared geometry, ink stats, per-form field config
            │
            ├─ fields_config.py / templates.json   anchor-relative field templates
            │      (field boxes as offsets from an OCR-detected anchor label,
            │       in units of the anchor's height — scan-scale invariant)
            ▼
       detectors (one concern each) ──▶ detect*_out/*.json + annotated images
```

**Anchor-relative templates.** Rather than fixed pixel boxes (which break across scans),
each field is located relative to its OCR-detected label anchor, measured in units of the
anchor height. This survives differences in scan resolution and placement.

---

## File map

### Core
| File | Role |
|------|------|
| `ocr_cache.py` | **Only** module that runs OCR; builds the page cache |
| `common.py` | Shared render / geometry / ink-stat helpers + per-form field config |
| `fields_config.py`, `templates.json` | Anchor-relative field templates |
| `vlm_classify.py` | Pluggable VLM backend (`stub` / `openai` / `anthropic`) for holistic judgments |
| `ground_truth.json` | Field-level labels for the known positives |

### Detectors
| File | Errors |
|------|--------|
| `detect12.py`, `error2_stray.py` | 1 `SIGNATURE_MISSING`, 2 `SIGNATURE_WRONG_BOX` |
| `detect3.py`, `error3_faint.py`, `error3_synth.py` | 3 `SIGNATURE_TOO_FAINT` |
| `detect4.py`, `error4_mask2.py` | 4 `TYPED_SIGNATURE` (OCR pre-filter → VLM) |
| `detect5.py`, `doc_precheck.py` | 5 `PASTED_IMAGE_SIGNATURE` |

### Visual gallery — `results_gallery/` (local only)
`visualize_results.py` produces at-a-glance annotated pages (field box + verdict)
for five of the six error types. Because these are renders of **real immigration
form pages (PII)**, they are kept **local only** and are not committed — run the
script to regenerate them. (Error 6 DATE_UNREADABLE is not yet implemented.)

### Machine-readable results (committed — trimmed to the final artifacts)
| Path | What it shows |
|------|---------------|
| `detect12_out/detect12.json` | Errors 1 & 2 full run (per-field status + metrics) |
| `detect3_out/detect3.json` | Error 3 full run (Weber bands: 93 PASS / 7 REVIEW / 0 FAIL) |
| `detect4_out/detect4.json` | Error 4 full run (OCR pre-filter → VLM candidates) |
| `faint_out/synth_ladder_grid.png` | Error 3 synthetic dimming ladder (FAIL-side validation) |

---

## Status & key findings

| Error | Status | Notes |
|-------|--------|-------|
| 1 Missing | ✅ Done | 4/4 positives caught, 0 false positives on 95 signed; 6 REVIEW (anchor-not-found on degraded scans) |
| 2 Wrong box | ✅ Done | Page-level stray-ink detection |
| 3 Too faint | ✅ Final (pixel + synthetic) | Weber-contrast bands. **6-axis pixel study shows "faint-but-legible" is not pixel-separable** — extreme faintness only; legibility judgement needs a VLM |
| 4 Typed | 🟡 Pipeline ready | OCR-confidence pre-filter (≥0.85 → VLM) → VLM verdict. A real typed sample is caught by confidence alone; the stroke-uniformity signal was dropped after it failed to separate typed from handwriting. Final typed-vs-handwriting call **needs VLM** |
| 5 Pasted image | ✅ Done | Layer-1 structural (separate image object) + document coverage pre-check (flattened screenshots). **Validated on 2 real fraud samples + 74 originals, 0 false positives.** Pixel forensics on clean scans proven ineffective |
| 6 Date unreadable | ⬜ Not started | Has real samples |

**Cross-cutting insight.** Errors 3 and 4 are *style/legibility* judgements that pixel
thresholds cannot make reliably; they need a vision-language model. A cleanly flattened
paste (error 5, type B) is information-theoretically undetectable by pixel forensics, so
the defense shifts to **document-level authenticity** (image coverage of the page).

---

## Running it

Paths are configured at the top of `ocr_cache.py` and `common.py` (`ROOT`, `CACHE`).
Point `ROOT` at the folder of source PDFs, then:

```bash
# 1. Build the OCR cache once (the only step that needs PaddleOCR)
python ocr_cache.py            # add --force to re-OCR

# 2. Run detectors (each reads the cache, writes detect*_out/)
python detect12.py             # errors 1 & 2
python detect3.py              # error 3
python detect4.py              # error 4 (VLM backend via vlm_classify.py)
python detect5.py              # error 5
python doc_precheck.py         # document-level authenticity pre-check
```

### Dependencies
`paddleocr` (PP-OCRv5), `opencv-python`, `numpy`, `PyMuPDF` (`fitz`). The VLM backend in
`vlm_classify.py` is optional and pluggable (defaults to a `stub` that returns
`uncertain`); set a backend + API key to enable error-4 verdicts.

> The OCR step runs on CPU. Do **not** run heavy local vision models on a low-memory
> machine.
