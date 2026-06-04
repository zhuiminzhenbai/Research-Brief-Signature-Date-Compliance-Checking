# Research Brief: Signature & Date Compliance Checking

**Date:** 2026-06-01  
**Topic:** Automated detection of signature and date quality issues in immigration forms (PDFs)

---

## Background

Immigration case forms (e.g., I-140, ETA-9089, G-28) require wet-ink handwritten signatures and legible dates in specific fields. When documents are submitted digitally, we need to programmatically verify that signatures and dates meet compliance requirements — before a paralegal reviews them.

Your task is to research how each of the error types below could be detected reliably from a PDF or rasterized page image.

---

## Scope

Research detection approaches for the following **6 error types**. For each, you should propose at least one concrete, implementable approach — preferably backed by existing libraries or models.

---

## Error Types

### 1. `SIGNATURE_MISSING`
A required signature field contains no handwritten signature.

**Research questions:**
- How can you distinguish a blank signature box from one containing ink strokes?
- What image processing or model-based techniques work well for detecting ink presence in a cropped region?
- How do you avoid false positives from pre-printed form content (lines, labels, watermarks)?

---

### 2. `SIGNATURE_WRONG_BOX`
A signature exists on the page but appears to be in the wrong location — outside the designated signature field.

**Research questions:**
- How can you detect whether ink strokes that look like a signature are located in an unexpected area?
- How do you characterize what a "signature-like" cluster of ink looks like geometrically?
- How do you measure spatial overlap between a detected stroke cluster and an expected bounding box?

---

### 3. `SIGNATURE_TOO_FAINT`
A signature is present but too faint or pale to be legible or legally valid.

**Research questions:**
- What image-level metrics (pixel intensity, contrast, etc.) best capture "faintness" of ink?
- How do you isolate the ink pixels from the background before measuring darkness?
- What threshold or range of values would you use to classify a signature as faint vs. acceptable?

---

### 4. `TYPED_SIGNATURE`
The signature field contains typed text rather than a handwritten signature.

**Research questions:**
- What visual or structural features distinguish typed text from handwritten strokes?
- Could OCR be used as part of the detection pipeline? If so, how?
- What signals suggest the content in the box is typed (e.g., uniform stroke width, regular baseline, character spacing)?

---

### 5. `PASTED_IMAGE_SIGNATURE`
The signature is a pasted image (e.g., a PNG or JPEG of a signature embedded in the PDF) rather than a handwritten original.

**Research questions:**
- Can you detect embedded image objects in a PDF's native structure without rasterizing? What libraries support this?
- How do you determine whether an embedded image is located inside a signature field?
- What distinguishes a pasted image signature from a genuine handwritten scan of a wet-ink page?

---

### 6. `DATE_UNREADABLE`
A date field exists but the content is missing, illegible, or not parseable as a valid date.

**Research questions:**
- How can OCR be used to extract text from a specific date box in a form?
- What date formats are common in immigration forms (e.g., mm/dd/yyyy, ROC calendar year)?
- How do you handle cases where OCR returns text but it does not match any known date format?
- How do you handle a completely blank date field?

---

## Deliverables

For each error type, please document:

1. **Proposed detection approach** — describe the method in plain language (e.g., "crop the signature box, apply adaptive thresholding, count connected ink components")
2. **Key signals** — what metrics, features, or model outputs drive the decision?
3. **Libraries / tools** — what Python packages or APIs would you use? (e.g., OpenCV, PyMuPDF, pytesseract, a vision model)
4. **Failure modes** — what cases might fool your approach? How would you handle them?
5. **Rough confidence estimate** — is this a high-confidence binary check or does it require a soft score with a review band?

---

## Constraints & Notes

- Input is a PDF file (may be a scanned raster or a native digital PDF).
- You must work at the crop level: the caller will provide the bounding box of the signature or date field within the page.
- Solutions should be deterministic and fast where possible; LLM-based approaches are acceptable only where simpler methods are insufficient.
- Each error type should be independently detectable — avoid coupling them.
- Do not assume access to any prior version of the document.

---

## Resources to Get Started

- PyMuPDF (`fitz`) — PDF parsing and image extraction
- OpenCV (`cv2`) — image processing
- Any OCR library of your choice (e.g., Google Document AI, Tesseract, Azure OCR)
- Academic literature on handwriting vs. print detection, signature verification

---

## Questions?

Feel free to reach out if any error type is ambiguous. Focus on practical, implementable approaches over theoretical completeness.
