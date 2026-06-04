"""Synthesize faint signatures (physically faithful: faded ink = original strokes
blended toward paper white + scan noise) to VALIDATE the error3 FAIL threshold.

Builds a faintness ladder over real signatures, runs detect3's band on each level,
reports the band distribution per alpha, and renders a grid so the threshold can be
checked against where strokes become visually illegible.
"""
import os, json
import numpy as np
import cv2
from error3_faint import collect, faint_metrics
from detect3 import band, WEBER_FAIL, WEBER_REVIEW

TPL = r"C:\Users\25775\Desktop\OCR_research\templates.json"
OUT = r"C:\Users\25775\Desktop\OCR_research\faint_out"
COL = {"TOO_FAINT": (0, 0, 230), "REVIEW": (0, 140, 255), "PASS": (0, 150, 0)}
ALPHAS = [0.6, 0.45, 0.33, 0.25, 0.18, 0.12]


def dim_realistic(crop, alpha, noise=3.0):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(float)
    bg = np.percentile(g, 90)
    out = bg - (bg - g) * alpha                      # shrink stroke depth toward paper
    out = out + np.random.normal(0, noise, out.shape)  # light scan noise
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def cell(crop, label, color, w=300, h=84):
    sc = min((w-8)/max(1, crop.shape[1]), (h-22)/max(1, crop.shape[0]))
    rs = cv2.resize(crop, (max(1, int(crop.shape[1]*sc)), max(1, int(crop.shape[0]*sc))))
    out = np.full((h, w, 3), 255, np.uint8)
    out[22:22+rs.shape[0], 4:4+rs.shape[1]] = rs
    cv2.rectangle(out, (1, 20), (w-2, h-2), color, 2)
    cv2.putText(out, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


def main():
    np.random.seed(0)
    tpl = json.load(open(TPL, encoding="utf-8"))
    rows = collect(tpl)
    strong = sorted(rows, key=lambda x: -x["weber"])[:20]   # clear originals to dim

    # band distribution per alpha over the 20 signatures
    print(f"thresholds: FAIL<{WEBER_FAIL}  REVIEW {WEBER_FAIL}-{WEBER_REVIEW}  PASS>={WEBER_REVIEW}")
    print(f"{'alpha':>6} {'medianWeber':>12}  band counts (over {len(strong)} synth-dimmed sigs)")
    for a in ALPHAS:
        webs, bands = [], {"TOO_FAINT": 0, "REVIEW": 0, "PASS": 0}
        for r in strong:
            fe = faint_metrics(dim_realistic(r["crop"], a))
            if fe is None:
                continue
            webs.append(fe["weber"]); bands[band(fe["weber"])] += 1
        print(f"{a:>6} {np.median(webs):>12.3f}  FAIL={bands['TOO_FAINT']:2} REVIEW={bands['REVIEW']:2} PASS={bands['PASS']:2}")

    # visual grid: 6 signatures x alpha ladder
    grid_rows = []
    for r in strong[:6]:
        cells = []
        for a in ALPHAS:
            d = dim_realistic(r["crop"], a)
            fe = faint_metrics(d); b = band(fe["weber"])
            cells.append(cell(d, f"a{a} w{fe['weber']:.2f} {b}", COL[b]))
        grid_rows.append(np.hstack(cells))
    cv2.imwrite(os.path.join(OUT, "synth_ladder_grid.png"), np.vstack(grid_rows))
    print(f"\n-> {os.path.join(OUT, 'synth_ladder_grid.png')}")


if __name__ == "__main__":
    main()
