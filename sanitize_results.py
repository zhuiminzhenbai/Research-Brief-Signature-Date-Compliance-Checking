"""Strip PII from the committed result JSONs so the repo can go public.

PII removed:
  - `rel` filenames (contain real applicant CASE IDs)  -> sample_NNN__<form>__<tag>
  - detect4.json `pd` field (the OCR-extracted real SIGNATURE NAME) -> dropped
  - ground_truth.json keys (real case filenames)        -> sample_NNN__<form>__<tag>
Kept (non-PII, the demonstrable results): form type, Good/NG tag, all metrics
(pd_score, cc_h_cv, weber, ink, status, reason...).

Case ID -> sample number is consistent across all files (same case = same sample).
Writes the sanitized JSON back in place. Idempotent-ish (re-running on already
sanitized files leaves them unchanged since there are no case IDs left to map).
"""
import json, re, os

FILES = [
    r"C:\Users\25775\Desktop\OCR_research\detect4_out\detect4.json",
    r"C:\Users\25775\Desktop\OCR_research\detect12_out\detect12.json",
    r"C:\Users\25775\Desktop\OCR_research\detect3_out\detect3.json",
]
GT = r"C:\Users\25775\Desktop\OCR_research\ground_truth.json"

CASE_RE = re.compile(r"(\d{4,6})")          # leading applicant case id
TAG_RE = re.compile(r"_(Good|NG|OK|edited)", re.I)
FORM_RE = re.compile(r"(ETA-9089|I-140 Page ?6|I-140 Page ?8|G-28 Page ?3|I-140|G-28)", re.I)


def basename(rel):
    return rel.replace("\\", "/").split("/")[-1]


def case_id(rel):
    m = CASE_RE.search(basename(rel))
    return m.group(1) if m else None


def form_tag(rel, form_field=None):
    b = basename(rel)
    fm = FORM_RE.search(b)
    form = (form_field or (fm.group(1) if fm else "form")).replace(" ", "-")
    tg = TAG_RE.search(b)
    tag = tg.group(1).upper().replace("GOOD", "Good").replace("EDITED", "edited") if tg else "unlabeled"
    # accepted vs need_correction directory is also a non-PII bucket hint
    return form, tag


def build_map():
    """Collect every case id across all files first, assign stable sample numbers."""
    ids = set()
    for fp in FILES:
        for rec in json.load(open(fp, encoding="utf-8")):
            cid = case_id(rec.get("rel", ""))
            if cid:
                ids.add(cid)
    for k in json.load(open(GT, encoding="utf-8")):
        if k.startswith("_"):
            continue
        cid = case_id(k)
        if cid:
            ids.add(cid)
    return {cid: f"{i:03d}" for i, cid in enumerate(sorted(ids), 1)}


def anon(rel, m, form_field=None):
    cid = case_id(rel)
    num = m.get(cid, "xxx") if cid else "xxx"
    form, tag = form_tag(rel, form_field)
    return f"sample_{num}__{form}__{tag}"


def main():
    m = build_map()
    print(f"mapped {len(m)} unique case IDs -> sample_001..sample_{len(m):03d}")

    for fp in FILES:
        data = json.load(open(fp, encoding="utf-8"))
        names_dropped = 0
        for rec in data:
            if "rel" in rec:
                rec["rel"] = anon(rec["rel"], m, rec.get("form"))
            for fld in rec.get("fields", {}).values():
                if isinstance(fld, dict) and "pd" in fld:   # detect4: drop extracted name
                    if fld["pd"]:
                        names_dropped += 1
                    del fld["pd"]
        json.dump(data, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  {os.path.basename(fp):16} rels anonymized={len(data)}  names dropped={names_dropped}")

    gt = json.load(open(GT, encoding="utf-8"))
    out = {}
    for k, v in gt.items():
        out[k if k.startswith("_") else anon(k, m)] = v
    json.dump(out, open(GT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  ground_truth.json keys anonymized")


if __name__ == "__main__":
    main()
