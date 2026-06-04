"""Complete per-form signature/date field definitions (all fields, not just primary).

Each field is located by (regex + order) among y-sorted matches in the OCR items,
so fields whose label text is identical (e.g. ETA's three 'Signature *') are
disambiguated by vertical order. Box offsets live in templates.json (anchor-relative,
units of anchor label height); DEFAULT_OFFSETS seed them before manual tuning.
"""
import re

# field: name, required, sig anchor (regex/exclude/order), date anchor (regex/order)
FIELDS = {
    "I-140 P6": [
        dict(name="petitioner", required=True,
             sig_re=r"(Petitioner|Authorized Signatory).{0,45}Signatur\w*",
             sig_excl=r"interpreter", sig_order=0,
             date_re=r"Dat\w* of Signatur\w*", date_order=1, date_dir="right"),
        dict(name="interpreter", required=False,
             sig_re=r"Interpreter.{0,3}\s*Signatur\w*", sig_excl=None, sig_order=0,
             date_re=r"Dat\w* of Signatur\w*", date_order=0, date_dir="right"),
    ],
    "I-140 P8": [
        # OCR often merges the signature into the label ("Signature: Ru"); match label prefix.
        dict(name="signatory", required=True,
             sig_re=r"^Signatur\w*\s*:", sig_excl=None, sig_order=0,
             date_re=r"^Dat\w*\s*:", date_order=0, date_dir="right"),
    ],
    "G-28 P3": [
        dict(name="attorney", required=True,
             sig_re=r"\d\.?\s*a\.\s*Signatur\w*\s+of\s+Attorney", sig_excl=r"^Part\b", sig_order=0,
             date_re=r"Dat\w* of Signatur\w*", date_order=0, date_dir="right"),
        dict(name="law_student", required=False,
             sig_re=r"Signatur\w*\s+of\s+Law\s+Student", sig_excl=None, sig_order=0,
             date_re=r"Dat\w* of Signatur\w*", date_order=1, date_dir="right"),
        dict(name="client", required=True,
             sig_re=r"\d\.?\s*a\.\s*Signatur\w*\s+of\s+Client", sig_excl=None, sig_order=0,
             date_re=r"Dat\w* of Signatur\w*", date_order=2, date_dir="right"),
    ],
    "ETA-9089": [
        dict(name="foreign_worker", required=True,
             sig_re=r"Signatur\w*\s*\*?\s*$", sig_excl=r"\bof the\b|virtue|hereby|certify",
             sig_order=0, date_re=r"Dat\w*\s+Signed", date_order=0, date_dir="below"),
        dict(name="attorney", required=True,
             sig_re=r"Signatur\w*\s*\*?\s*$", sig_excl=r"\bof the\b|virtue|hereby|certify",
             sig_order=1, date_re=r"Dat\w*\s+Signed", date_order=1, date_dir="below"),
        dict(name="employer", required=False,
             sig_re=r"Signatur\w*\s*\*?\s*$", sig_excl=r"\bof the\b|virtue|hereby|certify",
             sig_order=2, date_re=r"Dat\w*\s+Signed", date_order=2, date_dir="below"),
    ],
}

# seed offsets [a,b,c,d] relative to anchor top-left, unit = anchor height lh
DEFAULT_OFFSETS = {
    "I-140 P6":  {"sig": [0.0, 1.1, 19.9, 3.5],  "date": [14.3, -0.1, 21.3, 1.2]},
    "I-140 P8":  {"sig": [2.95, -0.5, 10.1, 1.2], "date": [2.2, -0.1, 8.1, 1.3]},
    "G-28 P3":   {"sig": [0.0, 1.1, 18.0, 3.0],  "date": [11.6, -0.2, 17.2, 1.2]},
    "ETA-9089":  {"sig": [0.0, 1.1, 37.0, 4.5],  "date": [0.0, 1.1, 6.5, 3.1]},
}


def _matches(items, rgx, excl, order):
    rgx = re.compile(rgx, re.I)
    exc = re.compile(excl, re.I) if excl else None
    cand = [it for it in items if rgx.search(it["text"]) and not (exc and exc.search(it["text"]))]
    cand.sort(key=lambda it: (it["box"][1] + it["box"][3]) / 2)  # y order
    if not cand or order >= len(cand):
        return None
    return cand[order]


def locate_field(items, fld):
    """Return (sig_anchor_item, date_anchor_item) or (None, None) if not found."""
    sig = _matches(items, fld["sig_re"], fld.get("sig_excl"), fld["sig_order"])
    dat = _matches(items, fld["date_re"], None, fld["date_order"])
    return sig, dat


def get_fields(form, required_only=True):
    """Fields for a form; required-only by default (the locked compliance set)."""
    return [f for f in FIELDS[form] if (f["required"] or not required_only)]
