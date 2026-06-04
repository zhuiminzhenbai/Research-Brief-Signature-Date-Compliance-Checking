"""Pluggable VLM interface for the typed-vs-handwritten (error4) ambiguous zone.

Model-agnostic by design. Nothing is hard-wired to a vendor: pick a backend and
credentials via environment variables. With no config it runs the 'stub' backend
(no network), returning 'uncertain' so the pipeline degrades gracefully to REVIEW.

Enable a real model with environment variables. Two backends cover all vendors:

  VLM_BACKEND=openai  -> any OpenAI-compatible vision chat endpoint:
    set VLM_BASE_URL=...   GPT     https://api.openai.com/v1                            (gpt-4o-mini)
                           Gemini  https://generativelanguage.googleapis.com/v1beta/openai  (gemini-2.0-flash)
                           Qwen    https://dashscope-intl.aliyuncs.com/compatible-mode/v1   (qwen-vl-max)
                           Doubao  https://ark.cn-beijing.volces.com/api/v3            (doubao-vision-pro)
                           local   http://localhost:11434/v1                           (Ollama / vLLM)
    set VLM_API_KEY=...
    set VLM_MODEL=...      e.g. gpt-4o-mini / gemini-2.0-flash / qwen-vl-max

  VLM_BACKEND=anthropic  -> Claude native Messages API (cheap tier = Haiku):
    set VLM_API_KEY=...    (Anthropic key; or use a Bedrock/Vertex proxy base for PII)
    set VLM_MODEL=...      e.g. claude-haiku-4-5-20251001
    set VLM_BASE_URL=...   optional, defaults to https://api.anthropic.com

Public API:
    classify(image_path) -> {"label": "typed"|"handwritten"|"uncertain",
                             "raw": <model text>, "backend": <name>}
The label drives error4: 'typed' -> TYPED_SIGNATURE, 'handwritten' -> ok,
'uncertain' -> REVIEW (also what you get with no backend configured).
"""
import os, base64, json, re, mimetypes, urllib.request

DEFAULT_PROMPT = (
    "This image is a tight crop of the content written in a signature field on a "
    "scanned government form. Decide whether it is a HANDWRITTEN signature or "
    "TYPED (machine-printed) text. Reply with exactly one word on the first line: "
    "HANDWRITTEN or TYPED. Then optionally one short reason."
)


def _data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def _parse_label(text):
    t = (text or "").strip().upper()
    head = t[:60]
    if "HANDWRIT" in head or "HAND-WRIT" in head or "CURSIVE" in head:
        return "handwritten"
    if "TYPED" in head or "PRINTED" in head or "MACHINE" in head:
        return "typed"
    return "uncertain"


def _backend_stub(path, prompt):
    # No model configured: defer every ambiguous crop to human REVIEW.
    return {"label": "uncertain", "raw": "(stub: no VLM configured)", "backend": "stub"}


def _backend_openai(path, prompt):
    base = os.environ["VLM_BASE_URL"].rstrip("/")
    key = os.environ["VLM_API_KEY"]
    model = os.environ.get("VLM_MODEL", "")
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 60,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _data_url(path)}},
        ]}],
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.load(r)
    text = resp["choices"][0]["message"]["content"]
    return {"label": _parse_label(text), "raw": text, "backend": f"openai:{model}"}


def _backend_anthropic(path, prompt):
    base = os.environ.get("VLM_BASE_URL", "https://api.anthropic.com").rstrip("/")
    key = os.environ["VLM_API_KEY"]
    model = os.environ.get("VLM_MODEL", "claude-haiku-4-5-20251001")
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    body = {
        "model": model,
        "max_tokens": 60,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
            {"type": "text", "text": prompt},
        ]}],
    }
    req = urllib.request.Request(
        base + "/v1/messages",
        data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.load(r)
    text = "".join(b.get("text", "") for b in resp.get("content", []))
    return {"label": _parse_label(text), "raw": text, "backend": f"anthropic:{model}"}


_BACKENDS = {"stub": _backend_stub, "openai": _backend_openai,
             "anthropic": _backend_anthropic}


def classify(image_path, prompt=DEFAULT_PROMPT):
    """Classify a signature-cell crop. Never raises: failures -> 'uncertain'."""
    backend = os.environ.get("VLM_BACKEND", "stub").lower()
    fn = _BACKENDS.get(backend, _backend_stub)
    try:
        return fn(image_path, prompt)
    except Exception as e:
        return {"label": "uncertain", "raw": f"ERROR: {e}", "backend": f"{backend}:error"}


if __name__ == "__main__":
    import glob
    d = r"C:\Users\25775\Desktop\OCR_research\typed_out\vlm_test"
    print(f"backend = {os.environ.get('VLM_BACKEND', 'stub')}")
    for p in sorted(glob.glob(os.path.join(d, "*.png"))):
        r = classify(p)
        print(f"  {os.path.basename(p):14} -> {r['label']:11} [{r['backend']}]  {r['raw'][:50]}")
