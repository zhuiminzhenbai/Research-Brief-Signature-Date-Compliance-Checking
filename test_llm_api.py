"""Quick connectivity/auth test for the LiteLLM proxy models.

Reads endpoint + key from litellm.env (never hard-coded). Sends a tiny chat
request to each model and reports: OK + reply, or the failure (TCP timeout vs
HTTP 4xx/5xx). Does NOT print the API key (only the last 4 chars).

Usage:
    python test_llm_api.py                       # test the default models
    python test_llm_api.py gpt-5.4 gpt-5.4-mini  # test specific models
"""
import json, time, sys, os, base64, urllib.request, urllib.error

ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "litellm.env")
DEFAULT_MODELS = ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini"]
VISION_PROMPT = ("Look at the image. Is the signature/text TYPED (a computer font) "
                 "or HANDWRITTEN? Reply with exactly one word: typed or handwritten.")


def load_env(path):
    cfg = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                cfg[k] = v
    return cfg


def post(url, key, body, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def make_synthetic_crops():
    """Two SYNTHETIC test images (no PII): a typed name and a freehand scribble.
    Saved to the gitignored testf_review/ tree."""
    import numpy as np, cv2
    out = os.path.join(os.path.dirname(ENV), "testf_review", "maskviz")
    os.makedirs(out, exist_ok=True)
    # typed
    a = np.full((120, 420, 3), 255, np.uint8)
    cv2.putText(a, "John Smith", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 2)
    pa = os.path.join(out, "_vt_typed.png"); cv2.imwrite(pa, a)
    # handwritten-ish: connected wavy strokes
    b = np.full((120, 420, 3), 255, np.uint8)
    pts = np.array([[30, 80], [55, 40], [80, 90], [110, 45], [140, 85], [175, 50],
                    [210, 95], [250, 45], [300, 80], [360, 55]], np.int32)
    cv2.polylines(b, [pts], False, (15, 15, 60), 3, cv2.LINE_AA)
    cv2.line(b, (30, 100), (380, 100), (15, 15, 60), 2, cv2.LINE_AA)
    pb = os.path.join(out, "_vt_hand.png"); cv2.imwrite(pb, b)
    return [("typed-img", pa), ("handwritten-img", pb)]


def vision_test(url, key, models, images=None):
    if images:
        print("=== VISION test (REAL crops) ===")
        cases = [(os.path.basename(p), p) for p in images]
    else:
        print("=== VISION test (synthetic, no PII) ===")
        cases = make_synthetic_crops()
    for m in models:
        for label, path in cases:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            body = {"model": m, "messages": [{"role": "user", "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]}
            t0 = time.time()
            try:
                d = post(url, key, body)
                dt = time.time() - t0
                msg = d["choices"][0]["message"]["content"].strip()
                u = d.get("usage", {})
                print(f"[OK]   {m:14} {label:16} {dt:5.1f}s  reply={msg!r}  "
                      f"tokens={u.get('total_tokens')} (in={u.get('prompt_tokens')} out={u.get('completion_tokens')})")
            except urllib.error.HTTPError as e:
                print(f"[FAIL] {m:14} {label:16} HTTP {e.code}  {e.read().decode()[:200]}")
            except Exception as e:
                print(f"[FAIL] {m:14} {label:16} {type(e).__name__}: {e}")


def main():
    cfg = load_env(ENV)
    base = cfg["LITELLM_BASE_URL"].rstrip("/")
    key = cfg["LITELLM_API_KEY"]
    url = base + "/v1/chat/completions"        # if 404, try base + "/chat/completions"
    argv = sys.argv[1:]
    vision = "--vision" in argv
    images = []
    if "--img" in argv:                       # everything after --img = real image paths
        i = argv.index("--img")
        images = argv[i + 1:]
        argv = argv[:i]
    args = [a for a in argv if not a.startswith("--")]
    models = args or (["gpt-5.4-mini"] if vision else DEFAULT_MODELS)
    print(f"endpoint: {url}\nkey: ...{key[-4:]}\n")

    if vision:
        vision_test(url, key, models, images or None)
        return

    for m in models:
        t0 = time.time()
        try:
            d = post(url, key, {"model": m,
                                "messages": [{"role": "user", "content": "Reply with exactly: OK"}]})
            dt = time.time() - t0
            msg = d["choices"][0]["message"]["content"]
            print(f"[OK]   {m:14} {dt:5.1f}s  reply={msg!r}  tokens={d.get('usage', {}).get('total_tokens')}")
        except urllib.error.HTTPError as e:
            print(f"[FAIL] {m:14} HTTP {e.code}  {e.read().decode()[:300]}")
        except Exception as e:
            print(f"[FAIL] {m:14} {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
