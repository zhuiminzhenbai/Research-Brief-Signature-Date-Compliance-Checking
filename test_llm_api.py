"""Quick connectivity/auth test for the LiteLLM proxy models.

Reads endpoint + key from litellm.env (never hard-coded). Sends a tiny chat
request to each model and reports: OK + reply, or the failure (TCP timeout vs
HTTP 4xx/5xx). Does NOT print the API key (only the last 4 chars).

Usage:
    python test_llm_api.py                       # test the default models
    python test_llm_api.py gpt-5.4 gpt-5.4-mini  # test specific models
"""
import json, time, sys, os, urllib.request, urllib.error

ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "litellm.env")
DEFAULT_MODELS = ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini"]


def load_env(path):
    cfg = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                cfg[k] = v
    return cfg


def main():
    cfg = load_env(ENV)
    base = cfg["LITELLM_BASE_URL"].rstrip("/")
    key = cfg["LITELLM_API_KEY"]
    url = base + "/v1/chat/completions"        # if 404, try base + "/chat/completions"
    models = sys.argv[1:] or DEFAULT_MODELS
    print(f"endpoint: {url}\nkey: ...{key[-4:]}\n")

    for m in models:
        body = json.dumps({"model": m,
                           "messages": [{"role": "user", "content": "Reply with exactly: OK"}]}).encode()
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {key}"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.load(r)
            dt = time.time() - t0
            msg = d["choices"][0]["message"]["content"]
            print(f"[OK]   {m:14} {dt:5.1f}s  reply={msg!r}  tokens={d.get('usage', {}).get('total_tokens')}")
        except urllib.error.HTTPError as e:
            print(f"[FAIL] {m:14} HTTP {e.code}  {e.read().decode()[:300]}")
        except Exception as e:
            print(f"[FAIL] {m:14} {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
