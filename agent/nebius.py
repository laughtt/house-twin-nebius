"""NVIDIA models over an OpenAI-compatible API: Nebius Token Factory, or NVIDIA's API catalog for development
(config [nebius] provider). Every call is logged to out/model_calls.jsonl (provider, model id, latency, tokens) so the
docs and the feedback can quote measured numbers. Keys come from the environment or .env."""
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request

import cv2 as cv

from twin import CFG, ROOT

if (ROOT / ".env").exists():
    for line in (ROOT / ".env").read_text().splitlines():
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
NB = CFG["nebius"]
NB["provider"] = os.environ.get("MODEL_PROVIDER", NB["provider"])  # MODEL_PROVIDER=nebius overrides config.toml
P = NB[NB["provider"]]  # base_url, key, and the vl / ultra / lightning / omni model ids
LOG = ROOT / CFG["paths"]["out"] / "model_calls.jsonl"


def _req(path, body=None, timeout=60):
    req = urllib.request.Request(P["base_url"].rstrip("/") + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ[P['key']]}"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def models():
    return sorted(m["id"] for m in _req("/models")["data"])


def image(bgr, side=None):
    """An image part for a chat message: JPEG, long side shrunk to [nebius] image_px (fewer image tokens)."""
    side = side or NB["image_px"]
    s = side / max(bgr.shape[:2])
    if s < 1:
        bgr = cv.resize(bgr, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
    b64 = base64.b64encode(cv.imencode(".jpg", bgr, [cv.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def parse_json(text):
    """The first JSON object in a reply (models wrap it in ```json fences or think aloud first)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def chat(model, messages, max_tokens=512, json_mode=False, tag="", timeout=60, think=False, tools=None):
    """→ (reply text, info). info = the log line: model, tag, seconds, tokens in/out, error, and with tools the calls
    made [[name, arguments], ...]. Nemotron reasons before it answers unless think=False (a room vote with thinking used
    all 300 tokens and returned no answer)."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": think}}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if tools:
        body["tools"], body["tool_choice"] = tools, "auto"
    t0, text, info = time.time(), "", {"provider": NB["provider"], "model": model, "tag": tag}
    try:
        for n in range(3):  # rate limited, or the service busy for a moment (NVIDIA's free tier: 503s): try again after a pause
            try:
                r = _req("/chat/completions", body, timeout)
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 502, 503, 504) or n == 2:
                    raise
                time.sleep(3 if e.code == 429 else 1)
        msg = r["choices"][0]["message"]
        text = msg.get("content") or ""
        info.update(tokens_in=r["usage"]["prompt_tokens"], tokens_out=r["usage"]["completion_tokens"])
        if tools:
            info["calls"] = [[c["function"]["name"], json.loads(c["function"]["arguments"] or "{}")] for c in msg.get("tool_calls") or []]
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"[:300]
    info.update(seconds=round(time.time() - t0, 3), t=round(t0, 3))
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(info) + "\n")
    return text, info


def cost(info):
    """$ for one call, from [nebius.price] (per 1M tokens in/out)."""
    p = NB["price"].get(info["model"]) or NB["price"].get(info["model"].removesuffix("-fast"), [0, 0])
    return (info.get("tokens_in", 0) * p[0] + info.get("tokens_out", 0) * p[1]) / 1e6


if __name__ == "__main__":
    # the provider's models, then each configured id: found, or the closest names to pick from
    have = models()
    print("\n".join(have), f"\n\nprovider: {NB['provider']} ({P['base_url']})")
    for job in ("vl", "ultra", "lightning", "omni"):
        near = [m for m in have if m.lower() == P[job].lower()] or [m for m in have if "nemotron" in m.lower() and P[job].split("/")[-1].lower()[:12] in m.lower()]
        print(f"{job:10} {P[job]:55} {'ok' if P[job] in have else 'MISSING, closest: ' + (', '.join(near) or 'none')}")
