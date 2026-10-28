"""Which room is this? On real iPhone frames: HSV colour vote (the duck's camera today) vs Nemotron VL vs both
(Nemotron when sure, else HSV). Two truths: "in" = the room the person tapped (where the phone is), "view" = the room
the camera looks at (LiDAR depth at the image centre, placed on the floor plan). In an open-plan flat they differ.

uv run --python 3.12 --with opencv-python-headless==5.0.0.93 python eval/vl_rooms.py <scan.zip> [--every N] [--ask in|view]
    [--refs N] [--hsv-only] [--model ID]

- map: out/houses/<session>/house.json; each room's colour fingerprint was built from these same frames (optimistic for HSV).
- frames: every Nth frame with normal tracking, turned upright.
- --refs N: N photos of each room from the same walk go with every question ("this home's kitchen looks like this"),
  the semantic twin of the colour fingerprints; frames within 12 of a reference photo are not tested.
- --ask: the question, "which room is the camera in" (in) or "which room does the photo show" (view, the default).
- out: out/vl_rooms/<session>/results.json (accuracy, p50/p95 latency, tokens, $ per frame) + frames.jsonl (every answer).
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from agent import nebius  # noqa: E402
from agent.localize import HouseMap  # noqa: E402
from agent.perceive import colour_hist  # noqa: E402
from twin.landmarks import _rotation  # noqa: E402
from twin.scan_io import Export, kmat  # noqa: E402

PROMPT = """This photo was taken by a small home robot's camera inside one home. {question}
Rooms in this home: {rooms}. Look at the furniture and fittings, not the colours alone.
Answer with one JSON object: {{"room": one of the room names exactly, "sure": 0 to 1, "why": at most 8 words}}"""


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def key(name):
    return name.strip().lower().replace(" ", "_")


ex = Export(sys.argv[1])
EVERY, HSV_ONLY, MODEL, ASK, REFS = int(arg("--every", 1)), "--hsv-only" in sys.argv, arg("--model", nebius.P["vl"]), arg("--ask", "view"), int(arg("--refs", 0))
QUESTION = {"in": "Which room is the camera in?", "view": "Which room does the photo show?"}[ASK]
house = json.loads((ROOT / "out/houses" / ex.session / "house.json").read_text())
hmap = HouseMap(house)
sf = json.loads((ROOT / "out/houses" / ex.session / "report.json").read_text())["scan_frame"]
c, s_ = np.cos(sf["theta"]), np.sin(sf["theta"])
R2, O2 = np.array([[c, s_], [-s_, c]]), np.array(sf["origin"])
rooms = [r for r in house["rooms"] if "colour_hs" in r]
names = [r["name"] for r in rooms]
OUT = ROOT / "out/vl_rooms" / ex.session
OUT.mkdir(parents=True, exist_ok=True)
frames = [f for f in ex.frames if f["tracking"] == "normal" and key(f["room"]) in names]
refs = {n: [fs[int(len(fs) * (j + 1) / (REFS + 1))] for j in range(REFS)] for n in names
        for fs in [[f for f in frames if key(f["room"]) == n]]}
frames = [f for f in frames if all(abs(f["i"] - r["i"]) > 12 for rs in refs.values() for r in rs)][::EVERY]


def upright(f, conf=2):
    d, P = ex.load(f, conf)
    img = ex.rgb(f)
    rot = _rotation(P[:3, :3].T @ [0, 0, 1])
    return (img if rot is None else cv.rotate(img, rot)), d, P


REF_PARTS = [p for n, rs in refs.items() for p in [{"type": "text", "text": f"This home's {n}:"}] + [nebius.image(upright(r)[0], 384) for r in rs]]


def in_view(f, d, P):
    """The room the camera looks at: the median LiDAR depth in the middle of the picture, as a point on the floor plan."""
    h, w = d.shape
    win = d[h * 2 // 5:h * 3 // 5, w * 2 // 5:w * 3 // 5]
    if (win > 0).sum() < 20:
        return None
    ray = np.linalg.inv(kmat(f["K_depth"])) @ [w / 2, h / 2, 1.0] * float(np.median(win[win > 0]))
    xy = R2 @ (P[:3, :3] @ ray + P[:3, 3])[:2] - O2
    k = int(hmap.room(xy[0], xy[1]))
    return key(house["rooms"][k]["name"]) if k >= 0 else None


def vl(img):
    content = REF_PARTS + [{"type": "text", "text": PROMPT.format(question=QUESTION, rooms=", ".join(names)) +
                            ("\nThe photos above show each room of this home; the room names are this home's own." if REFS else "")}, nebius.image(img)]
    text, info = nebius.chat(MODEL, [{"role": "user", "content": content}], max_tokens=200, json_mode=True, tag="vl_rooms")
    try:
        ans = nebius.parse_json(text)
        room, sure = key(str(ans.get("room", ""))), float(ans.get("sure", 0))
    except (ValueError, TypeError, AttributeError):
        room, sure, ans = None, 0.0, {}
    return {"vl": room if room in names else None, "vl_sure": sure, "vl_why": str(ans.get("why", ""))[:80], "vl_raw": text[:300],
            **{k: info.get(k) for k in ("seconds", "tokens_in", "tokens_out", "error")}, "usd": nebius.cost(info)}


def one(f):
    img, d, P = upright(f, conf=1)
    hist = colour_hist(img)
    row = {"i": f["i"], "truth": key(f["room"]), "view": in_view(f, d, P),
           "hsv": names[int(np.argmax([np.sum(np.sqrt(hist * np.array(r["colour_hs"]))) for r in rooms]))]}
    return row if HSV_ONLY else row | vl(img)


with ThreadPoolExecutor(1 if HSV_ONLY else 8) as pool:
    rows = list(pool.map(one, frames))
(OUT / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def acc(pred, truth="truth"):
    return round(100 * float(np.mean([p == r[truth] for p, r in zip(pred, rows) if r[truth]])), 1)


res = {"session": ex.session, "frames": len(rows), "rooms": names, "note": "HSV fingerprints built from these frames: optimistic for HSV",
       "per_room_frames": {n: sum(r["truth"] == n for r in rows) for n in names},
       "view_known_frames": sum(bool(r["view"]) for r in rows), "view_differs_from_tap_pct": round(100 * float(np.mean(
           [r["view"] != r["truth"] for r in rows if r["view"]])), 1),
       "hsv_right_pct": acc([r["hsv"] for r in rows]), "hsv_right_view_pct": acc([r["hsv"] for r in rows], "view")}
if not HSV_ONLY:
    s = np.array([r["seconds"] for r in rows if not r.get("error")] or [np.nan])
    res |= {"model": MODEL, "provider": nebius.NB["provider"], "ask": ASK, "refs_per_room": REFS, "ref_frames": {n: [r["i"] for r in rs] for n, rs in refs.items()}, "vl_right_pct": acc([r["vl"] for r in rows]),
            "vl_right_view_pct": acc([r["vl"] for r in rows], "view"),
            "both_right_pct": acc([r["vl"] if r["vl"] and r["vl_sure"] >= 0.5 else r["hsv"] for r in rows]),
            "vl_no_answer_pct": round(100 * float(np.mean([r["vl"] is None for r in rows])), 1),
            "errors": sum(bool(r.get("error")) for r in rows),
            "latency_p50_s": round(float(np.percentile(s, 50)), 2), "latency_p95_s": round(float(np.percentile(s, 95)), 2),
            "tokens_in_per_frame": round(float(np.mean([r["tokens_in"] or 0 for r in rows]))),
            "usd_per_1000_frames": round(1000 * float(np.mean([r["usd"] for r in rows])), 3)}
(OUT / "results.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
