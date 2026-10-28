"""Film of a live walk (agent/walk.py): the recorded body (agent/duck_recorder.py) replayed in the same MuJoCo scene.
Chase camera + the duck's own eye + the map (truth vs where the duck thinks it is) + what the brain is doing, a "thinks"
line from its state, and subtitles for the words: what the person said, what the duck says and asks (Nemotron, with the
model and latency from the trace's model_decision events), its arrivals, and the patrol report at the end.

uv run --python 3.12 --with-requirements requirements.lock --with pillow==11.3.0 python -m agent.film --house out/houses/<name> [--speed 1]
→ out/walk/<house>/walk.mp4 (Pillow draws the subtitles, accents included; without it they fall back to plain ASCII)
--live: the same picture while agent.walk --listen runs, in an ffplay window (10 fps), with what you said as it's heard
"""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import cv2 as cv
import mujoco
import numpy as np

from agent.twin_sim import ROBOT, ROOT

WIRE = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch", "head_yaw",
        "head_roll", "mouth", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
MODE = {"look": ((80, 190, 255), "looking around"), "go": ((110, 220, 120), "walking to goal"), "wait": ((200, 200, 200), "listening"),
        "back off": ((60, 140, 255), "backing off"), "stuck": ((80, 80, 240), "no route")}
FONT = cv.FONT_HERSHEY_SIMPLEX
TTF = [p for p in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") if Path(p).exists()]
INTRO, OUTRO = 8.0, 10.0  # seconds of film on the command and plan before the walk, and on the report after it


def text(img, s, xy, scale=0.6, color=(235, 235, 235), thick=1):
    cv.putText(img, s, xy, FONT, scale, color, thick, cv.LINE_AA)


def say(img, lines, width, bottom):
    """Subtitles on img (BGR): lines = [(text, (b, g, r), px size)], wrapped to width, in a dark box ending at bottom."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        for i, (t, c, z) in enumerate(lines):
            text(img, t.encode("ascii", "replace").decode()[:110], (40, bottom - 30 * (len(lines) - i)), z / 40, c, 2)
        return img
    fonts = {z: ImageFont.truetype(TTF[0], z) if TTF else ImageFont.load_default(z) for _, _, z in lines}
    pil = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB)).convert("RGBA")
    d = ImageDraw.Draw(pil)
    rows = []
    for t, c, z in lines:
        row = ""
        for w in t.split():
            if row and d.textlength(row + " " + w, font=fonts[z]) > width:
                rows.append((row, c, z))
                row = w
            else:
                row = (row + " " + w).strip()
        rows.append((row, c, z))
    h = sum(int(z * 1.3) for _, _, z in rows) + 28
    x0 = (img.shape[1] - width) // 2 - 20
    box = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    ImageDraw.Draw(box).rounded_rectangle((x0, bottom - h, x0 + width + 40, bottom), 14, fill=(10, 12, 14, 205))
    pil = Image.alpha_composite(pil, box)
    d, y = ImageDraw.Draw(pil), bottom - h + 14
    for t, c, z in rows:
        d.text((x0 + 20, y), t, font=fonts[z], fill=(c[2], c[1], c[0]))
        y += int(z * 1.3)
    return cv.cvtColor(np.array(pil.convert("RGB")), cv.COLOR_RGB2BGR)


def thought(s, goals):
    """What the brain is doing, in its own words: straight from the step's state (no model)."""
    conf, room, goal = s.get("conf", 0), s.get("room_said", "").replace("_", " "), s.get("goal", "")
    if s.get("mode") == "back off":
        return "Bumped into something. Backing off a little."
    if conf < 0.6 or "est" not in s:
        return "Not sure where I am. Turning on the spot to match my depth sensor to the map."
    if s.get("mode") == "wait":
        return f"I'm in the {room or 'doorway'} ({conf:.0%} sure). Listening: tell me where to go."
    if s.get("mode") == "stuck":
        return f"I'm in the {room or 'doorway'}, but I see no way through. Looking around."
    g = next((q["xy"] for q in goals if q["name"] == goal), None)
    dist = math.hypot(s["est"][0] - g[0], s["est"][1] - g[1]) if g else 0
    where = f"I'm in the {room}" if room else "I'm in a doorway"
    return f"{where} ({conf:.0%} sure). Walking to the {goal.replace('_', ' ')}: {dist:.1f} m to go."


def cues(trace, speed):
    """Subtitles from the trace's events: [(wall from, wall to, lines)], one after another (a question, its answer and
    the arrival can share one instant). Durations are film seconds × speed."""
    out, pend = [], {}

    def add(t, dur, lines):
        t = max(t, out[-1][1] if out else t)
        out.append((t, t + dur * speed, lines))
    for e in trace:
        tag = lambda: f"{e['model'].split('/')[-1]} · {e['seconds']:.1f} s" if e.get("model") and e.get("seconds") else ""  # noqa: E731
        if e.get("event") == "model_decision" and e["kind"] == "question":
            pend["q"] = (e["out"], tag())
        elif e.get("event") == "approval":
            q, t = pend.pop("q", (e["question"], ""))
            add(e["wall"], 6, [("Duck: " + q, (255, 255, 255), 30), (f"Nemotron · {t}" if t else "the duck's own words", (120, 220, 160), 20)])
            ans = e.get("heard") or ("yes" if e["answer"] else "no") + {"auto": " (auto-yes in this test run, logged)", "no answer": " (no answer)"}.get(e.get("by"), "")
            add(e["wall"], 3, [("Person: " + ans, (255, 220, 120), 30)])
        elif e.get("event") == "command":  # agent.walk --listen: what was heard, and the plan
            add(e["wall"], 3, [("You: " + e["heard"], (255, 220, 120), 30), (f"heard by Nemotron · {e['asr']}" if e.get("asr") else "typed", (120, 220, 160), 20)])
            if e.get("say"):
                add(e["wall"], 5, [("Duck: " + e["say"], (255, 255, 255), 30), (f"Nemotron · {e['plan']} · plan: " + (", then ".join(
                    g.replace("_", " ") for g in e["goals"]) or "none, asking") if e.get("plan") else "", (120, 220, 160), 20)])
        elif e.get("event") == "arrive":
            add(e["wall"], 4, [(f"Duck: I'm in the {e['goal'].replace('_', ' ')}.", (255, 255, 255), 30),
                               (f"{e['true_dist'] * 100:.0f} cm from the goal point", (170, 170, 165), 20)])
    return out


def map_panel(house, goals, trace, k, size=(640, 600)):
    xs = [p for w in house["walls"] for p in (w["a"][0], w["b"][0])]
    ys = [p for w in house["walls"] for p in (w["a"][1], w["b"][1])]
    lo, hi = np.array([min(xs), min(ys)]) - 0.2, np.array([max(xs), max(ys)]) + 0.2
    sc = min((size[0] - 20) / (hi[0] - lo[0]), (size[1] - 60) / (hi[1] - lo[1]))
    px = lambda p: (int(10 + (p[0] - lo[0]) * sc), int(size[1] - 10 - (p[1] - lo[1]) * sc))  # noqa: E731
    img = np.full((size[1], size[0], 3), 28, np.uint8)
    for r, col in zip(house["rooms"], [(52, 46, 40), (40, 50, 44), (46, 40, 52), (40, 46, 52)]):
        for x0, y0, x1, y1 in r["rects"]:
            cv.rectangle(img, px((x0, y0)), px((x1, y1)), col, -1)
    for o in house["objects"]:
        w, d = o["size"][0] / 2, o["size"][1] / 2
        c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
        pts = np.array([px((o["center"][0] + a * c - b * s, o["center"][1] + a * s + b * c)) for a, b in ((-w, -d), (w, -d), (w, d), (-w, d))])
        cv.fillPoly(img, [pts], (78, 78, 84))
    for w in house["walls"]:
        cv.line(img, px(w["a"]), px(w["b"]), (210, 210, 205), 3)
    for r in house["rooms"]:
        x0, y0, x1, y1 = max(r["rects"], key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
        text(img, r["name"].replace("_", " "), (px((x0, y1))[0] + 6, px((x0, y1))[1] + 20), 0.5, (170, 170, 165))
    steps = [s for s in trace[:k + 1] if "mode" in s]
    if len(steps) > 1:
        cv.polylines(img, [np.array([px(s["true"]) for s in steps])], False, (90, 200, 255), 2, cv.LINE_AA)
    cur = next((trace[j] for j in range(k, -1, -1) if "mode" in trace[j]), trace[k])  # a step, not an event line
    for g in goals:
        done = any(s.get("event") == "arrive" and s["goal"] == g["name"] for s in trace[:k + 1])
        col = (110, 220, 120) if done else (255, 255, 255) if g["name"] == cur.get("goal") else (130, 130, 130)
        cv.drawMarker(img, px(g["xy"]), col, cv.MARKER_TILTED_CROSS if not done else cv.MARKER_STAR, 22, 2)
    if "est" in cur:
        t, e = cur["true"], cur["est"]
        cv.circle(img, px(t), 9, (90, 200, 255), -1, cv.LINE_AA)
        cv.line(img, px(t), px((t[0] + 0.3 * math.cos(t[2]), t[1] + 0.3 * math.sin(t[2]))), (90, 200, 255), 3, cv.LINE_AA)
        a = int(255 * min(1, cur["conf"]))
        cv.circle(img, px(e), 13, (255, 120 + a // 2, 255 - a), 2, cv.LINE_AA)
        cv.line(img, px(e), px((e[0] + 0.35 * math.cos(e[2]), e[1] + 0.35 * math.sin(e[2]))), (255, 120 + a // 2, 255 - a), 2, cv.LINE_AA)
        if cur.get("aim"):
            cv.line(img, px(e), px(cur["aim"]), (110, 220, 120), 1, cv.LINE_AA)
    text(img, "map from the LiDAR scan", (12, 24), 0.55)
    cv.circle(img, (12 + 6, 44), 7, (90, 200, 255), -1)
    text(img, "where it really is", (32, 49), 0.5, (90, 200, 255))
    cv.circle(img, (230, 44), 9, (255, 220, 120), 2)
    text(img, "where it thinks it is", (246, 49), 0.5, (255, 220, 120))
    return img


def main():
    ap = argparse.ArgumentParser(prog="agent.film")
    ap.add_argument("--house", default=str(ROOT / "out/current"))
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--live", action="store_true", help="the walk as it happens (agent.walk --listen), in a window")
    a = ap.parse_args()
    house_dir = Path(a.house).resolve()
    out = ROOT / "out/walk" / house_dir.name
    house = json.loads((house_dir / "house.json").read_text())
    m = mujoco.MjModel.from_xml_path(str(ROBOT / "scene_myhouse.xml"))
    d = mujoco.MjData(m)
    adr = [m.jnt_qposadr[m.joint(n).id] if n != "mouth" else None for n in WIRE]
    chase = mujoco.Renderer(m, 720, 1280)
    eye = mujoco.Renderer(m, 360, 640)
    # the chase camera looks through walls and tall furniture, the duck's eye doesn't
    tall = np.where((m.geom_bodyid == m.body("house").id) & (m.geom_size[:, 2] > 0.05))[0]
    solid, wall_mat = m.geom_rgba[tall, 3].copy(), m.mat("wall").id
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation = 1.6, -48
    az = None

    def draw(tw, row, trace, wall, goals, speed):
        """One frame: the duck's body at row, the brain's state at wall time tw (subtitles go on top, by the caller)."""
        nonlocal az
        d.qpos[:7] = row[1:8]
        for q, v in zip(adr, row[8:]):
            if q is not None:
                d.qpos[q] = v
        mujoco.mj_forward(m, d)
        w, x, y, z = row[4:8]
        yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        az = yaw if az is None else az + 0.04 * ((yaw - az + 180) % 360 - 180)  # a camera operator who follows, gently
        cam.lookat[:] = [row[1], row[2], 0.12]
        cam.azimuth = az + 180 - 25
        m.geom_rgba[tall, 3], m.mat_rgba[wall_mat, 3] = 0.3, 0.2
        chase.update_scene(d, cam)
        m.geom_rgba[tall, 3], m.mat_rgba[wall_mat, 3] = solid, 1.0
        frame = np.zeros((1080, 1920, 3), np.uint8)
        frame[:720, :1280] = cv.cvtColor(chase.render(), cv.COLOR_RGB2BGR)
        eye.update_scene(d, "head_camera")
        frame[:360, 1280:1920] = cv.cvtColor(eye.render(), cv.COLOR_RGB2BGR)
        k = max(0, int(np.searchsorted(wall, tw)) - 1)
        s = next(trace[j] for j in range(k, -1, -1) if "mode" in trace[j]) if any("mode" in t for t in trace[:k + 1]) else trace[0]
        frame[480:1080, 1280:1920] = map_panel(house, goals, trace, k)
        frame[360:480, 1280:1920] = 20
        text(frame, "^ the duck's own eye (head camera, 20 cm above the floor)", (1296, 400), 0.55)
        # the brain, in words
        strip = frame[720:1080, :1280]
        strip[:] = 20
        col, what = MODE.get(s.get("mode"), ((200, 200, 200), "standing up"))
        text(strip, f"{tw - wall[0]:5.1f} s", (24, 46), 1.0)
        text(strip, what, (190, 46), 1.0, col, 2)
        text(strip, f"next goal: {s.get('goal', '').replace('_', ' ') or '-'}", (720, 46), 0.8)
        conf = s.get("conf", 0)
        cv.rectangle(strip, (24, 76), (424, 100), (60, 60, 60), -1)
        cv.rectangle(strip, (24, 76), (24 + int(400 * conf), 100), (110, 220, 120) if conf >= 0.6 else (80, 190, 255), -1)
        text(strip, f"sure of its position: {conf:.0%}", (440, 96), 0.7)
        said, true_room = s.get("room_said", "").replace("_", " "), s.get("room_true", "").replace("_", " ")
        right = conf < 0.6 or said == true_room
        text(strip, "thinks: " + thought(s, goals), (24, 140), 0.7, (255, 220, 120) if right else (80, 80, 240), 2)
        text(strip, f"(truth: in the {true_room or 'doorway'})", (1010, 96), 0.6, (170, 170, 165))
        seen = sorted(set(s.get("seen", []))) + ([f"colours like the {s['colour'].replace('_', ' ')}"] if s.get("colour") else [])
        if seen:
            text(strip, "sees: " + ", ".join(seen), (24, 180), 0.7, (220, 200, 150))
        if s.get("cmd"):
            text(strip, f"command to the walking policy: forward {s['cmd'][0]:.2f} m/s, turn {s['cmd'][1]:+.2f} rad/s", (24, 220), 0.6, (170, 170, 165))
        done = [t for t in trace[:k + 1] if t.get("event") == "arrive"]
        text(strip, "reached: " + ("  ".join(f"{t['goal'].replace('_', ' ')} ({t['true_dist'] * 100:.0f} cm off)" for t in done[-4:]) or "-"),
             (24, 262), 0.7, (110, 220, 120))
        ev = next((t for t in trace[k:max(0, k - 60):-1] if t.get("event") in ("vision_decision", "approval")), None)
        if ev and tw - ev["wall"] < 4:  # OpenCV changed what the brain does, or it asked a person
            msg = (f"OpenCV changed the plan [{ev['cause']}]: {ev['before']['mode']} -> {ev['after']['mode']}, "
                   f"thinks it's in: {ev['after']['room'] or 'a doorway'}".replace("_", " ") if ev["event"] == "vision_decision"
                   else f"asked a person ({ev.get('cause', 'lost')}) -> {'yes' if ev['answer'] else 'no'} ({ev['by']})")
            text(strip, msg, (24, 292), 0.65, (120, 200, 255), 2)
        text(strip, "Pollen Microduck - real robotd + trained walking policy (ONNX, 50 Hz) in MuJoCo - zero stickers", (24, 316), 0.6, (150, 150, 150))
        text(strip, f"house: built from an iPhone LiDAR scan - {'live' if a.live else f'{speed:g}x speed'}", (24, 344), 0.6, (150, 150, 150))
        return frame

    def report_card(frame, report):
        frame[:720, :1280] = (frame[:720, :1280] * 0.35).astype(np.uint8)
        lines = [("Patrol report", (255, 255, 255), 36), (f"written by Nemotron · {report['model'].split('/')[-1]} · {report['seconds']:.1f} s", (120, 220, 160), 20)]
        lines += [(r.strip("-• "), (235, 235, 235), 26) for r in report["out"].splitlines() if r.strip()]
        frame[:720, :1280] = say(frame[:720, :1280], lines, 1100, 660)

    def subtitle(frame, subs, tw):
        cur = [c for c in subs if c[0] <= tw < c[1]]
        if cur:
            frame[:720, :1280] = say(frame[:720, :1280], cur[-1][2], 1100, 700)

    if a.live:  # follow trace.live.jsonl and truth.json as agent.walk writes them; close the window to end
        play = subprocess.Popen(["ffplay", "-loglevel", "error", "-window_title", f"House Twin Patrol - {house_dir.name} - live", "-fflags", "nobuffer",
                                 "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", "1280x720", "-framerate", "10", "-i", "-"], stdin=subprocess.PIPE)
        feed, trace, buf = (out / "trace.live.jsonl").open(), [], ""
        while play.poll() is None:
            tick = time.time()
            if (out / "trace.live.jsonl").stat().st_size < feed.tell():  # a new walk
                feed.seek(0)
                trace, buf = [], ""
            *lines, buf = (buf + feed.read()).split("\n")
            for e in map(json.loads, lines):
                e.setdefault("wall", trace[-1]["wall"] if trace else tick)
                trace.append(e)
            if not any("mode" in e for e in trace):
                time.sleep(0.2)
                continue
            goals = next((e["goal_points"] for e in reversed(trace) if e.get("event") == "command"), [])
            frame = draw(tick, np.array(json.loads((out / "truth.json").read_text())["row"]), trace, np.array([e["wall"] for e in trace]), goals, 1)
            report = next((e for e in trace if e.get("event") == "model_decision" and e["kind"] == "report"), None)
            report_card(frame, report) if report else subtitle(frame, cues(trace, 1), tick)
            try:
                play.stdin.write(cv.resize(frame, (1280, 720), interpolation=cv.INTER_AREA).tobytes())
            except BrokenPipeError:
                break
            time.sleep(max(0.0, 0.1 - (time.time() - tick)))
        return

    body = np.load(out / "body.npy")
    trace = [json.loads(s) for s in (out / "trace.jsonl").read_text().splitlines()]
    summary = json.loads((out / "summary.json").read_text())
    walls = [s.get("wall") for s in trace]
    w0 = next(w for w in walls if w)
    for i, s in enumerate(trace):  # events carry only brain time: give them the wall time of the step before
        if not s.get("wall"):
            s["wall"] = trace[i - 1]["wall"] if i else w0
    wall = np.array([s["wall"] for s in trace])
    fps = 25
    subs = cues(trace, a.speed)
    times = np.arange(max(body[0, 0], w0), min(body[-1, 0], wall[-1] + 3), a.speed / fps)
    if subs and subs[-1][1] > times[-1]:  # hold the last frame until the last subtitle has been read
        times = np.r_[times, np.arange(times[-1], subs[-1][1], a.speed / fps)[1:]]
    plan = next((e for e in trace if e.get("event") == "model_decision" and e["kind"] == "plan"), None)
    report = next((e for e in trace if e.get("event") == "model_decision" and e["kind"] == "report"), None)
    times = np.r_[np.full(int(INTRO * fps) if plan else 0, times[0]), times, np.full(int(OUTRO * fps) if report else 0, times[-1])]
    raw = out / "walk_raw.mp4"
    vw = cv.VideoWriter(str(raw), cv.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
    for n, tw in enumerate(times):
        row = body[min(np.searchsorted(body[:, 0], tw), len(body) - 1)]  # past the recording: the last pose
        frame = draw(tw, row, trace, wall, summary["goal_points"], a.speed)
        intro, outro = n < int(INTRO * fps) and plan, n >= len(times) - int(OUTRO * fps) and report
        if intro:
            p = json.loads(plan["out"])
            first = n < int(INTRO * fps * 0.4)
            lines = [("You: " + plan["in"], (255, 220, 120), 34)] if first else [
                ("Duck: " + (p["say"] or p["ask"]), (255, 255, 255), 34),
                (f"Nemotron · {plan['model'].split('/')[-1]} · {plan['seconds']:.1f} s · plan: " + ", then ".join(g.replace("_", " ") for g in p["goals"]), (120, 220, 160), 20)]
            frame[:720, :1280] = say(frame[:720, :1280], lines, 1100, 700)
        elif outro:
            report_card(frame, report)
        else:
            subtitle(frame, subs, tw)
        vw.write(frame)
        if n % 250 == 0:
            print(f"{n}/{len(times)}", flush=True)
    vw.release()
    final = out / ("walk.mp4" if a.speed == 1 else f"walk_{a.speed:g}x.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
                    "-movflags", "+faststart", str(final)], check=True)
    raw.unlink()
    print(final)


if __name__ == "__main__":
    main()
