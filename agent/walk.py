"""The patrol of agent/mission.py on Pollen's duck-sim: the real robotd walks the duck with its trained policies in the
MuJoCo house, our brain steers. It steers by robotd's odometry, tofd's 8x8 ToF, OpenCV 5 on the duck's own head-camera
frames (room colour, ArUco markers; duck-sim with DUCK_SIM_CAMERAS=a) and object sightings from the twin's stand-in
detector at the true pose (agent/duck_recorder.py's truth.json, which also keeps the score). Lost for LOST_S, it asks a
person whether to keep looking (--ask; default: yes, logged); declined, it stops.
Nemotron does the words, never the steering (agent/language.py): --command → which rooms, in which order (Ultra); with a
model key, Ultra also phrases the questions to the person and writes report.md. Each call is a model_decision trace event.
--listen: the Mac's microphone gives the commands, any time (agent/voice.py); a new one replaces the plan, "para" stops,
the duck answers and asks aloud and waits between commands. Ctrl-C ends it. trace.live.jsonl grows as it goes (agent.film --live).

python -m agent.walk --house out/houses/<name> [--goals kitchen,hallway,living_room | --command "check the stove, then the bed"]
    [--goal-s 240] [--ask] [--no-standin] [--listen]
Out: out/walk/<house>/trace.jsonl (same fields as a mission trace, plus wall time, the command sent and the events
vision_decision, approval and model_decision: agent/README.md), summary.json, report.md; plan.json when it asks first.
"""
import argparse
import json
from collections import Counter
import math
import os
import signal
import time
from pathlib import Path

import numpy as np

from agent import language, nebius, perceive, voice
from agent.duck_rpc import Duck
from agent.localize import ParticleFilter, _wrap
from agent.loop import follow, route
from agent.mission import ARRIVE, ARRIVE_SURE, HZ, MARGIN, SQUEEZE, SURE, TURN, VX, goals
from agent.twin_sim import ROOT, TwinSim

LOST_S = 20  # looking around this long (in all) on one goal without knowing where it is: ask a person whether to go on


def gait(v, w):
    """The brain's intent → what Pollen's walking policy does (measured in this flat): it doesn't step below ~0.2 m/s
    forward or ~1.2 rad/s on the spot, walks at ~40 % of the asked speed, turns on the spot at ~0.9 rad/s."""
    if v < 0:
        return -0.3, 0.0
    if v < 0.05:
        return (0.0, math.copysign(1.5, w)) if abs(w) > 0.1 else (0.0, 0.0)
    return max(0.2, 1.6 * v), w


def lost_tick(pf, mode):
    """Seconds of being lost this tick: looking around, but not the spin match's normal first turns (a turn on the spot
    is how it finds itself; ~7 s live). Replayed on the live walk: 2 questions in 400 s instead of 7."""
    return (mode == "look" and (pf.spin is None or pf.spin.matched or pf.spin.tries >= 2)) / HZ


def colour_doubt(votes, goal):
    """Before "I'm in the <goal>": the room the camera's colours named most over the last 5 s. Another room → the
    question for a person (the vote names the true room at 45 of 48 arrivals, eval/vision_walk.py); same room → None."""
    vote = Counter(v for v in votes if v).most_common(1)
    if vote and vote[0][0] != goal:
        return f"I think I'm in the {goal.replace('_', ' ')}, but the camera's colours say {vote[0][0].replace('_', ' ')}. Say I've arrived?"


def approve(question, ask=False):
    """A person's OK to go on in doubt. Default: yes (logged in the trace). ask: on the terminal; the loop waits, and
    robotd stops the duck 500 ms after the last command, so it stands still while it asks."""
    return not ask or input(f"{question} [Y/n] ").strip().lower() in ("", "y", "yes")


def vision_decision(hmap, room_name, goal, before, after, seen, vote):
    """Did the camera change what the brain does: look around or act on its guess, the room it believes it is in, or
    (30 cm+) where its route starts? → a trace event with the decision before and after the frame, or None."""
    (e0, c0), (e1, c1) = before, after
    if (c0 >= SURE) == (c1 >= SURE) and room_name(e0[0], e0[1]) == room_name(e1[0], e1[1]) and \
            (c1 < SURE or math.hypot(e1[0] - e0[0], e1[1] - e0[1]) < 0.3):
        return None

    def decide(est, conf):
        path = conf >= SURE and (route(hmap, est[:2], goal, MARGIN) or route(hmap, est[:2], goal, SQUEEZE))
        return {"mode": "go" if path else "stuck" if conf >= SURE else "look", "room": room_name(est[0], est[1]),
                "est": [round(float(v), 3) for v in est], "conf": round(conf, 2), "aim": [round(v, 2) for v in path[0]] if path else None}
    return {"event": "vision_decision", "cause": ", ".join(f"aruco {i}" for i in seen) or f"room colour ({vote})",
            "before": decide(e0, c0), "after": decide(e1, c1)}


def main():
    ap = argparse.ArgumentParser(prog="agent.walk")
    ap.add_argument("--house", default=str(ROOT / "out/current"))
    ap.add_argument("--goals", default="", help="room names in visiting order (default: all, as the mission does)")
    ap.add_argument("--command", default="", help="what the person asks, in any language: Nemotron picks the rooms")
    ap.add_argument("--goal-s", type=float, default=240)
    ap.add_argument("--ask", action="store_true", help="ask on the terminal before going on lost (default: yes, logged)")
    ap.add_argument("--no-standin", action="store_true", help="no stand-in object detector: ToF and the camera only")
    ap.add_argument("--listen", action="store_true", help="commands by voice from the Mac's microphone, or typed; Ctrl-C ends")
    a = ap.parse_args()
    house_dir = Path(a.house).resolve()
    out = ROOT / "out/walk" / house_dir.name
    sim = TwinSim(house_dir)
    frame = json.loads((house_dir / "sim_frame.json").read_text())
    c, s, (px, py) = math.cos(frame["theta"]), math.sin(frame["theta"]), frame["pos"]

    def truth():  # MuJoCo world → house frame
        d = json.loads((out / "truth.json").read_text())
        wx, wy = d["xyz"][0] - px, d["xyz"][1] - py
        return (c * wx + s * wy, -s * wx + c * wy, float(_wrap(d["yaw"] - frame["theta"]))), d["xyz"][2]

    hmap, rooms = sim.world, [r["name"] for r in sim.house["rooms"]]
    room_name = lambda x, y: rooms[k] if (k := int(hmap.room(x, y))) >= 0 else ""  # noqa: E731
    todo, pre = goals(sim), []
    every = {g["name"]: g for g in todo}
    out.mkdir(parents=True, exist_ok=True)
    words = bool(os.environ.get(nebius.P["key"]))  # with a key: Ultra phrases the questions and the report; else the templates

    def said(kind, info, text_in, text_out):
        return {"t": 0.0, "event": "model_decision", "kind": kind, "model": info.get("model"), "seconds": info.get("seconds"),
                "in": text_in[:500], "out": text_out[:3000], **({"error": info["error"][:120]} if "error" in info else {})}
    if a.command:
        command = a.command
        for _ in range(3):  # the plan, or a question back to the person (with --ask, their answer and a new plan)
            p = language.plan(command, sim.house, places=[g["name"] for g in todo])
            pre.append(said("plan", {"model": p["by"], "seconds": p.get("seconds"), **({"error": p["error"]} if "error" in p else {})},
                            command, json.dumps({k: p[k] for k in ("goals", "ask", "say")}, ensure_ascii=False)))
            print("plan:", p["goals"], p["say"] or p["ask"], f"({p['by']})", flush=True)
            if not p["ask"] or not a.ask:
                break
            command = f"{a.command}\nThe robot asked: {p['ask']}\nThe person answered: {input(p['ask'] + ' ')}"
        a.goals = ",".join(p["goals"])
        if not p["goals"]:
            print("not moving: the command needs an answer first.", flush=True)
            (out / "plan.json").write_text(json.dumps({"command": a.command, "asked": p["ask"], "model_calls": pre}, indent=1, ensure_ascii=False))
            return  # plan.json only: a walk's own trace and summary in this folder stay as they were
    if a.goals:
        todo = [every[n] for n in a.goals.split(",") if n in every]
    ear = voice.Ear(sim.house, list(every)) if a.listen else None
    if ear:
        todo = []

    def phrase(q, t):  # the question in plain words, every fact kept (the template stands without a key or on any error)
        if not words:
            return q
        q2, info = language.question(q, ear.lang if ear else "")
        trace.append(said("question", info, q, q2) | {"t": round(t, 2)})
        return q2
    duck = Duck()
    while duck.state is None or duck.tof is None:
        time.sleep(0.1)
    try:
        img = duck.frame()[0]
        f = img.shape[1] / 2 / math.tan(math.radians(45) / 2)  # head camera: 45° across the upright frame
        K = np.array([[f, 0, img.shape[1] / 2], [0, f, img.shape[0] / 2], [0, 0, 1]])
        for r, hist in zip(hmap.rooms, sim.room_colours()):  # what each room looks like to a camera in this twin
            if hist:
                r["colour_hs"] = hist
    except (OSError, KeyError):  # no frame socket, or mediad up without a picture
        K = None
        print("no camera (duck-sim with DUCK_SIM_CAMERAS=a): ToF and stand-in only", flush=True)
    pf = ParticleFilter.anywhere(hmap, seed=0)
    true, _ = truth()
    start, last, seq = true, duck.state["odom"], None
    t0 = time.time()
    trace, results, bumps, found, back, history, stopped, votes = pre, [], 0, None, 0, [], False, []
    think, steps, plans, went, logged = 0.0, 0, 0, [], 0  # the brain's own compute
    live = out / "trace.live.jsonl"
    live.write_text("")

    def ask(q, t, cause=None):  # a person's OK: by voice (silence = yes), on the terminal (--ask), or yes, logged
        q = phrase(q, t)
        ok, heard = ear.ask(q) if ear else (approve(q, a.ask), None)
        trace.append({"t": round(t, 2), "wall": round(time.time(), 3), "event": "approval", "question": q, "answer": ok,
                      "by": "voice" if heard else "no answer" if ear else "terminal" if a.ask else "auto", "heard": heard,
                      **({"cause": cause} if cause else {}), "goal": g["name"]})
        return ok
    bye = []  # Ctrl-C: stop, then the summary and the report as usual
    signal.signal(signal.SIGINT, lambda *_: bye.append(1))
    while (todo or ear) and not bye:
        g = todo.pop(0) if todo else None  # None: no goal yet, listening
        went += [g] if g else []
        g0, path, last_plan, goal_bumps, lost, doubt, new = time.time(), None, -9.0, 0, 0.0, -1.0, None
        while (not g or time.time() - g0 < a.goal_s) and goal_bumps < 12 and not stopped and not bye:
            tick, t = time.time(), time.time() - t0
            if ear and (order := ear.command()):  # heard (Omni) and planned (Ultra) off the loop, in agent/voice.py
                asr, p = order["asr"], order.get("plan") and {"model": order["plan"]["by"], **order["plan"]}
                if asr.get("model"):
                    trace.append(said("asr", asr, "speech", order["heard"]) | {"t": round(t, 2)})
                if p:
                    trace.append(said("plan", p, order["command"], json.dumps({q: p[q] for q in ("goals", "ask", "say")}, ensure_ascii=False)) | {"t": round(t, 2)})
                if order["goals"] or order["stop"]:
                    new = [every[n] for n in order["goals"]]
                trace.append({"t": round(t, 2), "wall": round(tick, 3), "event": "command", **{k: order[k] for k in ("heard", "via", "say", "goals")},
                              **{k: f"{i['model'].split('/')[-1]} · {i['seconds']:.1f} s" for k, i in (("asr", asr), ("plan", p)) if i and i.get("seconds")},
                              "goal_points": new if new is not None else [g] if g else []})
                if new is not None:
                    if g:
                        results.append({"goal": g["name"], "t": round(time.time() - g0, 1), "declared": False, "why": "new command"})
                    todo = list(new)
                    break
            true, trunk_z = truth()
            objs = [] if a.no_standin else sim.objects(*true)
            c0 = time.perf_counter()
            od = duck.state["odom"]  # odometry since the last tick, in the duck's own frame
            dx, dy = od["position"][0] - last["position"][0], od["position"][1] - last["position"][1]
            cl, sl = math.cos(last["yaw"]), math.sin(last["yaw"])
            pf.predict(dx * cl + dy * sl, -dx * sl + dy * cl, float(_wrap(od["yaw"] - last["yaw"])))
            last = od
            if duck.tof["seq"] != seq:
                seq = duck.tof["seq"]
                pf.update_tof(np.array(duck.tof["distance_mm"]), np.array(duck.tof["status"]))
            pf.update_landmarks(objs, sim.hfov)
            before, seen, vote = (pf.estimate(), pf.confidence()), [], None
            if K is not None:
                try:
                    seen, vote = perceive.see(pf, duck.frame()[0], K)
                except (OSError, KeyError):
                    pass  # a frame late or lost: this tick without the camera
            est, conf = pf.estimate(), pf.confidence()
            votes = (votes + [vote])[-5 * HZ:]
            if K is not None and g and (ev := vision_decision(hmap, room_name, g["xy"], before, (est, conf), seen, vote)):
                trace.append({"t": round(t, 2), **ev, "goal": g["name"], "true": [round(v_, 3) for v_ in true]})
                last_plan = -9.0  # plan the route again from the corrected guess
            err = math.hypot(est[0] - true[0], est[1] - true[1])
            if found is None and conf >= SURE and err < 0.3:
                found = t
            mode, v, w, ahead = "look", 0.0, 0.6, None
            if back:
                mode, v, w, back = "back off", -0.1, 0.0, back - 1
            elif conf >= SURE and not g:
                mode, w = "wait", 0.0
            elif conf >= SURE:
                at = math.hypot(est[0] - g["xy"][0], est[1] - g["xy"][1]) < ARRIVE and conf >= ARRIVE_SURE and t > doubt
                q = at and K is not None and colour_doubt(votes, g["name"])
                if q:  # the camera disagrees: a person decides
                    ok = ask(q, t, "room colour")
                    print("asked:", trace[-1]["question"], "->", "yes" if ok else "no: not there yet", flush=True)
                    at, doubt = ok, t if ok else t + 10
                if at:
                    results.append({"goal": g["name"], "t": round(time.time() - g0, 1), "declared": True,
                                    "true_dist": round(math.hypot(true[0] - g["xy"][0], true[1] - g["xy"][1]), 3),
                                    "room_said": room_name(est[0], est[1]), "room_true": room_name(true[0], true[1])})
                    trace.append({"t": round(t, 2), "event": "arrive", "goal": g["name"], **results[-1]})
                    print("arrived", results[-1], flush=True)
                    if ear:
                        ear.speak(f"Listo: {g['name'].replace('_', ' ')}." if ear.es else f"I'm in the {g['name'].replace('_', ' ')}.")
                    break
                ahead, reached = follow(path, est) if path else (None, False)
                if reached or t - last_plan > 2.0:  # a corner reached (plan again from where it really is), or now and then
                    path, last_plan = route(hmap, est[:2], g["xy"], MARGIN) or route(hmap, est[:2], g["xy"], SQUEEZE), t
                    plans += 1
                mode = "go" if path else "stuck"
                if path:
                    if last_plan == t:  # just planned
                        ahead = follow(path, est)[0]
                    e = float(_wrap(math.atan2(ahead[1] - est[1], ahead[0] - est[0]) - est[2]))
                    w = max(-TURN, min(TURN, 2.0 * e))
                    v = VX * max(0.0, math.cos(e)) ** 3
                    if sim.clearance[int((est[1] - hmap.o[1]) / 0.02) + 1, int((est[0] - hmap.o[0]) / 0.02) + 1] < MARGIN + 0.02:
                        v = min(v, 0.08)
            lost += lost_tick(pf, mode)
            if lost > LOST_S and g:
                ok = ask(f"Lost for {LOST_S} s ({conf:.0%} sure, best guess: {room_name(est[0], est[1]) or '?'}, the camera's colours "
                         f"say: {vote or 'no camera'}). Keep looking for the {g['name']}?", t)
                print("asked:", trace[-1]["question"], "->", "yes" if ok else "no: stopping", flush=True)
                lost, stopped = 0.0, not ok
                if stopped:
                    mode, v, w = "stop", 0.0, 0.0
                if stopped and ear:  # listening: drop this plan, wait for the next command
                    results.append({"goal": g["name"], "t": round(time.time() - g0, 1), "declared": False, "why": "declined"})
                    stopped, todo = False, []
                    break
            think, steps = think + time.perf_counter() - c0, steps + 1
            vx, vyaw = gait(v, w)
            duck.move(vx, vyaw)
            history = (history + [(vx, true)])[-2 * HZ:]  # walking for 2 s and the body hasn't moved: a bump
            bumped = vx > 0 and len(history) == 2 * HZ and all(h[0] > 0 for h in history) and \
                math.hypot(true[0] - history[0][1][0], true[1] - history[0][1][1]) < 0.02
            if bumped:
                bumps += 1
                back, last_plan, goal_bumps, history = 5, -9.0, goal_bumps + 1, []
            trace.append({"t": round(t, 2), "wall": round(tick, 3), "true": [round(v_, 3) for v_ in true],
                          "est": [round(v_, 3) for v_ in est], "conf": round(conf, 2), "mode": mode,
                          "aim": [round(ahead[0], 2), round(ahead[1], 2)] if ahead else None, "cmd": [round(vx, 2), round(vyaw, 2)],
                          "goal": g["name"] if g else "", "room_said": room_name(est[0], est[1]), "room_true": room_name(true[0], true[1]),
                          "seen": [o["cls"] for o in objs] + [f"aruco {i}" for i in seen], "colour": vote, "bump": bumped, "trunk_z": round(trunk_z, 3)})
            with live.open("a") as f:  # the live view's feed
                f.write("".join(json.dumps(e) + "\n" for e in trace[logged:]))
            logged = len(trace)
            time.sleep(max(0.0, 1 / HZ - (time.time() - tick)))
        else:
            if g:
                results.append({"goal": g["name"], "t": round(time.time() - g0, 1), "declared": False,
                                "why": "declined" if stopped else "ended" if bye else "stuck" if goal_bumps >= 12 else "time"})
                print("gave up", results[-1], flush=True)
        if stopped:
            break
    for _ in range(5):
        duck.move(0.0, 0.0)
        time.sleep(0.1)
    sure = [s for s in trace if "conf" in s and s["conf"] >= SURE]
    summary = {"house": house_dir.name, "start": [round(v, 3) for v in start], "goals": [g["name"] for g in went],
               "goal_points": went, "declared": sum(r["declared"] for r in results),
               "reached": sum(r["declared"] and r["true_dist"] < 0.5 for r in results),
               "found_s": round(found, 1) if found is not None else None, "time_s": round(time.time() - t0, 1), "bumps": bumps, "camera": K is not None,
               "vision_decisions": sum(s.get("event") == "vision_decision" for s in trace),
               "approvals": [s["answer"] for s in trace if s.get("event") == "approval"],
               "think_ms_per_step": round(1000 * think / max(1, steps), 2), "plans": plans,
               "room_right_pct": round(100 * np.mean([s["room_said"] == s["room_true"] for s in sure]), 1) if sure else None,
               "err_when_sure_median_m": round(float(np.median([math.hypot(s["est"][0] - s["true"][0], s["est"][1] - s["true"][1]) for s in sure])), 3) if sure else None,
               "fell": any(s["trunk_z"] < 0.08 for s in trace if "trunk_z" in s), "results": results,
               "command": a.command or [e["heard"] for e in trace if e.get("event") == "command"] or None, "model_calls": sum(s.get("event") == "model_decision" for s in trace)}
    if words:
        text, info = language.report(summary, trace, ear.lang if ear else "")
        trace.append(said("report", info, "summary", text) | {"t": round(time.time() - t0, 2), "wall": round(time.time(), 3)})
        (out / "report.md").write_text(text + "\n")
        print(text, flush=True)
        if ear:
            with live.open("a") as f:
                f.write(json.dumps(trace[-1]) + "\n")
            ear.speak(text, wait=True)
    (out / "trace.jsonl").write_text("\n".join(json.dumps(s) for s in trace))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "goal_points"}, indent=1))


if __name__ == "__main__":
    main()
