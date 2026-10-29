"""A spoken command → a route for one robot body in the scanned home, cut where the floor steps more than it can climb.

uv run --python 3.12 --with-requirements requirements.lock --with matplotlib python -m sim_isaac.plan_route \\
    out/houses/<name> "check the kitchen" --robot carter --start 14.4,1.0
→ <house>/isaac/route.json: {command, plan, robot, start, path, stop: {at, step, limit, rooms, ask} | null}
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Poly

from agent import language
from agent.localize import HouseMap
from agent.loop import route
from twin.grounds import grounds

ROBOTS = {"carter": {"radius": 0.30, "max_step": 0.03},  # NVIDIA Nova Carter: wheeled; step limit assumed, not measured
          "jetbot": {"radius": 0.10, "max_step": 0.01}, "duck": {"radius": 0.12, "max_step": 0.015}}


def goal(room):
    a = max(room["rects"], key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
    return ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)


def cut(path, rooms, levels, limit):
    """Walk the path in 5 cm steps; the first time the room under it changes level by more than the limit, stop 40 cm before."""
    polys = [(r["name"], Poly(r["polygon"])) for r in rooms]
    walked, last = [path[0]], None
    for a, b in zip(path, path[1:]):
        n = max(1, int(math.dist(a, b) / 0.05))
        for t in np.linspace(0, 1, n + 1)[1:]:
            p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            here = next((name for name, poly in polys if poly.contains_point(p)), None)
            if here and last and abs(levels[here] - levels[last]) > limit:
                back = [q for q in walked if math.dist(q, p) > 0.4] or walked[:1]
                return back, {"at": [round(v, 3) for v in back[-1]], "step": round(abs(levels[here] - levels[last]), 3),
                                   "limit": limit, "rooms": [last, here]}
            last = here or last
            walked.append(p)
    return path, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="sim_isaac.plan_route")
    ap.add_argument("house", type=Path)
    ap.add_argument("command")
    ap.add_argument("--robot", default="carter", choices=ROBOTS)
    ap.add_argument("--start", default="14.4,1.0")
    a = ap.parse_args()
    house = json.loads((a.house / "house.json").read_text())
    levels = {r["name"]: r["level"] for r in grounds(a.house)[1]["rooms"]}
    plan = language.plan(a.command, house)
    print("plan", plan)
    robot, start = ROBOTS[a.robot], tuple(map(float, a.start.split(",")))
    path, stop, hmap = [start], None, HouseMap(house)
    for name in plan["goals"]:
        leg = route(hmap, path[-1], goal(next(r for r in house["rooms"] if r["name"] == name)), radius=robot["radius"])
        if leg is None:
            stop = {"at": list(path[-1]), "no_route": name}
            break
        leg, stop = cut([path[-1]] + leg, house["rooms"], levels, robot["max_step"])
        path += leg[1:]
        if stop:
            break
    if stop and "step" in stop:
        a_, b_ = stop["rooms"]
        stop["ask"], _ = language.question(f"The floor steps {stop['step'] * 100:.0f} cm from {a_} into {b_} and I can climb "
                                           f"{stop['limit'] * 100:.0f} cm. Can you check {b_} for me?")
    out = {"command": a.command, "plan": plan, "robot": a.robot, "start": list(start), "levels": levels,
           "path": [[round(x, 3), round(y, 3)] for x, y in path], "stop": stop}
    (a.house / "isaac").mkdir(exist_ok=True)
    (a.house / "isaac/route.json").write_text(json.dumps(out, indent=1))
    print(len(path), "points; stop:", stop)
