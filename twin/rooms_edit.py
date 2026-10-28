"""Name and split a twin's rooms by hand, in the browser (the big open space of a flat → kitchen, dining, office ...).

python -m twin.rooms_edit out/houses/<name>                      → <house>/rooms_edit.html: draw, rename, download rooms.json
python -m twin.rooms_edit out/houses/<name> --apply rooms.json   → house.json rooms replaced (house.before-rooms.json kept once)

Walls and doors don't change: rooms are what the duck calls places. They set the goals (room centres), the words the
planner knows and, after `python -m twin sim`, each room's floor in duck-sim.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import cv2 as cv
import numpy as np


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_") or "room"


def outline(rects, cell=0.05):
    """The outline of the rectangles' union (the largest piece), as a polygon in metres."""
    lo = np.min([r[:2] for r in rects], 0) - cell
    hi = np.max([r[2:] for r in rects], 0) + cell
    m = np.zeros((int((hi[1] - lo[1]) / cell) + 2, int((hi[0] - lo[0]) / cell) + 2), np.uint8)
    for x0, y0, x1, y1 in rects:
        m[round((y0 - lo[1]) / cell):round((y1 - lo[1]) / cell), round((x0 - lo[0]) / cell):round((x1 - lo[0]) / cell)] = 1
    c = max(cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0], key=cv.contourArea)[:, 0]
    return [[round(float(lo[0] + x * cell), 3), round(float(lo[1] + y * cell), 3)] for x, y in c]


def editor(house_dir):
    h = json.loads((house_dir / "house.json").read_text())
    data = {"house": house_dir.name, "walls": [[w["a"], w["b"]] for w in h["walls"]],
            "doors": [{"c": d["center"], "w": d["width"], "axis": d["axis"]} for d in h.get("doors", [])],
            "objects": [{"label": o["label"], "c": o["center"][:2], "s": o["size"][:2], "yaw": o.get("yaw", 0)} for o in h.get("objects", [])],
            "landmarks": [{"label": lm["label"], "c": lm["center"][:2]} for lm in h.get("landmarks", [])],
            "rooms": [{"name": r["name"], "label": r.get("label", r["name"]), "rects": r["rects"]} for r in h["rooms"]]}
    page = (Path(__file__).parent / "rooms_edit.html").read_text().replace("/*DATA*/null", json.dumps(data))
    (house_dir / "rooms_edit.html").write_text(page)
    return house_dir / "rooms_edit.html"


def apply(house_dir, rooms_file):
    path = house_dir / "house.json"
    h = json.loads(path.read_text())
    if not (house_dir / "house.before-rooms.json").exists():
        shutil.copy(path, house_dir / "house.before-rooms.json")
    old, rooms, seen = {r["name"]: r for r in h["rooms"]}, [], set()
    for r in json.loads(Path(rooms_file).read_text())["rooms"]:
        rects = [[round(float(v), 3) for v in q] for q in r.get("rects", []) if len(q) == 4 and (q[2] - q[0]) * (q[3] - q[1]) >= 0.01]
        name = slug(r.get("name") or r.get("label"))
        if not rects or name in seen:
            continue
        seen.add(name)
        keep = old.get(name, {})
        rooms.append({"name": name, "label": str(r.get("label") or name)[:40], "space": keep.get("space", name), "polygon": outline(rects),
                      "rects": rects, **({"colour_hs": keep["colour_hs"]} if "colour_hs" in keep else {})})
    h["rooms"] = rooms
    path.write_text(json.dumps(h, indent=1))
    return rooms


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="twin.rooms_edit")
    ap.add_argument("house", help="out/houses/<name>")
    ap.add_argument("--apply", help="the rooms.json downloaded from the editor")
    a = ap.parse_args()
    house_dir = Path(a.house)
    if a.apply:
        rooms = apply(house_dir, a.apply)
        print(f"{len(rooms)} rooms: " + ", ".join(f"{r['label']} ({r['name']})" for r in rooms))
        frame = house_dir / "sim_frame.json"
        spawn = ",".join(f"{v:g}" for v in json.loads(frame.read_text())["spawn"]) if frame.exists() else "x,y,heading"
        print(f"floors in duck-sim: python -m twin sim {house_dir / 'house.json'} --spawn {spawn} --no-markers")
    else:
        print(editor(house_dir))
