"""The house as training ground for the walking policy: floor height per 50 cm tile, each room's level and slope, the
steps between rooms, door sills and the pool, in one page with a 3D view; pick each room's surface, export grounds.json.

python -m twin.grounds out/houses/<name>   → <house>/grounds.html

The duck lifts its feet 1–2 cm (Pollen's rough terrain caps steps at 1.5 cm): a bigger step is a wall, not training.
Tile heights carry a few cm of ARKit drift and LiDAR noise, so room levels are medians and only steps over 5 cm count.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Poly

STEP = 0.05  # a level change between rooms bigger than the noise


def grounds(house_dir):
    h = json.loads((house_dir / "house.json").read_text())
    z = np.load(house_dir / "occupancy.npz")
    g, lo, t = z["ground"].astype(float), z["ground_lo"], float(z["ground_tile"])
    g -= np.nanmedian(g)
    ys, xs = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    centres = np.c_[(xs.ravel() + .5) * t + lo[0], (ys.ravel() + .5) * t + lo[1]]
    owner = np.full(g.shape, -1)
    for i, r in enumerate(h["rooms"]):
        owner[Poly(r["polygon"]).contains_points(centres).reshape(g.shape) & (owner < 0)] = i
    rooms = []
    for i, r in enumerate(h["rooms"]):
        v = g[(owner == i) & np.isfinite(g)]
        rooms.append({"name": r["name"], "label": r.get("label", r["name"]), "polygon": r["polygon"], "tiles": int(v.size),
                      "level": round(float(np.median(v)), 3), "spread": round(float(np.percentile(v, 90) - np.percentile(v, 10)), 3)})
    shown = np.where(owner >= 0, g, np.nan)  # tiles outside every room are filled-in guesses: not drawn
    for _ in range(2):  # rooms grow into the wall gaps between them (walls are 10–20 cm, tiles 50 cm), so neighbours touch
        grown = owner.copy()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            src = np.roll(owner, (dy, dx), (0, 1))
            grown[(grown < 0) & (src >= 0)] = src[(grown < 0) & (src >= 0)]
        owner = grown
    steps = []  # tile edges between two rooms whose levels differ by more than STEP
    for dy, dx in ((0, 1), (1, 0)):
        a, b = owner[:g.shape[0] - dy, :g.shape[1] - dx], owner[dy:, dx:]
        for y, x in zip(*np.nonzero((a >= 0) & (b >= 0) & (a != b))):
            dh = rooms[b[y, x]]["level"] - rooms[a[y, x]]["level"]
            if abs(dh) > STEP:
                x0, y0 = lo[0] + (x + dx) * t, lo[1] + (y + dy) * t
                steps.append({"a": [round(x0, 3), round(y0, 3)], "b": [round(x0 + dy * t, 3), round(y0 + dx * t, 3)],
                              "h": round(abs(dh), 3), "rooms": sorted([rooms[a[y, x]]["name"], rooms[b[y, x]]["name"]])})
    data = {"house": house_dir.name, "tile": t, "lo": [float(lo[0]), float(lo[1])],
            "ground": [[None if np.isnan(v) else round(float(v), 3) for v in row] for row in shown], "rooms": rooms, "steps": steps,
            "walls": [[w["a"], w["b"]] for w in h["walls"]],
            "doors": [{"c": d["center"], "w": d["width"], "axis": d["axis"], "sill": d.get("threshold_h", 0.015)} for d in h.get("doors", [])],
            "objects": [{"label": o["label"], "c": o["center"][:2], "s": o["size"], "yaw": o.get("yaw", 0)} for o in h.get("objects", [])]}
    page = (Path(__file__).parent / "grounds.html").read_text().replace("/*DATA*/null", json.dumps(data))
    (house_dir / "grounds.html").write_text(page)
    return house_dir / "grounds.html", data, owner


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="twin.grounds")
    ap.add_argument("house", type=Path)
    path, d, _ = grounds(ap.parse_args().house)
    print(path)
    for r in d["rooms"]:
        print(f"  {r['name']:12} level {r['level']:+.3f} m  spread {r['spread']:.3f} m  tiles {r['tiles']}")
    pairs = {}
    for s in d["steps"]:
        pairs.setdefault(tuple(s["rooms"]), s["h"])
    for (a, b), hh in pairs.items():
        print(f"  step {a} ↔ {b}: {hh * 100:.0f} cm")
