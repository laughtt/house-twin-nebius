"""house.json → house.usd for Isaac Sim: walls, each room's floor at its measured level (so the steps are real), furniture,
a collider on every solid, Z up, metres. Rooms are invisible named prims (/House/Rooms/<name>) with their label and polygon.

python -m twin.to_usd out/houses/<name>   → <house>/house.usd (binary)   (needs usd-core: uv run --with usd-core ...)

Floor levels come from twin.grounds (median per room; tiles carry a few cm of ARKit drift). Wall gaps between rooms are
filled at the lower side's level, so a robot meets a step only where two rooms really differ, never a trench.
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Poly
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

from twin.grounds import grounds
from twin.to_mjcf import color

FLOOR = 0.10  # slab thickness under the lowest floor
WALL = (0.93, 0.92, 0.89)
SURFACE = [(0.76, 0.60, 0.36), (0.80, 0.82, 0.76), (0.55, 0.52, 0.50), (0.72, 0.64, 0.82)]


def box(stage, path, centre, size, rgb, yaw=0.0):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*centre))
    c.AddRotateZOp().Set(math.degrees(yaw))
    c.AddScaleOp().Set(Gf.Vec3f(*size))
    c.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return c


def name(s):
    return re.sub(r"[^A-Za-z0-9_]", "_", s) or "_"


def scan(stage, ply, floor):
    """The LiDAR walk's coloured points, for looks only (no collider). Points over 2 m are the ceiling: dropped."""
    with open(ply) as f:
        while not f.readline().startswith("end_header"):
            pass
        p = np.loadtxt(f, usecols=range(6), dtype=np.float32)
    p[:, 2] -= floor
    p = p[p[:, 2] < 2.0]
    pts = UsdGeom.Points.Define(stage, "/House/Scan")
    pts.CreatePointsAttr(p[:, :3])
    pts.CreateWidthsAttr(np.full(len(p), 0.035, np.float32))
    pts.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set((p[:, 3:6] / 255) ** 2.2)  # sRGB → linear


def to_usd(house_dir):
    h = json.loads((house_dir / "house.json").read_text())
    _, g, owner = grounds(house_dir)
    levels = {r["name"]: r["level"] for r in g["rooms"]}
    base = min(levels.values())
    stage = Usd.Stage.CreateNew(str(house_dir / "house.usd"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/House").GetPrim())
    UsdPhysics.Scene.Define(stage, "/House/Physics").CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    UsdLux.DomeLight.Define(stage, "/House/Sky").CreateIntensityAttr(250)
    UsdLux.DistantLight.Define(stage, "/House/Sun").CreateIntensityAttr(600)

    xs = [p for w in h["walls"] for p in (w["a"][0], w["b"][0])]
    ys = [p for w in h["walls"] for p in (w["a"][1], w["b"][1])]
    (x0, x1), (y0, y1) = (min(xs), max(xs)), (min(ys), max(ys))
    box(stage, "/House/Floor/base", ((x0 + x1) / 2, (y0 + y1) / 2, base - FLOOR / 2), (x1 - x0, y1 - y0, FLOOR), (0.6, 0.6, 0.58))
    for i, r in enumerate(h["rooms"]):
        top = levels[r["name"]] - base
        rects = r.get("rects") or [[min(p[0] for p in r["polygon"]), min(p[1] for p in r["polygon"]),
                                    max(p[0] for p in r["polygon"]), max(p[1] for p in r["polygon"])]]
        for k, (a, b, c, d) in enumerate(rects):
            if top > 0.005:  # the base slab already is the lowest room's floor
                box(stage, f"/House/Floor/{name(r['name'])}_{k}", ((a + c) / 2, (b + d) / 2, base + top / 2), (c - a, d - b, top), SURFACE[i % 4])
            else:
                box(stage, f"/House/Floor/{name(r['name'])}_{k}", ((a + c) / 2, (b + d) / 2, base + 0.0005), (c - a, d - b, 0.001), SURFACE[i % 4])
        room = UsdGeom.Xform.Define(stage, f"/House/Rooms/{name(r['name'])}").GetPrim()
        room.SetCustomDataByKey("label", r.get("label", r["name"]))
        room.SetCustomDataByKey("level", levels[r["name"]])
        room.CreateAttribute("polygon", Sdf.ValueTypeNames.Float2Array).Set([Gf.Vec2f(*p) for p in r["polygon"]])

    t, lo, lv = g["tile"], g["lo"], [levels[r["name"]] for r in h["rooms"]]
    for y, x in zip(*((owner >= 0) & np.array([[v is None for v in row] for row in g["ground"]])).nonzero()):
        near = owner[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2]  # a wall gap between rooms: filled at the lower side's level
        top = min(lv[k] for k in near[near >= 0]) - base
        if top > 0.005:
            box(stage, f"/House/Floor/gap_{y}_{x}", (lo[0] + (x + .5) * t, lo[1] + (y + .5) * t, base + top / 2), (t, t, top), (0.6, 0.6, 0.58))

    for j, w in enumerate(h["walls"]):  # the wall line is the measured surface: thickness goes outward, as in to_mjcf
        (ax, ay), (bx, by) = w["a"], w["b"]
        n = math.hypot(bx - ax, by - ay)
        ox, oy = (ay - by) / n, (bx - ax) / n
        if ox * ((ax + bx) / 2 - (x0 + x1) / 2) + oy * ((ay + by) / 2 - (y0 + y1) / 2) < 0:
            ox, oy = -ox, -oy
        push = 0 if w.get("shared") else w["t"] / 2
        box(stage, f"/House/Walls/w{j}", ((ax + bx) / 2 + ox * push, (ay + by) / 2 + oy * push, base + w["h"] / 2),
            (n + (0 if w.get("shared") else 2 * w["t"]), w["t"], w["h"]), WALL, math.atan2(by - ay, bx - ax))
    for j, o in enumerate(h["objects"]):
        (cx, cy), (sw, sd, top) = o["center"], o["size"]
        rgb = tuple(float(v) for v in color(o["label"]).split()[:3])
        z = next((levels[r["name"]] for r in h["rooms"] if Poly(r["polygon"]).contains_point((cx, cy))), base)
        box(stage, f"/House/Objects/{name(o['label'])}_{j}", (cx, cy, z + top / 2), (sw, sd, top), rgb, o.get("yaw", 0.0))
    if (house_dir / "cloud.ply").exists():
        scan(stage, house_dir / "cloud.ply", float(np.nanmedian(np.load(house_dir / "occupancy.npz")["ground"])))
    stage.GetRootLayer().Save()
    return house_dir / "house.usd", levels


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="twin.to_usd")
    ap.add_argument("house", type=Path)
    path, levels = to_usd(ap.parse_args().house)
    print(path, {k: round(v * 100) for k, v in levels.items()}, "cm")
