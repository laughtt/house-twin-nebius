"""An iPhone LiDAR scan → a Nerfstudio dataset for a Gaussian splat trainer (Brush, gsplat, nerfstudio), in the house
frame of the twin built from the same scan (z up, floor at 0, house.json's axes and origin), so the splat lines up with
house.json and the sim.

python -m twin.splat data/lidar/<scan>.zip --house out/houses/<scan> [--out out/splat/<scan>]
→ images/*.jpg, transforms.json (every frame's pose and intrinsics), init.ply (LiDAR points, coloured: the starting splats)
"""
import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from twin.scan_io import FLIP, M, Export, kmat


def house_from_arkit(scan_frame):
    """4x4: ARKit world (y up) → house frame. Rigid: the twin's per-tile floor correction (a few cm) is left out."""
    t = scan_frame["theta"]
    A = np.eye(4)
    A[:3, :3] = np.array([[np.cos(t), np.sin(t), 0], [-np.sin(t), np.cos(t), 0], [0, 0, 1]]) @ M
    A[:3, 3] = [-scan_frame["origin"][0], -scan_frame["origin"][1], -scan_frame["z0"]]
    return A


def main():
    ap = argparse.ArgumentParser(prog="twin.splat")
    ap.add_argument("scan")
    ap.add_argument("--house", required=True, help="the twin built from this scan (its report.json has the frame)")
    ap.add_argument("--out")
    ap.add_argument("--drop", type=float, default=0.2, help="share of the blurriest frames left out")
    a = ap.parse_args()
    ex = Export(a.scan)
    out = Path(a.out or f"out/splat/{ex.session}")
    (out / "images").mkdir(parents=True, exist_ok=True)
    A = house_from_arkit(json.loads((Path(a.house) / "report.json").read_text())["scan_frame"])
    frames = [f for f in ex.frames if f.get("tracking", "normal") == "normal"]
    sharp = {f["i"]: cv.Laplacian(cv.cvtColor(cv.resize(ex.rgb(f), (320, 240)), cv.COLOR_BGR2GRAY), cv.CV_32F).var() for f in frames}
    cut = np.quantile(list(sharp.values()), a.drop)
    keep = [f for f in frames if sharp[f["i"]] >= cut]
    rows, pts, cols = [], [], []
    for n, f in enumerate(keep):
        name = f"images/{f['i']:06d}.jpg"
        (out / name).write_bytes(ex.read(f["rgb"]))
        w, h = f["rgb_wh"]
        rows.append({"file_path": name, "transform_matrix": (A @ np.array(f["T"]).reshape(4, 4)).tolist(),
                     "fl_x": f["K"][0], "fl_y": f["K"][1], "cx": f["K"][2], "cy": f["K"][3], "w": w, "h": h})
        if n % 4 == 0:  # the starting points: high-confidence depth, every 4th pixel, coloured from the photo
            d, _ = ex.load(f)
            v, u = np.nonzero(d[::4, ::4])
            v, u = v * 4, u * 4
            p = np.linalg.inv(kmat(f["K_depth"])) @ np.stack([u, v, np.ones_like(u)]).astype(float) * d[v, u]
            T = np.array(f["T"]).reshape(4, 4)
            pts.append((A[:3, :3] @ (T[:3, :3] @ FLIP @ p + T[:3, 3:4]) + A[:3, 3:4]).T)
            s = w / d.shape[1]
            cols.append(ex.rgb(f)[np.minimum((v * s).astype(int), h - 1), np.minimum((u * s).astype(int), w - 1)][:, ::-1])
    (out / "transforms.json").write_text(json.dumps({"camera_model": "OPENCV", "frames": rows}, indent=1))
    p, c = np.concatenate(pts), np.concatenate(cols)
    key = np.floor(p / 0.02).astype(np.int64)  # one point per 2 cm voxel (the mean)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    p = np.stack([np.bincount(inv, p[:, k]) for k in range(3)], 1) / cnt[:, None]
    c = (np.stack([np.bincount(inv, c[:, k].astype(float)) for k in range(3)], 1) / cnt[:, None]).astype(np.uint8)
    v = np.empty(len(p), [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"], v["red"], v["green"], v["blue"] = *p.T, *c.T
    with open(out / "init.ply", "wb") as fh:
        fh.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode())
        fh.write(v.tobytes())
    print(json.dumps({"frames": len(ex.frames), "kept": len(keep), "init_points": len(v), "out": str(out),
                      "extent_m": np.round(p.max(0) - p.min(0), 1).tolist()}))


if __name__ == "__main__":
    main()
