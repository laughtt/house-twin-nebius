"""Isaac frames → MP4: the passway camera full frame, the top view as an inset with the robot's track, and captions
(the command, Nemotron's plan, the robot's step limit, its question at the step).

uv run --python 3.12 --with pillow python -m sim_isaac.make_clip <clip dir from the PC> out/houses/<name>/isaac/route.json out.mp4
"""
import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

clip, route, mp4 = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), sys.argv[3]
poses = json.loads((clip / "drive.json").read_text())["poses"]
wide, top = sorted((clip / "wide").glob("rgb_*.png")), sorted((clip / "top").glob("rgb_*.png"))
font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 30)
small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 24)
plan, stop = route["plan"], route["stop"]
lines = [(0, f"“{route['command']}”"),
         (25, f"Nemotron 3 Ultra → {', '.join(plan['goals'])}  ({plan['seconds']:.1f} s)"),
         (50, f"NVIDIA Nova Carter · can climb {stop['limit'] * 100:.0f} cm (assumed)" if stop else "NVIDIA Nova Carter")]
to_px = lambda x, y: (640 + (x - 9.5) * 54, 360 - (y - 4.6) * 54)  # noqa: E731  top camera, 30 m up: ~54 px/m (measured)
with tempfile.TemporaryDirectory() as tmp:
    n = min(len(wide), len(top))
    for i in range(n):
        im = Image.open(wide[i]).convert("RGB")
        inset = Image.open(top[i]).convert("RGB")
        d = ImageDraw.Draw(inset)
        d.line([to_px(*p) for p in route["path"]], fill=(60, 120, 255), width=4)
        d.line([to_px(x, y) for x, y, _ in poses[:i + 1]], fill=(255, 210, 0), width=6)
        im.paste(inset.crop((100, 60, 1180, 660)).resize((430, 239)), (830, 461))
        d = ImageDraw.Draw(im)
        shown = [t for f, t in lines if i >= f]
        d.rectangle((0, 0, 1280, 18 + 40 * len(shown)), fill=(0, 0, 0))
        for k, t in enumerate(shown):
            d.text((20, 12 + 40 * k), t, font=font if k == 0 else small, fill=(255, 255, 255))
        if stop and i >= n - 40:
            d.rectangle((0, 610, 820, 720), fill=(170, 30, 30))
            d.text((20, 618), f"Step {stop['step'] * 100:.0f} cm > {stop['limit'] * 100:.0f} cm: stops and asks", font=small, fill="white")
            d.multiline_text((20, 650), textwrap.fill(f"“{stop['ask']}”", 62), font=small, fill="white", spacing=4)
        im.save(f"{tmp}/f_{i:04d}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "20", "-i", f"{tmp}/f_%04d.png", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", mp4], check=True)
print(mp4, n, "frames")
