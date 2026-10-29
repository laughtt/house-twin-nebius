#!/bin/bash
# The landing page's media → out/site, then (with "deploy") → nebius.adtomic.co.
set -e
cd "$(dirname "$0")/.."
H=out/houses/lidar-20261024-120616 A=out/site/assets
mkdir -p $A
cp site/index.html out/site/
mp4() { ffmpeg -y -loglevel error -i "$1" -an -vf "scale='min($3,iw)':-2,format=yuv420p" -c:v libx264 -crf 27 -preset slow -movflags +faststart "$A/$2.mp4"; }
ffmpeg -y -loglevel error -i docs/house_3d.gif -an -vf "crop=iw:ih-110:0:60,format=yuv420p" -c:v libx264 -crf 27 -preset slow -movflags +faststart $A/house_3d.mp4
mp4 docs/robot_walk.gif robot_walk 1280
mp4 docs/map_build.gif map_build 1100
mp4 $H/isaac/carter_kitchen.mp4 carter_kitchen 1280
for f in docs/app_screens.jpg $H/isaac/passway.png $H/twin_overview.png; do
  cwebp -quiet -q 80 $f -o $A/$(basename ${f%.*}).webp
done
cwebp -quiet -q 80 -crop 0 60 960 450 docs/house_3d.png -o $A/house_3d.webp
python3 - $H <<'PY'
import json, sys
h, r = json.load(open(sys.argv[1] + "/house.json")), json.load(open(sys.argv[1] + "/isaac/route.json"))
json.dump({"rooms": [{"name": x["name"], "label": x.get("label", x["name"]), "polygon": x["polygon"], "level": r["levels"].get(x["name"])} for x in h["rooms"]],
           "walls": [[*w["a"], *w["b"]] for w in h["walls"]], "doors": [d["center"] + [d["width"], d["axis"]] for d in h["doors"]],
           "objects": [[*o["center"], *o["size"][:2], o["yaw"], o["label"]] for o in h["objects"]],
           "route": {k: r[k] for k in ("command", "plan", "path", "stop")}}, open("out/site/assets/twin.json", "w"))
PY
du -sh out/site/assets/*
if [ "$1" = deploy ]; then
  rsync -az --delete -e "ssh -i ~/.ssh/castle-fight.pem" out/site/ ubuntu@52.87.11.155:/tmp/nebius-site/
  ssh -i ~/.ssh/castle-fight.pem ubuntu@52.87.11.155 'sudo rsync -a --delete /tmp/nebius-site/ /opt/nebius-site/ && sudo chmod -R a+rX /opt/nebius-site'
fi
