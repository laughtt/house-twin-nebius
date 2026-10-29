"""Isaac Sim 6.2, headless: the scanned home with NVIDIA's Nova Carter driving a planned route (from plan_route.py).
The LiDAR scan's colours sit on the solid boxes. Frames (one per 3 physics steps, 20 fps): <out>/{top,wide}/rgb_*.png

E:\\isaac\\.venv\\Scripts\\python.exe house_clip.py house.usd route.json out_dir
"""
import json
import math
import sys
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import numpy as np  # noqa: E402  (Kit modules load only after SimulationApp)
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.robot.experimental.wheeled_robots.controllers.differential_controller import DifferentialController  # noqa: E402
from isaacsim.robot.experimental.wheeled_robots.robots.wheeled_robot import WheeledRobot  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import UsdGeom  # noqa: E402

house, route, out = Path(sys.argv[1]).resolve(), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3]).resolve()
omni.usd.get_context().open_stage(str(house))
for _ in range(20):
    app.update()
stage = omni.usd.get_context().get_stage()
path = [tuple(p) for p in route["path"]]
(x0, y0), (x1, y1) = path[0], path[min(5, len(path) - 1)]
yaw0 = math.atan2(y1 - y0, x1 - x0)
UsdGeom.Xform.Define(stage, "/Robots")
carter = WheeledRobot("/Robots/carter", wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
                      usd_path=get_assets_root_path() + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
                      positions=[[x0, y0, max(route["levels"].values()) + 0.05]],  # dropped onto whatever floor is there
                      orientations=[[math.cos(yaw0 / 2), 0, 0, math.sin(yaw0 / 2)]])
drive = DifferentialController(wheel_radius=0.14, wheel_base=0.413, max_linear_speed=0.6, max_angular_speed=1.2)

cams = {"top": rep.create.camera(position=(9.5, 4.5, 30), look_at=(9.5, 4.6, 0)),
        "wide": rep.create.camera(position=(14.3, 0.4, 2.2), look_at=(15.0, 4.8, 0), focal_length=16)}
for name, cam in cams.items():
    w = rep.WriterRegistry.get("BasicWriter")
    w.initialize(output_dir=str(out / name), rgb=True)
    w.attach([rep.create.render_product(cam, (1280, 720))])

omni.timeline.get_timeline_interface().play()
for _ in range(30):
    app.update()
print("dofs", carter.dof_names, flush=True)

todo, frames, log = list(path[1:]), 0, []
while frames < 300:
    pos, quat = (v.numpy()[0] for v in carter.get_world_poses())
    w_, z_ = quat[0], quat[3]
    yaw = 2 * math.atan2(z_, w_)
    while len(todo) > 1 and math.dist(todo[0], pos[:2]) < 0.5:  # pure pursuit, 50 cm ahead
        todo.pop(0)
    gx, gy = todo[0]
    err = (math.atan2(gy - pos[1], gx - pos[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
    left = math.dist(todo[-1], pos[:2])
    v = 0.0 if left < 0.08 else min(0.6, left) * max(0.0, math.cos(err))
    carter.apply_wheel_actions(drive.forward(np.array([v, 1.5 * err])))
    app.update()
    app.update()
    rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
    log.append([round(float(pos[0]), 3), round(float(pos[1]), 3), round(yaw, 3)])
    frames += 1
    if left < 0.08:
        if not route.get("_held"):
            route["_held"] = frames
        if frames - route["_held"] > 40:
            break
rep.orchestrator.wait_until_complete()
(out / "drive.json").write_text(json.dumps({"frames": frames, "poses": log}))
print("done", frames, "frames; end", log[-1], "goal", path[-1], flush=True)
app.close()
