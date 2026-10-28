# House Twin Patrol (Nebius × NVIDIA pass)

One iPhone LiDAR walk of a home becomes a digital twin (floor plan, rooms, doors, 3D, MuJoCo scene). A small home robot
(Pollen's Microduck: real `robotd` + walking policy, simulated body) finds itself in that twin with no markers, using an
8×8 ToF and its camera, patrols rooms, and asks a person when unsure. This repo is the entry for the Nebius × NVIDIA
Global AI Hackathon, Physical AI track. It started as a copy of the OpenCV AI Competition 2026 entry (first commit).

@.claude/HANDOFF.md

## Layout
| path | what |
|---|---|
| `app/` | iPhone app (SwiftUI, ARKit): record RGB + LiDAR + pose with room taps, login, upload, My maps. `Recorder.swift` packs frames |
| `lidar/pipeline`, `twin/` | the twin build (OpenCV 5): fuse → floor → walls → doors → rooms → landmarks → `house.json`, MuJoCo scene (`to_mjcf.py`) |
| `twin/curate.py` | landmark check by a model (backends in `config.toml [curate]`) |
| `agent/` | the robot's brain: `localize.py` (particle filter), `loc/` (spin match, fit and jump checks), `perceive.py` (camera: HSV room vote), `walk.py` (live loop on Pollen's duck-sim, questions, `vision_decision` trace events), `loop.py` (A*), `film.py` |
| `eval/` | `selfloc.py` (22 test houses), `vision_walk.py` (camera on/off in the twin), `lowview_test.py` (real duck-height frames from a scan), adversarial fast tests |
| `service/` | FastAPI server (login, upload, twin build per upload); `v0/` room-measure agent |
| `infra/` | AWS deploy (Caddy, systemd, Bedrock role) |
| `scripts/demo.sh` | `offline | live | server | app` |
| `samples/` | a built sample home and a live walk trace, for testing without a phone |

Private, git-ignored: `data/` (real home scans: `data/lidar/*.zip`), `out/` (built houses, walks, results), `service/models/`, `.env`, `.claude/`.

## Run
- Python deps: `requirements.lock`; scripts use `uv run --python 3.12 --with opencv-python-headless==5.0.0.93 ...` (see each file's docstring).
- Fast check: `scripts/demo.sh offline`. Details: `DEMO.md`, `TESTING.md`.

## Conventions
- Less code is better. Match the existing style: short modules, plain names, few comments, plain-word docs.
- Before a change, write `.claude/YYYY-MM-DD-<topic>.md` (what, flow, files, trade-offs).
- Every model call goes to Nebius Token Factory (OpenAI-compatible) with an NVIDIA model. No Bedrock/Gemini on the default path.
- Model calls never block the 5 Hz robot loop: one call in flight, stale frames dropped, the loop uses the last answer.
- Numbers in docs come from a results file in `out/`; one set of numbers everywhere.
- Don't commit or push unless asked. Never commit `.env`, `data/`, `out/` or `.claude/`.
