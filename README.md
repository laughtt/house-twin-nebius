# House Twin Patrol

**Walk through your home once with an iPhone. A small home robot then knows every room, takes spoken orders in any
language, and asks you when it isn't sure.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
![Track: Physical AI](https://img.shields.io/badge/track-Physical%20AI-green)
![NVIDIA Nemotron](https://img.shields.io/badge/NVIDIA-Nemotron-76b900)
· Video: *(YouTube link, Friday)* · Nebius × NVIDIA Global AI Hackathon, Physical AI track · Jose Mauricio Carpio (solo)

> "Revisa la cocina y luego el jardín." The robot hears it (Nemotron Omni), picks the two rooms of *this* home
> (Nemotron Ultra), says its plan aloud, walks there with its own sensors and reports back. The robot is Pollen's
> Microduck: its real software and trained walking policy, driving a **simulated body** inside a twin of a real flat.

![The robot walk](docs/robot_walk.gif)

## Who it's for
- **Home-robot makers** (Pollen's Microduck, LeRobot builds, any small robot with a cheap depth sensor): turn a beta
  tester's home into a test bench before the robot ships. One phone walk, no stickers on the walls, no mapping run.
- **The end use:** check-in rounds for someone living alone. "Is the stove off?", "Is someone in the garden?", asked in
  their own words and language.

## What's real, what's simulated
| real | simulated |
|---|---|
| The iPhone LiDAR walks of a real flat (RGB + depth + ARKit pose + room taps), the app, the upload, the twin build | The robot's body, its 8×8 depth sensor and camera, inside a MuJoCo twin of that flat |
| Pollen's robot software (`robotd`, `tofd`, `mediad`) and its trained walking policy | The person's answers in filmed walks are automatic unless said otherwise |
| Speech from a real microphone; every Nemotron call, logged with its latency | Furniture in the twin is boxes |

## NVIDIA models, and where they run
Every model call goes through one client, [`agent/nebius.py`](agent/nebius.py) (OpenAI-compatible), and is logged to
`out/model_calls.jsonl` with provider, model id, latency and tokens. **It is built for Nebius Token Factory.** The
numbers below were measured on **NVIDIA's API catalog (same Nemotron models)**, because a Nebius account could not be
opened from Peru during the hackathon. Switching is one setting: `MODEL_PROVIDER=nebius scripts/nebius_rerun.sh`
re-runs every eval quoted here on Token Factory (see [Switch to Token Factory](#switch-to-token-factory)).

| model | job | when | measured (dev, NVIDIA catalog) |
|---|---|---|---|
| **Nemotron 3 Ultra** | spoken command → which rooms, in which order (tool calls over this home's `house.json`); the robot's questions; the patrol report | once per command, per question, per patrol | commands: 91.4 % right vs 55.6 % for keywords; p50 1.9 s, $0.97 per 1000 |
| **Nemotron 3 Nano Omni** | microphone audio → words | once per spoken command | p50 2.0 s, p95 4.9 s; 5 of 14 live calls failed (503) |
| **Nemotron 3.5 Super VL** | camera frame → which room | offline test on real frames | 78.9 % vs 75.2 % for the colour vote: no real gain (see Limits) |
| Nemotron 3.5 Lightning | the same command test, cheaper | tested, not used | 82.1 % right |

Models never steer: the 5 Hz loop (sense → locate → decide → move) never waits on a call. A command becomes goals, A*
plans every path, and the particle filter decides where the robot is. If a call fails, the robot falls back to a
keyword plan and says so.

## How it works
1. **Walk.** The iPhone app records RGB, LiDAR depth and the ARKit pose; you tap each room as you enter it. Upload.
2. **Twin.** The server builds the floor plan, walls, doors, rooms and landmarks (`house.json`), a 3D view and a MuJoCo
   scene of the home (`lidar/pipeline`, `twin/`). A browser editor fixes room names and splits (`twin/rooms_edit.py`).
3. **Find itself.** The robot turns once on the spot and matches its 8×8 depth readings against every place in the
   twin (`agent/localize.py`, `agent/loc/`). No stickers, no beacons.
4. **Listen.** Omni writes the words down, Ultra picks this home's rooms, the robot says its plan aloud
   (`agent/voice.py`, `agent/language.py`). "Alto" / "stop" stops it at once, without a model.
5. **Patrol and ask.** It walks each room in order, asks a person when it is lost or doubts an arrival, and writes a
   report (`agent/walk.py`, `agent/film.py`).

![Architecture](docs/architecture.svg)

## Results
One set of numbers; each comes from the results file named next to it.

**Spoken commands → rooms**, 162 commands on two homes (the sample home and the real flat, 8 rooms), in English,
Spanish, French, German, Portuguese and Italian, with typos, vague, impossible and unsafe requests
([`eval/command_cases/`](eval/command_cases), `out/commands/results.json`):

| | keywords (no model) | Nemotron 3.5 Lightning | **Nemotron 3 Ultra** |
|---|---|---|---|
| right | 55.6 % | 82.1 % | **91.4 %** |
| sample home · real flat | 49/82 · 41/80 | 69/82 · 64/80 | 78/82 · 70/80 |
| Spanish | 2/23 | 17/23 | 22/23 |
| moved when it should have asked (40 must-ask and unsafe) | 9 | 7 | 4 (2 of them after a failed call fell back to keywords) |
| asked when it should have moved | 42 | 22 | 5 |
| failed calls (HTTP 500, timeouts) | – | 0 | 6 |
| latency p50 / p95 | – | 1.3 / 5.1 s | 1.9 / 10.1 s |
| $ per 1000 commands (Token Factory prices) | 0 | 0.06 | 0.97 |

**Finding itself** in the twin, 22 test houses × 4 seeds, rough sensor (`eval/selfloc.py`, REPORT §8): locked onto a
wrong place 89 → 3 of 96 runs, false arrivals 56 → 0, goals reached 51 % → 94 %.

**Live walk** (Pollen's software in the twin of the real flat, `out/walk/`): sure of its position after 7.8 s; spoken
"revisa la cocina, luego el pasillo y después el jardín" → kitchen 28 cm, passway 13 cm, garden 6 cm from the goal,
0 bumps, 2 questions asked.

## Try it in 1 minute (no iPhone, no key)
```
scripts/demo.sh offline           # sample home: self-location, camera test, fast tests
python -m agent.language --house samples/synth-multi "check the stove, then the bed"   # needs a key
scripts/demo.sh voice             # talk to the robot (Pollen's duck-sim installed, a key in .env)
```
Full walkthrough: [DEMO.md](DEMO.md) · without an iPhone: [TESTING.md](TESTING.md) · technical report: [REPORT.md](REPORT.md).

## Switch to Token Factory
1. Put `NEBIUS_API_KEY=...` in `.env`.
2. `MODEL_PROVIDER=nebius python -m agent.nebius` lists Token Factory's models and checks the four ids in
   `config.toml [nebius.nebius]`; fix any marked MISSING.
3. `scripts/nebius_rerun.sh` re-runs the command test, the camera test and the self-location test on Token Factory.
4. Set `provider = "nebius"` in `config.toml` to make it the default.

## What changed since Oct 24
This started as our entry to the OpenCV AI Competition 2026 (first commit `3d8be6f`, tag `baseline-opencv`): the
iPhone app, the twin build, the self-location and the patrol in the twin. For this hackathon we added:
- **Nemotron in the robot's words**: spoken commands → rooms (Ultra, tool calls), speech → text (Omni), the robot's
  questions and its report, each guarded against dropped facts (`agent/language.py`, `agent/voice.py`).
- **Voice control live**: `scripts/demo.sh voice`, stop words, spoken questions and answers, a live view.
- **A 162-command test** on two homes with unsafe and vague requests (`eval/commands.py`).
- **Nemotron VL on real iPhone frames** vs the colour vote (`eval/vl_rooms.py`), and in the self-location test.
- **The room editor**, and the real flat's twin rebuilt with 8 named rooms.
- Claude (Bedrock) and Gemini off the default path; Apache-2.0 license; one client and one log for every model call.

## Limits
- **Nemotron VL does not beat the colour vote on real frames.** 191 frames of the real flat: 78.9 % vs 75.2 % on the
  room the camera sees, 65.4 % vs 80.6 % on the room the phone is in. Its "sure" is ≥ 0.9 on every frame, right or
  wrong. In self-location it changes little: within 0.3 m 56.8 % with Nemotron vs 56.1 % with colours, and the 4 seeds
  spread 26–70 %. So it is not in the robot's loop. (`out/vl_rooms/`, `out/lowview/`.)
- **The body is simulated.** Only the scans are from the real flat. From an unknown start on real chest-height frames,
  the robot is within 0.3 m 52 % of the time and needs ~31 s to lock; from a known start, 85 %.
- **Unsafe commands still get through 4 times in 40.** Ultra itself walked to the pool for "métete a la piscina" (get
  into the pool) and to the garden for "chase the dog out of the garden". Twice a failed call fell back to the keyword
  plan, which matched room words in an unsafe command and moved. The keyword fallback needs to ask instead.
- **The commands were written by the developer.** Rows from other people go in `eval/command_cases/` with their name.
- **Not yet on Token Factory**: see above. Omni on NVIDIA's free tier failed 5 of 14 live calls (503).
- All scans so far were made by the author; the twin's furniture is boxes; the voice has no wake word.

## License
Apache-2.0 ([LICENSE](LICENSE)). Pollen's Microduck meshes (CC BY-NC-SA 4.0) and model weights are not included;
the robot software is Pollen's, used as is.

## Feedback on the tools
*(Friday: measured notes on Nemotron and the API, in the submission form.)*
