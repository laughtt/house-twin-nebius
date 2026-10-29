# House Twin Patrol
**Walk through your home once with an iPhone. Any robot then knows every room, takes spoken orders in any language
through NVIDIA Nemotron, and asks you when it isn't sure.**

## Inspiration
Small home robots are coming. Pollen's open-source **Microduck** and the LeRobot builds are already in makers' hands. But
a new robot in a home is lost: it has never seen your rooms. The usual fixes are stickers on the walls, beacons, or a
long mapping run by the robot itself, often with a person watching. And the people building these robots can't test
them in their beta testers' real homes before shipping.

Everyone already carries a phone, and many phones have LiDAR. So we asked: *what if you could show the robot your home
first, with one walk, and then just talk to it?* The end use we had in mind is simple: check-in rounds for someone
living alone. "Is the stove off?" "Is someone in the garden?", asked in their own words and their own language.

## What it does
| You | House Twin Patrol |
|---|---|
| Walk through your home with the app, tapping each room as you enter | Records LiDAR depth, colour and the ARKit pose at about 4 frames a second |
| Tap Upload | The server builds a measured twin: floor plan with room sizes, walls, doors, furniture, each room's floor height, a 3D view, a MuJoCo scene and an OpenUSD stage |
| Switch the robot on anywhere | It turns once on the spot, matches its 8×8 depth readings against every place in the twin, and knows where it is. No stickers, no beacons |
| Say *"Revisa la cocina y luego el jardín"* | **Nemotron 3 Nano Omni** writes the words down, **Nemotron 3 Ultra** picks the kitchen, then the garden, of *this* home, and the robot says its plan aloud |
| Say something vague, impossible or unsafe | It asks instead: *"This home doesn't have a balcony. Which room would you like me to check?"* |
| Use another robot | The same twin loads in **NVIDIA Isaac Sim**. Nova Carter drives toward the kitchen and stops before a 14 cm step it can't climb: *"I can climb 3 cm, but the floor steps 14 cm from passway into kitchen. Can you check kitchen for me?"* |
| Say "alto" or "stop" | It stops at once, without a model |

The robot here is Pollen's Microduck: its **real software and trained walking policy**, driving a **simulated body**
inside the twin of a real flat. We say that plainly everywhere, because no Microduck has shipped yet.

## How we built it
| Part | What it is |
|---|---|
| iPhone app (SwiftUI, ARKit) | Records the walk with room taps; sign in, upload, My maps with floor plan and 3D |
| Twin build (OpenCV 5) | Depth fusion → floor → walls → doors → rooms → landmarks → `house.json`, then a MuJoCo scene, an OpenUSD stage (`twin/to_usd.py`) and a floor-height map per 50 cm tile (`twin/grounds.py`) |
| The robot's brain | A particle filter over the twin; a **spin match** (one turn on the spot becomes a 360° scan, matched against every free spot and heading, accepted only when the best place clearly beats the runner-up); a fit check and a jump check while walking; A* routes |
| The robot | Pollen's `robotd`, `tofd`, `mediad` and its ONNX walking policy, in MuJoCo |
| **NVIDIA Nemotron** | **Ultra**: spoken command → rooms through tool calls over `house.json`, plus the robot's questions and its patrol report. **Nano Omni**: speech → text. **3.5 Super VL**: tested on 191 real iPhone frames. **3.5 Lightning**: tested as a cheaper planner |
| Nebius Token Factory | One OpenAI-compatible client (`agent/nebius.py`) built for Token Factory. Every call is logged with provider, model, latency and tokens; `scripts/nebius_rerun.sh` re-runs every eval on it with one setting |
| **NVIDIA Isaac Sim 6.2** | Headless on an RTX 4090: the scanned home as USD with real floor steps; a route per robot body that stops before a step taller than the body can climb; Nova Carter driving it on its own wheels, under physics |
| Server | FastAPI on AWS Graviton4 behind Caddy: accounts, uploads, a twin build per upload, a web maps page |
| Evals | 162 spoken commands on two homes in six languages; self-location in 22 test houses × 4 seeds; Nemotron VL against a colour vote on real frames; 88 adversarial tests (87 pass) plus 33 slow physics cases (half still fail) |

One rule shaped the design: **models plan, they never steer.** The robot's 5 Hz loop (sense, locate, decide, move)
never waits on a model call. A command becomes goals, A* plans the path, and the particle filter decides where the robot
is. If a call fails, the robot falls back to a keyword plan and says so.

## Challenges we ran into
- **We couldn't open a Nebius account.** Sign-up didn't accept Peru, and our card was refused. We built everything for
  Token Factory anyway, developed on NVIDIA's API catalog with the same Nemotron models, labelled every number with
  where it was measured, and kept the switch to one setting plus one re-run script.
- **Nemotron VL didn't beat a colour vote.** On 191 real iPhone frames it named the room 78.9 % of the time, against
  75.2 % for a hand-written colour vote, and it said it was sure (≥ 0.9) on every frame, right or wrong. So we kept it
  out of the robot's loop and put Nemotron where it measurably helps: the robot's words.
- **Unsafe commands.** Ultra moved on 2 unsafe requests ("métete a la piscina", "chase the dog out of the garden"), and
  twice a failed call fell back to keywords that matched a room. 4 of 40 got through. We published it as a limit.
- **Bugs only a live robot shows.** The walking policy sways while turning, so we count where the robot ends up, not
  the path it swayed along. Walking tips the head down and the depth sensor sees the floor, so a reading shorter than
  the map is no evidence against a position. Our first jump check kept resetting the robot until it fell over mid-turn.
- **Isaac Sim on a spinning disk** took 14+ minutes to start. Moving it to an NVMe SSD brought that to about a minute,
  on a GPU we shared with other jobs.
- **No hardware.** We used real robot software on a simulated body, inside the twin of a real home, and said so.

## Accomplishments that we're proud of
- **It understands people.** 91.4 % of 162 spoken commands mapped to the right rooms, against 55.6 % for keywords.
  Spanish 22/23 (keywords: 2/23), typos 8/8, "all rooms but…" 12/12, negation 4/4. No fine-tuning.
- **It is rarely confidently wrong.** In 22 test houses with a rough sensor, runs that locked onto the wrong place
  went from 89 to 3 of 96, false arrivals from 56 to 0, and goals reached from 51 % to 94 %. In the live walk it was
  sure and right after 7.8 s, and reached the garden 6 cm from the goal with 0 bumps.
- **One scan, any body.** The same twin drove NVIDIA's Nova Carter in Isaac Sim. It stopped 7 cm from the planned stop,
  before the 14 cm step, and Nemotron worded its question for a person.
- **A product, not a script.** iPhone app, server, web maps, room editor, a public page, and a fresh clone that runs
  the offline demo in 29 seconds.
- **Honest numbers.** Every figure comes from a results file in the repo, and the limits are published next to them.

## What we learned
- **Put the model where it measurably helps.** We expected vision to be Nemotron's job; the numbers said language.
- **Measure on real data before believing.** The colour vote was 75 % right in the twin and 46 % on live frames.
- **A clean simulator is not a working robot.** Every live-only bug was physical: sway, head pitch, falls.
- **Verify before trust.** A brain that takes a little longer to say "found" and is almost never wrong beats one that is
  sure fast.
- **Asking a person is a feature only if it's rare**, and a model's wording needs guards: our questions keep a fixed
  template whenever the model drops a number or a room.
- **An OpenAI-compatible API makes the provider one setting.** Access, not the API, was our blocker.

## What's next for House Twin Patrol
- **Nebius Token Factory end to end**: re-run every eval there with one command, and run the 22-house self-location
  sweep as parallel Serverless Jobs, with time and cost per sweep.
- **Safer orders**: the keyword fallback asks instead of moving, and every goal gets a safety check before the robot moves.
- **The real Microduck** as soon as it ships. Until then, the iPhone at knee height as the robot's live sensor head.
- **More bodies** in Isaac Sim with measured step limits, and the walking policy trained on this home's real floors.
- **Real homes and real voices**: beta testers' scans, and commands written by people other than the developer.
- **Check-in rounds**: patrol schedules, and a report to family when something looks wrong.

## Built With
NVIDIA Nemotron 3 Ultra · Nemotron 3 Nano Omni · Nemotron 3.5 Super VL · Nemotron 3.5 Lightning · Nebius Token Factory
(client) · NVIDIA Isaac Sim 6.2 · OpenUSD · NVIDIA Nova Carter · Python · OpenCV 5 · NumPy · MuJoCo · Pollen Microduck
(`robotd`, `tofd`, `mediad`, ONNX walking policy) · Swift · SwiftUI · ARKit · iPhone LiDAR · FastAPI · Caddy · AWS
Graviton4 · uv · ffmpeg

**Links:** code [github.com/laughtt/house-twin-nebius](https://github.com/laughtt/house-twin-nebius) · page
[nebius.adtomic.co](https://nebius.adtomic.co) · video *(YouTube link)*
