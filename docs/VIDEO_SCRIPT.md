# 3-minute video script

Nebius × NVIDIA Global AI Hackathon, Physical AI track. About 390 words of voice-over (~2:50 at 140 words a minute).
Rules to meet: ≤ 3 min, public YouTube, **≥ 1 min of real hardware** (here: the iPhone scan walk, the app, the
microphone driving the robot), the payoff within 30 s, the models named aloud. Every number is from a results file
(README "Results").

Clips:
- voice walk: `out/walk/lidar-20261024-120616/` (`scripts/demo.sh voice`, screen-record the live window)
- spoken patrol film: `out/walk/lidar-20261024-120616-20261027-154514/walk_2x.mp4` (kitchen → passway → garden)
- scan walk and app: screen recording on the LiDAR iPhone, plus a second phone filming the walk

## Shot list and voice-over

### 0:00–0:08 · The command
**Show:** you at the MacBook, saying it into the mic. Cut to the live window: the subtitle "revisa la cocina y luego el
jardín", then "Nemotron Ultra · 1.9 s · plan: kitchen, then garden".
**Say (own voice, Spanish):**
> Revisa la cocina y luego el jardín.

### 0:08–0:30 · The payoff
**Show:** the duck answers aloud, turns on the spot, the "sure" bar fills (7.8 s), it walks to the kitchen; the map
inset shows belief vs truth with the error in cm.
**On screen:** "Real robot software · simulated body · a twin of a real home"
**Say:**
> This robot has never been told where it is. Nemotron Omni wrote down what I said, Nemotron Ultra picked the two rooms
> of this home, in my order, and the robot told me its plan. Now it finds itself from an eight-by-eight depth sensor and
> a map we made with one iPhone walk. No stickers on the walls.

### 0:30–1:30 · Real: the walk that makes the map (≥ 1 min of hardware)
**Show:** the phone filmed walking through the home, the LiDAR app recording, room taps ("kitchen", "garden"), Upload,
My maps going from "Building" to "Ready", then `docs/map_build.gif` and `docs/house_3d.gif`, then the room editor
renaming rooms.
**On screen:** "iPhone LiDAR · RGB + depth + ARKit pose · one walk"
**Say:**
> It starts with a phone. I walk through my home once with a LiDAR iPhone and tap each room as I enter it. The app
> records colour, depth and where the phone is, and uploads it. The server fuses the depth into walls, doors and rooms,
> and builds a 3D twin of the home that a robot simulator can load. If a room is wrong, I fix it in the browser. That's
> the whole setup: one walk, a few minutes, and no mapping run by the robot.

### 1:30–2:10 · It listens, and it asks
**Show:** the spoken patrol film: kitchen 28 cm, passway 13 cm, garden 6 cm; a question banner ("my camera says
another room: say I've arrived?"). Then the terminal: "go swim in the pool" → "I can't swim, but I can patrol the
pool. Should I?"; "alto" → stops at once.
**On screen:** "Models never steer: A* plans every path" · "answers automatic in this walk" when the banner shows
**Say:**
> The model never steers. Nemotron only picks rooms of this home, and the robot plans every step itself. When it's
> unsure, it asks, in plain words that Ultra writes and checks. Ask it something unsafe, like jumping into the pool, and
> it asks back instead of moving. Say "alto", and it stops, without waiting for any model.

### 2:10–2:40 · The numbers
**Show:** a results card.
**On screen:**
- "162 spoken commands, 2 homes, 6 languages: keywords 56 % → Nemotron Ultra 91 %"
- "Finds itself in 7.8 s · 22 test houses: wrong locks 89 → 3 of 96 runs"
- "Nemotron VL on real frames: 79 % vs colours 75 %: not enough, so it stays out of the loop"
**Say:**
> On a hundred and sixty-two commands in six languages, with typos, vague and unsafe requests, keyword matching gets
> fifty-six percent right. Nemotron Ultra gets ninety-one. It isn't perfect: a few unsafe commands still got through,
> and Nemotron's vision model barely beat a colour histogram on real frames, so it stays out of the robot's loop. We
> publish both.

### 2:40–3:00 · Who it's for, what's next
**Show:** the app's My maps, then the repo URL and "Apache-2.0".
**Say:**
> This is for home-robot makers: turn a tester's home into a test bench before the robot ships, and give people living
> alone a robot that understands "is the stove off?". It started as our OpenCV contest entry; the voice, Nemotron and
> the command test are new. Next: Token Factory end to end, and the real Microduck. House Twin Patrol.

## Honest notes for the edit
- Say "simulated body" whenever the duck is on screen: the robot software and walking policy are real; the body,
  sensors and furniture are simulated.
- Show "answers automatic" whenever a question banner appears in a filmed walk, unless you answered live.
- Every model number so far is from NVIDIA's API catalog. If the Token Factory re-run is done before filming, say
  "on Nebius Token Factory"; if not, don't say it.
- Omni on the free tier fails some calls (503). Record several takes of the opening command.
- The VL result is a limitation: say it as one.

## Recording tips
- Live window: `scripts/demo.sh voice`, then macOS screen recording (⇧⌘5) of the ffplay window plus the mic audio.
- iPhone: Control Centre → Screen Recording for the app; a second phone films the walk.
- Film clips: `ffmpeg -ss 0:20 -to 0:32 -i walk_2x.mp4 -c copy clip.mp4`.
- GIFs into video: `ffmpeg -i docs/map_build.gif -movflags faststart -pix_fmt yuv420p map_build.mp4`.
