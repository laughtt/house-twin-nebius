"""The duck's ears and voice on a Mac. The microphone → a stretch of speech (a loudness gate) → Nemotron Omni writes the
words down → Ultra picks the rooms (agent/language.py) → the duck says its plan (macOS `say`). Typed lines go the same
way. All of it off the 5 Hz loop: the loop picks up a finished command (`command()`), or stands still for a yes/no
(`ask()`). It doesn't listen while it speaks. A short "alto" / "stop" / "para" stops at once, without a model.

python -m agent.voice [--house out/houses/<name>]   → what it hears and the plan, no robot
"""
import argparse
import base64
import io
import json
import queue
import re
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

from agent import language, nebius

RATE, MIC = 16000, 48000  # what Omni gets; what the Mac's mic records natively (asking CoreAudio for 16 kHz failed once, err -50)
STOP = {"stop", "para", "paro", "detente", "alto", "quieto", "basta", "halt", "freeze"}
NO = {"no", "nope", "nah", "negativo"}


def spanish(text):
    return bool(re.search(r"[áéíóúñ¿¡]|\b(el|la|los|las|del|al|y|luego|después|revisa|ve|anda|mira|sí|si|qué|dónde)\b", text.lower()))


def english(text):
    return bool(re.search(r"\b(the|go|to|check|please|and|then|yes|stop|room)\b", text.lower()))


def transcribe(pcm, rooms=()):
    """int16 mono at 16 kHz → (the words, info). Empty when there was no speech. Telling Omni what it may hear helps the
    one-word answers (text-to-speech "sí" was "See." without it); example sentences don't (it wrote one down instead of "para")."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    url = "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()
    text, info = nebius.chat(nebius.P["omni"], [{"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": url}}, {
        "type": "text", "text": "A person, speaking Spanish or English, talks to a small home robot. It can be one word (sí, no, para, alto, "
        f"stop, yes). Room names it may hear: {', '.join(rooms)}. Write down what the person says, in their language. Reply with the "
        "words only, or nothing if no one speaks."}]}],
        max_tokens=100, tag="asr", timeout=20)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip().strip('"'), info


class Ear:
    def __init__(self, house, places, mic=True):
        self.house, self.places = house, places
        self.commands, self.answers, self.blocks = queue.Queue(), queue.Queue(), queue.Queue()
        self.asking, self.mute, self.es, self.context = False, False, False, None
        self.voice = threading.Lock()
        if mic:
            import sounddevice as sd
            self.stream = sd.InputStream(samplerate=MIC, channels=1, dtype="int16", blocksize=MIC // 10, callback=lambda d, *_: self.blocks.put(
                None if self.mute else d[:, 0].reshape(-1, MIC // RATE).mean(1).astype(np.int16)))  # 100 ms, down to 16 kHz
            self.stream.start()
            threading.Thread(target=self.listen, daemon=True).start()
        threading.Thread(target=lambda: [self.heard(text=s.strip(), via="typed") for s in sys.stdin], daemon=True).start()

    def listen(self):
        floor, pre, speech, quiet = [], [], [], 0
        while True:
            b = self.blocks.get()
            if b is None:  # its own voice
                pre, speech = [], []
                continue
            r = float(np.sqrt(np.mean(b.astype(float) ** 2)))
            gate = max(3 * float(np.median(floor)), 500) if len(floor) >= 10 else None
            if speech:
                speech.append(b)
                quiet = quiet + 1 if r < gate else 0
                if quiet >= 8 or len(speech) >= 100:  # 0.8 s of quiet, or 10 s
                    if len(speech) - quiet >= 3:
                        try:
                            self.heard(np.concatenate(speech))
                        except Exception as e:  # one bad answer never deafens the duck
                            print("not heard:", repr(e)[:200], flush=True)
                    speech = []
            elif gate and r >= gate:
                speech, quiet = pre + [b], 0
            else:
                pre, floor = (pre + [b])[-3:], (floor + [r])[-50:]

    def heard(self, pcm=None, text="", via="mic"):
        t0, info = time.time(), {}
        if pcm is not None:
            text, info = transcribe(pcm, [r.get("label", r["name"]).replace("_", " ") for r in self.house["rooms"]])
            if "error" in info:
                print("not heard:", info["error"], flush=True)
                self.speak("Perdón, no te entendí." if self.es else "Sorry, I didn't catch that.")
        if not text:
            return
        print(f"heard ({via}): {text}", flush=True)
        if self.asking:
            self.answers.put(text)
            return
        self.es = spanish(text) or self.es and not english(text)  # a word it can't place keeps the language
        c = {"wall": round(t0, 3), "heard": text, "via": via, "asr": {k: info[k] for k in ("model", "seconds", "error") if k in info}}
        w = set(re.findall(r"\w+", text.lower()))
        if len(w) <= 3 and w & STOP:
            self.context = None
            self.commands.put(c | {"stop": True, "goals": [], "say": "Me detengo." if self.es else "Stopping."})
            self.speak(self.commands.queue[-1]["say"])
            return
        cmd = f"{self.context}\nThe person answered: {text}" if self.context else text
        p = language.plan(cmd, self.house, places=self.places)
        self.context = f"{cmd}\nThe robot asked: {p['ask']}" if p["ask"] else None
        self.commands.put(c | {"stop": False, "command": cmd, "plan": p, "goals": p["goals"], "say": p["say"] or p["ask"]})
        self.speak(p["say"] or p["ask"])

    @property
    def lang(self):
        return "Spanish" if self.es else "English"

    def speak(self, text, wait=False):
        def run():
            with self.voice:
                self.mute = True
                subprocess.run(["say", "-v", "Paulina" if self.es else "Samantha", text.replace("_", " ")])
                time.sleep(0.3)  # the room's echo
                self.mute = False
        if text:
            print("duck:", text, flush=True)
            run() if wait else threading.Thread(target=run, daemon=True).start()

    def command(self):
        try:
            return self.commands.get_nowait()
        except queue.Empty:
            return None

    def ask(self, question, wait=10):
        """Says the question, waits for the answer → (yes?, what was heard). No answer = yes (as in an unattended run)."""
        while not self.answers.empty():
            self.answers.get()
        self.asking = True
        self.speak(question, wait=True)
        try:
            text = self.answers.get(timeout=wait)
        except queue.Empty:
            text = ""
        self.asking = False
        return not set(re.findall(r"\w+", text.lower())) & (NO | STOP), text


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="agent.voice")
    ap.add_argument("--house", default="samples/synth-multi")
    a = ap.parse_args()
    h = json.loads((Path(a.house) / "house.json").read_text())
    ear = Ear(h, [r["name"] for r in h["rooms"]])
    print("listening (speak, or type a line; Ctrl-C ends)", flush=True)
    while True:
        if c := ear.command():
            print(json.dumps({k: c[k] for k in ("heard", "goals", "say", "asr")}, ensure_ascii=False), flush=True)
        time.sleep(0.2)
