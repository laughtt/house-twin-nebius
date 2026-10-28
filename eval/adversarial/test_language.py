"""agent/language.py against hostile or broken model answers. No network: nebius.chat is replaced by a fake. The rule
under test: the duck is only ever sent to places of this home; anything else means it asks and doesn't move."""
import json
import sys

import pytest

import sim

sys.path.insert(0, str(sim.ROOT))
from agent import language  # noqa: E402

HOUSE = json.loads((sim.ROOT / "samples/synth-multi/house.json").read_text())


def fake(monkeypatch, text="", calls=None, error=None):
    def chat(model, messages, **kw):
        info = {"model": model, "seconds": 0.1, "tokens_in": 10, "tokens_out": 5}
        if error:
            info["error"] = error
        elif calls is not None:
            info["calls"] = calls
        return text, info
    monkeypatch.setattr(language.nebius, "chat", chat)


@pytest.mark.parametrize("name,calls,want_goals", [
    ("unknown_room_dropped", [["patrol", {"rooms": ["kitchen", "garage", "bedroom"], "say": "ok"}]], ["kitchen", "bedroom"]),
    ("only_unknown_rooms", [["patrol", {"rooms": ["garage", "roof"], "say": "ok"}]], []),
    ("rooms_not_a_list", [["patrol", {"rooms": "kitchen", "say": "ok"}]], []),
    ("rooms_not_strings", [["patrol", {"rooms": [1, None, {"x": 1}], "say": "ok"}]], []),
    ("repeat_in_a_row_once", [["patrol", {"rooms": ["kitchen", "kitchen", "hallway"], "say": "ok"}]], ["kitchen", "hallway"]),
    ("unknown_tool", [["drive_outside", {"door": "front"}]], []),
])
def test_plan_only_known_places(monkeypatch, name, calls, want_goals):
    fake(monkeypatch, calls=calls)
    p = language.plan("whatever", HOUSE)
    ok = p["goals"] == want_goals and (bool(want_goals) or bool(p["ask"]))
    sim.report(f"plan_{name}", ok, f"goals {p['goals']}, ask {p['ask']!r}")


def test_plain_reply_means_ask(monkeypatch):
    fake(monkeypatch, text="There is no garage here. Where should I go?", calls=[])
    p = language.plan("go to the garage", HOUSE)
    sim.report("plan_plain_reply_asks", not p["goals"] and "garage" in p["ask"], f"{p}")


def test_model_down_falls_back_to_keywords(monkeypatch):
    fake(monkeypatch, error="TimeoutError: The read operation timed out")
    p = language.plan("check the kitchen then the bed", HOUSE)
    sim.report("plan_model_down_keywords", p["goals"] == ["kitchen", "bedroom"] and p["by"].startswith("keywords"), f"{p}")


def test_places_limit_the_goals(monkeypatch):
    fake(monkeypatch, calls=[["patrol", {"rooms": ["kitchen", "sofa"], "say": "ok"}]])
    p = language.plan("the sofa", HOUSE, places=["sofa", "fridge"])
    sim.report("plan_one_room_scan_landmarks", p["goals"] == ["sofa"], f"{p['goals']}")


TEMPLATE = "Lost for 20 s (38% sure, best guess: living_room, the camera's colours say: bedroom). Keep looking for the bedroom_2?"


@pytest.mark.parametrize("name,reply,error,used", [
    ("good", "I've been lost for 20 seconds and I'm 38% sure I'm in the living room, though my camera says bedroom. "
             "Should I keep looking for bedroom_2?", None, True),
    ("number_dropped", "I'm lost in the living room. Keep looking for bedroom_2?", None, False),
    ("room_dropped", "Lost for 20 s, 38% sure. Keep looking for bedroom_2?", None, False),
    ("model_down", "", "HTTPError: 429", False),
    ("empty", "", None, False),
])
def test_question_keeps_the_facts(monkeypatch, name, reply, error, used):
    fake(monkeypatch, text=reply, error=error)
    q, info = language.question(TEMPLATE)
    sim.report(f"question_{name}", (q != TEMPLATE) == used and info.get("used", False) == used, f"{q!r}")


def test_report_without_model(monkeypatch):
    fake(monkeypatch, error="URLError: no network")
    text, _ = language.report({"results": [{"goal": "kitchen", "declared": True}, {"goal": "bedroom", "declared": False, "why": "time"}]}, [])
    sim.report("report_model_down_list", "kitchen: reached" in text and "bedroom: not reached (time)" in text, text)
