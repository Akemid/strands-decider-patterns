"""The simulator entry points and the scripted stand-in Decider (no network, no model)."""

from __future__ import annotations

from simulator.engine import SimMatch
from simulator.fake_decider import answer_for
from simulator.serve import build_game, parse_args


def test_defaults_serve_the_documented_endpoint():
    args = parse_args([])
    assert (args.host, args.port, args.seed) == ("127.0.0.1", 8765, 0)


def test_build_game_wires_an_engine_and_the_tick_limit():
    game = build_game(parse_args(["--ticks", "30", "--seed", "4"]))
    assert isinstance(game.engine, SimMatch) and game.engine.max_ticks == 30 and game.turns == 30
    assert (game.server.settings.host, game.server.settings.port) == ("127.0.0.1", 8765)


OPTIONS = {
    "enemy_flag_direct": "goal enemy_flag, 12 steps, first step right, 1 risk: can be tagged by blue-1 at step 4",
    "enemy_flag_detour": "goal enemy_flag, 14 steps, first step up",
    "wait": "do nothing this tick",
}


def test_scripted_decider_prefers_the_option_without_risks_or_blocks():
    questions = {"action": {"type": "choice", "instructions": "x", "criteria": OPTIONS},
                 "threat": {"type": "noul", "instructions": "t"}}
    answers = answer_for(questions)
    assert answers["action"]["choice"] == "enemy_flag_detour"
    assert answers["action"]["confidence"] >= 0.9
    assert answers["threat"]["noul"] == 0.0


def test_scripted_decider_grabs_the_flag_when_offered():
    options = {"grab_flag": "pick up the enemy flag you are standing on", **OPTIONS}
    questions = {"action": {"type": "choice", "instructions": "x", "criteria": options}}
    assert answer_for(questions)["action"]["choice"] == "grab_flag"
