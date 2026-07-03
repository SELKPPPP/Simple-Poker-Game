"""Human CLI smoke tests: drive run_match with mocked input()."""

import builtins

import pytest

from cfr.agent.regret_table import RegretTable
from cfr.eval.cfr_agent import CFRAgent
from cfr.scripts.play_human import card_str, prompt_redraw_action, run_match


def _make_input_feeder(responses):
    """Return a callable that yields each response in order, then raises."""
    it = iter(responses)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration as e:
            raise RuntimeError(f"ran out of mock input at prompt: {prompt!r}") from e

    return fake_input


def test_card_str_renders_known_cards():
    # card 0 = spade A (raw_rank 0 -> 'A', suit 0 -> 'S')
    assert "A" in card_str(0) and "S" in card_str(0)
    # None renders as hidden
    assert "?" in card_str(None)


def test_prompt_redraw_handles_blank_stop_and_indices(monkeypatch):
    monkeypatch.setattr(builtins, "input", _make_input_feeder([""]))
    assert prompt_redraw_action(budget=5) == 0  # STOP

    monkeypatch.setattr(builtins, "input", _make_input_feeder(["0,2"]))
    assert prompt_redraw_action(budget=5) == 0b101  # bits 0,2


def test_prompt_redraw_reprompts_on_invalid(monkeypatch):
    # bad input then good
    monkeypatch.setattr(builtins, "input",
                        _make_input_feeder(["garbage", "stop"]))
    assert prompt_redraw_action(budget=3) == 0


def test_prompt_redraw_rejects_over_budget(monkeypatch):
    # 3 indices but budget=2 -> reprompt -> then STOP
    monkeypatch.setattr(builtins, "input",
                        _make_input_feeder(["0,1,2", ""]))
    assert prompt_redraw_action(budget=2) == 0


def test_run_match_completes_with_always_play(monkeypatch, capsys):
    """Answer 'p' to FOLD prompts, 'stop' to redraw prompts (prompt-aware,
    so the schedule doesn't depend on how the AI plays). Game must terminate
    without crashing.
    """
    calls = 0

    def fake_input(prompt=""):
        nonlocal calls
        calls += 1
        if calls > 200:  # a full game needs far fewer prompts -> loop bug
            raise RuntimeError(f"input loop did not terminate; prompt: {prompt!r}")
        return "p" if "[p]lay" in prompt else "stop"

    monkeypatch.setattr(builtins, "input", fake_input)

    cfr = CFRAgent(RegretTable())
    run_match(cfr, human_seat=0, game_seed=42)

    out = capsys.readouterr().out
    assert "GAME OVER" in out
