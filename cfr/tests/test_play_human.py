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
    """Feed 'p' (play) for any FOLD prompt, 'stop' for any redraw prompt.
    Game must terminate without crashing.
    """
    # Generous buffer of responses — game won't ask more than this
    feeder = _make_input_feeder(["p"] * 6 + ["stop"] * 50)
    monkeypatch.setattr(builtins, "input", feeder)

    cfr = CFRAgent(RegretTable(), seed=0)
    run_match(cfr, human_seat=0, game_seed=42)

    out = capsys.readouterr().out
    assert "GAME OVER" in out
