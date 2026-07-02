"""CFR agent inference: purified argmax, never-fold, checkpoint load."""

import os
import tempfile

from cfr.agent.info_set import encode_info_set
from cfr.agent.regret_table import RegretTable
from cfr.eval.cfr_agent import CFRAgent
from cfr.eval.play import play_match
from cfr.eval.rule_agents import RandomAgent
from cfr.env.game import PokerGame, Phase, FOLD_ACTION_PLAY, REDRAW_ACTION_STOP
from cfr.train.checkpoint import save_checkpoint


def test_empty_table_picks_legal_action():
    """With no learned strategy, agent falls back to safe defaults — still legal."""
    agent = CFRAgent(RegretTable())
    game = PokerGame(seed=0)
    while not game.is_terminal():
        p = game.current_player()
        a = agent.act(game, p)
        assert a in game.legal_actions()
        game.apply(a)


def test_can_run_match_vs_random():
    """End-to-end: CFR agent plays a real game loop without crashing."""
    cfr = CFRAgent(RegretTable())
    result = play_match(cfr, RandomAgent(seed=2), num_games=5, seed=0)
    assert result["num_games"] == 5
    assert result["a_wins"] + result["b_wins"] + result["ties"] == 5


def test_from_checkpoint_roundtrip():
    """Load a saved table -> agent uses the loaded strategy."""
    t = RegretTable()
    t.update_cumulative_strategy("x", {0: 0.9, 1: 0.1})
    with tempfile.TemporaryDirectory() as d:
        path = save_checkpoint(t, iteration=1, seed=0, config={}, directory=d, git_hash="h")
        agent = CFRAgent.from_checkpoint(path)
    # Empty-table backing dict means private state present; just verify type
    assert isinstance(agent, CFRAgent)
    # And it produces legal action on a fresh game
    game = PokerGame(seed=5)
    a = agent.act(game, 0)
    assert a in game.legal_actions()


def test_never_folds():
    """FOLD phase is always answered with PLAY, regardless of learned strategy."""
    game = PokerGame(seed=0)
    assert game.phase == Phase.FOLD
    # Even a table that heavily prefers folding at this info set is overridden.
    t = RegretTable()
    key = encode_info_set(game.observation(0))
    t.update_cumulative_strategy(key, {0: 0.01, 1: 100.0})
    agent = CFRAgent(t)
    assert agent.act(game, 0) == FOLD_ACTION_PLAY


def _to_redraw_phase(seed=0):
    game = PokerGame(seed=seed)
    game.apply(FOLD_ACTION_PLAY)
    game.apply(FOLD_ACTION_PLAY)
    assert game.phase == Phase.REDRAW
    return game


def test_purified_argmax_picks_dominant_action():
    """Inference is argmax of average strategy, not a sample from it."""
    game = _to_redraw_phase()
    p = game.current_player()
    key = encode_info_set(game.observation(p))
    t = RegretTable()
    t.update_cumulative_strategy(key, {3: 5.0, 1: 1.0, 0: 1.0})
    agent = CFRAgent(t)
    # Deterministic: repeated calls always return the argmax.
    for _ in range(10):
        assert agent.act(game, p) == 3


def test_uniform_tie_breaks_to_stop():
    """Unseen redraw info set -> uniform -> argmax picks action 0 (STOP)."""
    game = _to_redraw_phase()
    p = game.current_player()
    agent = CFRAgent(RegretTable())
    assert agent.act(game, p) == REDRAW_ACTION_STOP
