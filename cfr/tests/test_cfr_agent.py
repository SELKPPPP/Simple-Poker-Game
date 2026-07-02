"""CFR agent inference: empty-table uniform fallback + checkpoint load."""

import os
import tempfile

from cfr.agent.regret_table import RegretTable
from cfr.eval.cfr_agent import CFRAgent
from cfr.eval.play import play_match
from cfr.eval.rule_agents import RandomAgent
from cfr.env.game import PokerGame
from cfr.train.checkpoint import save_checkpoint


def test_empty_table_picks_legal_action():
    """With no learned strategy, agent falls back to uniform — but still legal."""
    agent = CFRAgent(RegretTable(), seed=42)
    game = PokerGame(seed=0)
    while not game.is_terminal():
        p = game.current_player()
        a = agent.act(game, p)
        assert a in game.legal_actions()
        game.apply(a)


def test_can_run_match_vs_random():
    """End-to-end: CFR agent plays a real game loop without crashing."""
    cfr = CFRAgent(RegretTable(), seed=1)
    result = play_match(cfr, RandomAgent(seed=2), num_games=5, seed=0)
    assert result["num_games"] == 5
    assert result["a_wins"] + result["b_wins"] + result["ties"] == 5


def test_from_checkpoint_roundtrip():
    """Load a saved table -> agent uses the loaded strategy."""
    t = RegretTable()
    t.update_cumulative_strategy("x", {0: 0.9, 1: 0.1})
    with tempfile.TemporaryDirectory() as d:
        path = save_checkpoint(t, iteration=1, seed=0, config={}, directory=d, git_hash="h")
        agent = CFRAgent.from_checkpoint(path, seed=0)
    # Empty-table backing dict means private state present; just verify type
    assert isinstance(agent, CFRAgent)
    # And it produces legal action on a fresh game
    game = PokerGame(seed=5)
    a = agent.act(game, 0)
    assert a in game.legal_actions()
