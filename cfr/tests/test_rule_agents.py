"""Rule agents return legal actions in every phase, and respect redraw budget."""

from cfr.env.game import PokerGame, Phase, REDRAW_ACTION_STOP
from cfr.eval.rule_agents import (
    AlwaysCallAgent, HeuristicAgent, LooseAgent, RandomAgent, TightAgent,
)

ALL_AGENT_FACTORIES = [
    lambda: RandomAgent(seed=0),
    lambda: AlwaysCallAgent(),
    lambda: TightAgent(),
    lambda: LooseAgent(),
    lambda: HeuristicAgent(),
]


def _drive_one_game(agent, opp_agent, seed: int) -> None:
    """Each step asserts the chosen action is in legal_actions."""
    game = PokerGame(seed=seed)
    seats = {0: agent, 1: opp_agent}
    while not game.is_terminal():
        p = game.current_player()
        a = seats[p].act(game, p)
        assert a in game.legal_actions(), \
            f"agent {seats[p].name} returned illegal action {a} in phase {game.phase}"
        game.apply(a)


def test_all_agents_play_legal_against_random():
    opp = RandomAgent(seed=99)
    for factory in ALL_AGENT_FACTORIES:
        agent = factory()
        for seed in range(5):
            _drive_one_game(agent, opp, seed=seed)


def test_always_call_never_folds_never_redraws():
    """AlwaysCall is the simplest invariant agent."""
    a = AlwaysCallAgent()
    for seed in range(3):
        game = PokerGame(seed=seed)
        while not game.is_terminal():
            if game.current_player() == 0:
                action = a.act(game, 0)
                assert action == 0  # PLAY in FOLD, STOP in REDRAW
            else:
                action = RandomAgent(seed=seed).act(game, 1)
            game.apply(action)


def test_tight_folds_high_card_hands():
    """At least once across many seeds, tight should fold on a weak deal."""
    t = TightAgent()
    saw_fold = False
    for seed in range(50):
        game = PokerGame(seed=seed)
        if game.phase == Phase.FOLD:
            a = t.act(game, 0)
            if a == 1:  # FOLD_ACTION_FOLD
                saw_fold = True
                break
    assert saw_fold, "Tight never folded across 50 seeds — threshold too lax?"
