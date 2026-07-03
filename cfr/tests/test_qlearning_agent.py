"""Tests for the legacy Q-learning bot adapter.

Card encoding (see cfr/env/poker_rules.get_rank_suit):
    card = suit * 13 + raw_rank;  raw_rank: 0=A, 1=2, ..., 12=K
    post-adjusted rank: 2=0, ..., K=11, A=12
"""

import pickle

from cfr.env.game import (
    PokerGame, Phase, FOLD_ACTION_PLAY, REDRAW_ACTION_STOP,
)
from cfr.eval.qlearning_agent import QTableAgent, legacy_state
from cfr.eval.rule_agents import AlwaysCallAgent
from cfr.eval.play import play_match

SUIT_S, SUIT_H, SUIT_D, SUIT_C = 0, 1, 2, 3

_RAW_RANK = {
    "A": 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8,
    "T": 9, "J": 10, "Q": 11, "K": 12,
}


def card(rank, suit):
    return suit * 13 + _RAW_RANK[rank]


def _fake_agent(q_table, tmp_path):
    p = tmp_path / "q.pkl"
    with open(p, "wb") as f:
        pickle.dump(q_table, f)
    return QTableAgent(q_table_path=str(p))


# ============ legacy_state encoding ============

def test_legacy_state_full_house():
    # 8,8,8,K,K -> full house (rank 6), top tiebreaker = trip rank 8 (adj 6).
    # No 3-of-suit -> suited mask zero; no 3-in-straight-window -> zero mask.
    h = [card(8, SUIT_S), card(8, SUIT_H), card(8, SUIT_D),
         card("K", SUIT_C), card("K", SUIT_S)]
    assert legacy_state(h, 7, 0, 0) == \
        (6, 6, (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), 7, 0, 0)


def test_legacy_state_suited_and_straight_masks():
    # 3 spades -> suited mask marks spade positions.
    # Straight windows scan low-to-high and FIRST max-hit window wins (same
    # quirk as the original get_state): window {2..6} hits 5,6,2 before
    # {3..7} can hit 5,6,7 -> mask marks 5,6 and the 2, not the 7.
    h = [card(5, SUIT_S), card(6, SUIT_S), card(7, SUIT_S),
         card("K", SUIT_H), card(2, SUIT_D)]
    st = legacy_state(h, 4, 1, 0)
    assert st[0] == 0                      # high card
    assert st[1] == 11                     # top card K (adj 11)
    assert st[2] == (1, 1, 1, 0, 0)        # suited mask: 3 spades
    assert st[3] == (1, 1, 0, 0, 1)        # straight mask: 5,6 + 2
    assert st[4:] == (4, 1, 0)


# ============ adapter behavior ============

def test_never_folds(tmp_path):
    agent = _fake_agent({}, tmp_path)
    game = PokerGame(seed=0)
    assert game.phase == Phase.FOLD
    assert agent.act(game, 0) == FOLD_ACTION_PLAY


def _to_redraw_phase(seed=0):
    game = PokerGame(seed=seed)
    game.apply(FOLD_ACTION_PLAY)
    game.apply(FOLD_ACTION_PLAY)
    assert game.phase == Phase.REDRAW
    return game


def test_unseen_state_stops(tmp_path):
    agent = _fake_agent({}, tmp_path)
    game = _to_redraw_phase()
    assert agent.act(game, game.current_player()) == REDRAW_ACTION_STOP


def test_greedy_argmax_over_legal(tmp_path):
    game = _to_redraw_phase()
    p = game.current_player()
    obs = game.observation(p)
    state = legacy_state(list(obs.my_hand), obs.my_remaining_redraws,
                         obs.my_round_wins, obs.opp_round_wins)
    row = [0.0] * 32
    row[5] = 1.0   # popcount(5)=2 <= budget 7 -> legal -> should be picked
    agent = _fake_agent({state: row}, tmp_path)
    assert agent.act(game, p) == 5

    # If the argmax action is illegal (popcount > budget), the legal-subset
    # argmax must win instead.
    row2 = [0.0] * 32
    row2[31] = 9.0  # popcount 5
    row2[3] = 1.0   # popcount 2
    state_low_budget = state[:4] + (1,) + state[5:]  # budget 1: 31 and 3 illegal
    game2 = _to_redraw_phase()
    game2.remaining_redraws[game2.current_player()] = 1
    p2 = game2.current_player()
    obs2 = game2.observation(p2)
    assert legacy_state(list(obs2.my_hand), 1, 0, 0) == state_low_budget
    row2[2] = 0.5   # popcount 1 -> best legal
    agent2 = _fake_agent({state_low_budget: row2}, tmp_path)
    assert agent2.act(game2, p2) == 2


def test_one_decision_per_round_then_stop(tmp_path):
    game = _to_redraw_phase()
    p = game.current_player()
    obs = game.observation(p)
    state = legacy_state(list(obs.my_hand), obs.my_remaining_redraws,
                         obs.my_round_wins, obs.opp_round_wins)
    row = [0.0] * 32
    row[1] = 1.0
    agent = _fake_agent({state: row}, tmp_path)

    first = agent.act(game, p)
    assert first == 1
    game.apply(first)          # p's iter-1 action committed/pending
    game.apply(REDRAW_ACTION_STOP)  # other player stops
    # Next iter, same round: agent already redrew -> must STOP regardless of q.
    if not game.is_terminal() and game.phase == Phase.REDRAW \
            and game.current_player() == p:
        assert agent.act(game, p) == REDRAW_ACTION_STOP


# ============ rules-copy guard ============

def test_poker_rules_copy_stays_identical():
    """legacy_state's bit-for-bit q_table compatibility depends on
    cfr/env/poker_rules.py being an exact copy of the backend original.
    If this fails, a rules fix landed in one copy only — sync them.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    ours = (root / "cfr" / "env" / "poker_rules.py").read_bytes()
    theirs = (root / "backend" / "model_lib" / "poker_rules.py").read_bytes()
    assert ours == theirs


# ============ real q_table smoke ============

def test_real_q_table_plays_full_match():
    agent = QTableAgent()  # loads backend/model_lib/q_table.pkl
    result = play_match(agent, AlwaysCallAgent(), num_games=20, seed=7)
    assert result["num_games"] == 20
    assert result["a_wins"] + result["b_wins"] + result["ties"] == 20
