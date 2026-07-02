"""Phase 2 acceptance: Kuhn Poker rules + MCCFR convergence to Nash.

Per plan §4: MCCFR exploitability < 0.01 on Kuhn (CFR correctness gold standard).
"""

import pytest

from cfr.agent.mccfr import train_kuhn
from cfr.env.kuhn import (
    ALL_CARDS, BET_CALL, CHECK_FOLD, JACK, KING, QUEEN, KuhnGame,
)
from cfr.eval.exploitability import compute_kuhn_exploitability


# ============ Kuhn game rules ============

def test_deal_produces_two_distinct_cards():
    g = KuhnGame(seed=42)
    assert g.hands[0] != g.hands[1]
    assert g.hands[0] in ALL_CARDS
    assert g.hands[1] in ALL_CARDS


def test_explicit_hands_override():
    g = KuhnGame(hands=(KING, JACK))
    assert g.hands == [KING, JACK]


def test_initial_state_not_terminal():
    g = KuhnGame(hands=(KING, JACK))
    assert not g.is_terminal()
    assert g.current_player() == 0
    assert g.legal_actions() == [CHECK_FOLD, BET_CALL]


def test_both_check_terminal_higher_card_wins():
    g = KuhnGame(hands=(KING, JACK))
    g.apply(CHECK_FOLD)
    g.apply(CHECK_FOLD)
    assert g.is_terminal()
    assert g.utility(0) == 1.0
    assert g.utility(1) == -1.0


def test_p1_folds_after_p0_bet():
    g = KuhnGame(hands=(JACK, QUEEN))
    g.apply(BET_CALL)
    g.apply(CHECK_FOLD)
    assert g.is_terminal()
    assert g.utility(0) == 1.0
    assert g.utility(1) == -1.0


def test_p0_folds_after_check_bet():
    g = KuhnGame(hands=(JACK, KING))
    g.apply(CHECK_FOLD)
    g.apply(BET_CALL)
    g.apply(CHECK_FOLD)
    assert g.is_terminal()
    assert g.utility(0) == -1.0
    assert g.utility(1) == 1.0


def test_check_bet_call_showdown_pot_2():
    g = KuhnGame(hands=(KING, JACK))
    g.apply(CHECK_FOLD)
    g.apply(BET_CALL)
    g.apply(BET_CALL)
    assert g.is_terminal()
    assert g.utility(0) == 2.0
    assert g.utility(1) == -2.0


def test_bet_call_showdown_pot_2():
    g = KuhnGame(hands=(JACK, KING))
    g.apply(BET_CALL)
    g.apply(BET_CALL)
    assert g.is_terminal()
    assert g.utility(0) == -2.0
    assert g.utility(1) == 2.0


def test_zero_sum_all_terminals_all_deals():
    deals = [(JACK, KING), (KING, JACK), (QUEEN, JACK),
             (JACK, QUEEN), (KING, QUEEN), (QUEEN, KING)]
    terminals = [(0, 0), (0, 1, 0), (0, 1, 1), (1, 0), (1, 1)]
    for hands in deals:
        for hist in terminals:
            g = KuhnGame(hands=hands)
            for a in hist:
                g.apply(a)
            assert g.utility(0) + g.utility(1) == 0.0


def test_legal_actions_empty_at_terminal():
    g = KuhnGame(hands=(JACK, KING))
    g.apply(BET_CALL)
    g.apply(BET_CALL)
    assert g.legal_actions() == []


def test_apply_on_terminal_raises():
    g = KuhnGame(hands=(JACK, KING))
    g.apply(BET_CALL)
    g.apply(BET_CALL)
    with pytest.raises(RuntimeError):
        g.apply(CHECK_FOLD)


def test_illegal_action_raises():
    g = KuhnGame(hands=(JACK, KING))
    with pytest.raises(ValueError):
        g.apply(2)


def test_info_set_includes_only_own_card():
    g = KuhnGame(hands=(JACK, KING))
    assert g.info_set_key(0).startswith(f"{JACK}:")
    assert g.info_set_key(1).startswith(f"{KING}:")
    assert g.info_set_key(0) != g.info_set_key(1)


def test_info_set_evolves_with_history():
    g = KuhnGame(hands=(JACK, KING))
    k_initial = g.info_set_key(0)
    g.apply(CHECK_FOLD)
    k_after = g.info_set_key(0)
    assert k_initial != k_after
    assert k_after == f"{JACK}:0"


# ============ MCCFR sanity ============

def test_mccfr_short_run_strong_card_bets_more():
    """1000 iter: K should bet more than J in P0's first decision (qualitative)."""
    table = train_kuhn(num_iterations=1000, seed=42)
    avg_j = table.average_strategy(f"{JACK}:", [CHECK_FOLD, BET_CALL])
    avg_k = table.average_strategy(f"{KING}:", [CHECK_FOLD, BET_CALL])
    assert avg_k[BET_CALL] > avg_j[BET_CALL]


def test_mccfr_k_calls_more_than_j_after_check_bet():
    """K should call almost always; J should fold almost always."""
    table = train_kuhn(num_iterations=10_000, seed=42)
    avg_j_call = table.average_strategy(f"{JACK}:01", [CHECK_FOLD, BET_CALL])
    avg_k_call = table.average_strategy(f"{KING}:01", [CHECK_FOLD, BET_CALL])
    assert avg_k_call[BET_CALL] > avg_j_call[BET_CALL]


# ============ Phase 2 acceptance: exploitability < 0.01 ============

def test_mccfr_converges_to_nash_under_001():
    """Phase 2 gate: 10^5 iterations of External Sampling MCCFR → exploit < 0.01."""
    table = train_kuhn(num_iterations=100_000, seed=42)
    exploit = compute_kuhn_exploitability(table)
    assert abs(exploit) < 0.01, f"exploitability {exploit:.4f} >= 0.01"
