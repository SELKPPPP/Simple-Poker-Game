"""
Phase 0 tests: game rules + information hiding.

CRITICAL: information hiding tests are CFR correctness prerequisites.
If they fail, no point proceeding to Phase 1.

Per-round action sequence:
    FOLD phase (P1 fold/play, then P2 fold/play)
    IF any fold: round resolves
    ELSE REDRAW phase (iterative loop):
        each iter: both active players commit action (0=STOP, 1..31=bitmask)
        loop ends when both have chosen STOP
    Showdown
"""

import pytest

from cfr.env.game import (
    PokerGame, Phase, HOLE_INDICES, FACE_UP_INDICES,
    FOLD_ACTION_PLAY, FOLD_ACTION_FOLD,
    REDRAW_ACTION_STOP,
    INITIAL_REDRAWS, MAX_ROUNDS,
)


# ---------- helpers ----------

def both_play(g):
    g.apply(FOLD_ACTION_PLAY)
    g.apply(FOLD_ACTION_PLAY)


def both_fold(g):
    g.apply(FOLD_ACTION_FOLD)
    g.apply(FOLD_ACTION_FOLD)


def p1_folds(g):
    g.apply(FOLD_ACTION_FOLD)
    g.apply(FOLD_ACTION_PLAY)


def p2_folds(g):
    g.apply(FOLD_ACTION_PLAY)
    g.apply(FOLD_ACTION_FOLD)


def both_stop_redraw(g):
    """Both players immediately STOP in REDRAW phase (no redraws taken)."""
    g.apply(REDRAW_ACTION_STOP)
    g.apply(REDRAW_ACTION_STOP)


def play_round_no_redraws(g):
    """Both play in fold, both STOP immediately → showdown with starting hands."""
    both_play(g)
    both_stop_redraw(g)


# ---------- basic state ----------

def test_reset_initial_state():
    g = PokerGame(seed=42)
    assert g.round_num == 1
    assert g.round_wins == [0, 0]
    assert g.remaining_redraws == [INITIAL_REDRAWS, INITIAL_REDRAWS]
    assert g.phase == Phase.FOLD
    assert g.current_player() == 0
    assert not g.is_terminal()
    assert len(g.hands[0]) == 5 and len(g.hands[1]) == 5


def test_no_duplicate_cards_in_initial_deal():
    g = PokerGame(seed=42)
    assert len(set(g.hands[0] + g.hands[1])) == 10


# ---------- legal actions ----------

def test_fold_actions_in_fold_phase():
    g = PokerGame(seed=1)
    assert g.legal_actions() == [FOLD_ACTION_PLAY, FOLD_ACTION_FOLD]


def test_redraw_actions_cap_by_remaining():
    g = PokerGame(seed=1)
    g.remaining_redraws[0] = 2
    both_play(g)
    legal = g.legal_actions()
    for a in legal:
        assert bin(a).count('1') <= 2
    assert 31 not in legal
    assert REDRAW_ACTION_STOP in legal


def test_zero_budget_only_stop_is_legal():
    g = PokerGame(seed=1)
    g.remaining_redraws[0] = 0
    both_play(g)
    assert g.legal_actions() == [REDRAW_ACTION_STOP]


def test_phase_advances_fold_to_redraw_when_both_play():
    g = PokerGame(seed=1)
    both_play(g)
    assert g.phase == Phase.REDRAW
    assert g.current_player() == 0


# ---------- INFORMATION HIDING ----------

def test_opp_hole_cards_hidden_during_round():
    g = PokerGame(seed=42)
    obs_p1 = g.observation(0)
    for i in HOLE_INDICES:
        assert obs_p1.opp_visible_cards[i] is None
    for i in FACE_UP_INDICES:
        assert obs_p1.opp_visible_cards[i] == g.hands[1][i]


def test_my_full_hand_always_visible_to_self():
    g = PokerGame(seed=42)
    assert g.observation(0).my_hand == tuple(g.hands[0])
    assert g.observation(1).my_hand == tuple(g.hands[1])


def test_fold_decision_hidden_until_both_commit():
    g = PokerGame(seed=42)
    g.apply(FOLD_ACTION_FOLD)
    obs_p2 = g.observation(1)
    assert obs_p2.opp_fold_this_round is None


def test_redraw_action_hidden_from_opp_in_same_iter():
    """Simultaneous trick: P1's pending action invisible to P2."""
    g = PokerGame(seed=7)
    both_play(g)
    g.apply(7)  # P1 redraws 3 cards
    obs_p2 = g.observation(1)
    # P2 sees no committed actions from P1 yet (iter not finalized)
    assert obs_p2.opp_redraw_counts_this_round == ()


def test_my_pending_redraw_visible_to_self():
    g = PokerGame(seed=7)
    both_play(g)
    g.apply(7)  # P1: 3 cards (pending)
    obs_p1 = g.observation(0)
    # P1 sees their own pending action
    assert obs_p1.my_redraw_actions_this_round == (7,)


def test_redraw_revealed_after_iter_finalizes():
    g = PokerGame(seed=7)
    both_play(g)
    g.apply(7)   # P1: 3 cards
    g.apply(3)   # P2: 2 cards → iter finalizes, new iter starts
    obs_p1 = g.observation(0)
    obs_p2 = g.observation(1)
    assert obs_p1.opp_redraw_counts_this_round == (2,)
    assert obs_p2.opp_redraw_counts_this_round == (3,)


# ---------- iterative redraw mechanics ----------

def test_both_stop_immediately_ends_redraw():
    g = PokerGame(seed=42)
    both_play(g)
    both_stop_redraw(g)
    # Round resolved, new round started
    assert g.round_num == 2 or g.is_terminal()
    # Budget unchanged
    assert g.remaining_redraws == [INITIAL_REDRAWS, INITIAL_REDRAWS]


def test_single_iter_redraw_then_both_stop():
    g = PokerGame(seed=7)
    both_play(g)
    g.apply(3)   # P1: 2 cards
    g.apply(7)   # P2: 3 cards → iter finalizes
    # New iter starts, both active, P1 to act
    g.apply(REDRAW_ACTION_STOP)  # P1 stops
    g.apply(REDRAW_ACTION_STOP)  # P2 stops → all done, showdown
    # Verify budgets
    assert g.redraw_history[0][-1] == (3, REDRAW_ACTION_STOP)
    assert g.redraw_history[1][-1] == (7, REDRAW_ACTION_STOP)


def test_iterative_chase_until_budget_exhausted():
    """P1 has budget 3, redraws 1 card per iter 3 times, then forced STOP."""
    g = PokerGame(seed=42)
    g.remaining_redraws[0] = 3
    both_play(g)
    # iter 1: P1 redraws 1 card (bit 0), P2 stops
    g.apply(1)
    g.apply(REDRAW_ACTION_STOP)
    assert g.remaining_redraws[0] == 2
    # iter 2: P1 still active, redraws 1 card
    g.apply(1)
    assert g.remaining_redraws[0] == 1
    # iter 3: P1 redraws 1 card → budget 0
    g.apply(1)
    assert g.remaining_redraws[0] == 0
    # iter 4: P1 forced to STOP (only legal action)
    assert g.legal_actions() == [REDRAW_ACTION_STOP]
    g.apply(REDRAW_ACTION_STOP)
    # All done now
    assert g.redraw_history[0][-1] == (1, 1, 1, REDRAW_ACTION_STOP)
    assert g.redraw_history[1][-1] == (REDRAW_ACTION_STOP,)


def test_asymmetric_play_p2_keeps_going_after_p1_stops():
    g = PokerGame(seed=42)
    both_play(g)
    g.apply(REDRAW_ACTION_STOP)   # P1 stops
    g.apply(3)                     # P2 redraws 2 → iter finalizes, P1 done, P2 active
    # Only P2 acts now
    assert g.current_player() == 1
    g.apply(3)                     # P2 redraws 2 more
    g.apply(REDRAW_ACTION_STOP)    # P2 stops
    # All done, round resolved
    assert g.redraw_history[0][-1] == (REDRAW_ACTION_STOP,)
    assert g.redraw_history[1][-1] == (3, 3, REDRAW_ACTION_STOP)


def test_redraw_decreases_remaining_budget():
    g = PokerGame(seed=42)
    both_play(g)
    g.apply(7)                  # P1: 3 cards
    g.apply(3)                  # P2: 2 cards
    g.apply(REDRAW_ACTION_STOP)
    g.apply(REDRAW_ACTION_STOP)
    assert g.remaining_redraws[0] == INITIAL_REDRAWS - 3
    assert g.remaining_redraws[1] == INITIAL_REDRAWS - 2


def test_no_card_duplication_after_multi_iter_redraw():
    g = PokerGame(seed=7)
    both_play(g)
    g.apply(31)                  # P1 redraws all 5
    g.apply(31)                  # P2 redraws all 5
    # Both still active, can keep going
    g.apply(REDRAW_ACTION_STOP)
    g.apply(REDRAW_ACTION_STOP)
    # New round started (or terminal): hands fresh
    assert len(set(g.hands[0] + g.hands[1])) == 10


# ---------- fold mechanics ----------

def test_one_fold_other_wins_round():
    g = PokerGame(seed=42)
    p1_folds(g)
    assert g.round_wins == [0, 1]


def test_fold_preserves_full_redraw_budget():
    g = PokerGame(seed=42)
    both_fold(g)
    assert g.remaining_redraws == [INITIAL_REDRAWS, INITIAL_REDRAWS]


def test_one_fold_preserves_full_budget():
    g = PokerGame(seed=42)
    p2_folds(g)
    assert g.remaining_redraws == [INITIAL_REDRAWS, INITIAL_REDRAWS]


def test_fold_skips_redraw_phase():
    g = PokerGame(seed=42)
    p1_folds(g)
    # Round resolved → next round starts → phase back to FOLD
    assert g.phase == Phase.FOLD
    assert g.fold_history[0][-1] is True
    assert g.fold_history[1][-1] is False


def test_both_fold_is_push():
    g = PokerGame(seed=42)
    both_fold(g)
    assert g.round_wins == [0, 0]


def test_showdown_reveals_hole_cards_at_terminal():
    g = PokerGame(seed=42)
    for _ in range(MAX_ROUNDS):
        if g.is_terminal():
            break
        play_round_no_redraws(g)
    assert g.is_terminal()
    obs_p1 = g.observation(0)
    for i in HOLE_INDICES:
        assert obs_p1.opp_visible_cards[i] is not None


def test_fold_at_terminal_does_not_reveal_hole():
    g = PokerGame(seed=42)
    p2_folds(g)
    p2_folds(g)
    assert g.is_terminal()
    assert g.round_wins == [2, 0]
    obs_p1 = g.observation(0)
    for i in HOLE_INDICES:
        assert obs_p1.opp_visible_cards[i] is None


# ---------- BO3 termination ----------

def test_terminal_at_two_wins():
    g = PokerGame(seed=42)
    p2_folds(g)
    assert not g.is_terminal()
    p2_folds(g)
    assert g.is_terminal()
    assert g.round_wins == [2, 0]


def test_terminal_at_three_rounds_even_without_two_wins():
    g = PokerGame(seed=42)
    for _ in range(MAX_ROUNDS):
        both_fold(g)
    assert g.is_terminal()
    assert g.round_wins == [0, 0]


# ---------- utility ----------

def test_utility_zero_sum():
    g = PokerGame(seed=42)
    p2_folds(g); p2_folds(g)
    assert g.utility(0) == 1.0
    assert g.utility(1) == -1.0


def test_utility_tie_is_zero():
    g = PokerGame(seed=42)
    for _ in range(MAX_ROUNDS):
        both_fold(g)
    assert g.utility(0) == 0.0
    assert g.utility(1) == 0.0


def test_utility_raises_on_nonterminal():
    g = PokerGame(seed=42)
    with pytest.raises(RuntimeError):
        g.utility(0)


# ---------- history tracking ----------

def test_fold_history_records_per_round():
    g = PokerGame(seed=42)
    p1_folds(g)
    both_fold(g)
    play_round_no_redraws(g)
    assert g.fold_history[0] == [True, True, False]
    assert g.fold_history[1] == [False, True, False]


def test_redraw_history_records_iter_sequences():
    g = PokerGame(seed=42)
    both_play(g)
    g.apply(3)                    # P1: 2 cards
    g.apply(REDRAW_ACTION_STOP)   # P2: STOP
    g.apply(1)                    # P1: 1 card
    g.apply(REDRAW_ACTION_STOP)   # P1 STOPs
    # Round resolved, history saved
    assert g.redraw_history[0][-1] == (3, 1, REDRAW_ACTION_STOP)
    assert g.redraw_history[1][-1] == (REDRAW_ACTION_STOP,)


def test_redraw_history_empty_tuple_for_folded_round():
    g = PokerGame(seed=42)
    both_fold(g)
    assert g.redraw_history[0][-1] == ()
    assert g.redraw_history[1][-1] == ()


# ---------- observation hashability ----------

def test_observation_is_hashable():
    g = PokerGame(seed=42)
    assert isinstance(hash(g.observation(0)), int)


def test_observation_hashable_after_complex_history():
    g = PokerGame(seed=42)
    both_fold(g)
    both_play(g)
    g.apply(3); g.apply(7)
    g.apply(REDRAW_ACTION_STOP); g.apply(REDRAW_ACTION_STOP)
    assert isinstance(hash(g.observation(0)), int)
    assert isinstance(hash(g.observation(1)), int)


# ---------- error paths ----------

def test_illegal_redraw_action_raises():
    g = PokerGame(seed=1)
    g.remaining_redraws[0] = 1
    both_play(g)
    with pytest.raises(ValueError):
        g.apply(31)


def test_illegal_fold_action_raises():
    g = PokerGame(seed=1)
    with pytest.raises(ValueError):
        g.apply(5)


def test_apply_on_terminal_raises():
    g = PokerGame(seed=42)
    for _ in range(MAX_ROUNDS):
        both_fold(g)
    with pytest.raises(RuntimeError):
        g.apply(FOLD_ACTION_PLAY)
