"""Tests for info set encoding.

Card encoding (consistent with cfr/env/poker_rules.get_rank_suit):
    card = suit * 13 + raw_rank
    raw_rank: 0=A, 1=2, 2=3, ..., 12=K
    suit: 0=Spades, 1=Hearts, 2=Diamonds, 3=Clubs
    post-adjusted rank (used in features): 2=0, 3=1, ..., K=11, A=12
"""

from cfr.agent.info_set import (
    DRAW_GUTSHOT,
    DRAW_NONE,
    DRAW_OPEN_ENDED,
    DRAW_STRAIGHT_MADE,
    _straight_features,
    _suited_mask,
    encode_info_set,
)
from cfr.env.game import PokerGame
from cfr.env.observation import Observation

SUIT_S, SUIT_H, SUIT_D, SUIT_C = 0, 1, 2, 3

_RAW_RANK = {
    "A": 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8,
    "T": 9, "J": 10, "Q": 11, "K": 12,
}


def card(rank, suit):
    return suit * 13 + _RAW_RANK[rank]


def make_obs(
    my_hand,
    opp_visible_cards=None,
    my_remaining_redraws=7,
    opp_remaining_redraws=7,
    my_round_wins=0,
    opp_round_wins=0,
    phase="redraw",
    round_num=1,
    my_redraw_actions_this_round=(),
    opp_redraw_counts_this_round=(),
    my_redraw_done=False,
    opp_redraw_done=False,
    opp_fold_this_round=None,
    my_fold_history=(),
    opp_fold_history=(),
    my_redraw_history=(),
    opp_redraw_history=(),
    current_player=0,
):
    if opp_visible_cards is None:
        opp_visible_cards = (None, None, None, None, None)
    return Observation(
        my_hand=tuple(my_hand),
        my_remaining_redraws=my_remaining_redraws,
        my_round_wins=my_round_wins,
        my_fold_history=tuple(my_fold_history),
        my_redraw_history=tuple(my_redraw_history),
        my_redraw_actions_this_round=tuple(my_redraw_actions_this_round),
        my_redraw_done=my_redraw_done,
        opp_visible_cards=tuple(opp_visible_cards),
        opp_remaining_redraws=opp_remaining_redraws,
        opp_round_wins=opp_round_wins,
        opp_fold_history=tuple(opp_fold_history),
        opp_redraw_history=tuple(opp_redraw_history),
        opp_redraw_counts_this_round=tuple(opp_redraw_counts_this_round),
        opp_redraw_done=opp_redraw_done,
        opp_fold_this_round=opp_fold_this_round,
        round_num=round_num,
        phase=phase,
        current_player=current_player,
    )


# ============ draw_type classification ============

def test_draw_type_open_ended_classic():
    hand = (card(5, SUIT_H), card(6, SUIT_D), card(7, SUIT_C),
            card(8, SUIT_S), card("K", SUIT_S))
    mask, draw_type = _straight_features(hand)
    assert draw_type == DRAW_OPEN_ENDED
    assert mask == (1, 1, 1, 1, 0)


def test_draw_type_gutshot_middle_gap():
    # 8,9,J,Q,3 - only T completes (rank 9 missing in {7,8,9,10} no wait)
    # 8=rank6, 9=rank7, J=rank10, Q=rank11, 3=rank1
    # Straight set {7,8,9,10,11} contains 7,10,11 (3 hits)
    # {6,7,8,9,10} contains 6,7,10 (3 hits)
    # Hmm, max 3 hits. Let me reconsider.
    # 8(6), 9(7), J(10), Q(11), 3(1)
    # {7,8,9,10,11}: hits = {7,10,11} = 3
    # No 4-in-a-row possible. Use different example.
    # Try 7,8,T,J,3:
    # 7=5, 8=6, T=8, J=9, 3=1
    # {5,6,7,8,9} contains {5,6,8,9} = 4 → missing {7}=9
    # {6,7,8,9,10} contains {6,8,9} = 3
    # Best 4 with missing rank 7 (=9). Only 1 completing rank → gutshot.
    hand = (card(7, SUIT_S), card(8, SUIT_H), card("T", SUIT_D),
            card("J", SUIT_C), card(3, SUIT_S))
    mask, draw_type = _straight_features(hand)
    assert draw_type == DRAW_GUTSHOT


def test_draw_type_wheel_open_ended():
    # 2,3,4,5,K - completes with A (wheel) or 6 → open-ended
    hand = (card(2, SUIT_S), card(3, SUIT_H), card(4, SUIT_D),
            card(5, SUIT_C), card("K", SUIT_S))
    mask, draw_type = _straight_features(hand)
    assert draw_type == DRAW_OPEN_ENDED


def test_draw_type_wheel_gutshot_only():
    # A,2,3,4,K → only 5 completes wheel; {0,1,2,3,4} missing 2 ranks → gutshot
    hand = (card("A", SUIT_S), card(2, SUIT_H), card(3, SUIT_D),
            card(4, SUIT_C), card("K", SUIT_S))
    mask, draw_type = _straight_features(hand)
    assert draw_type == DRAW_GUTSHOT


def test_draw_type_made_straight():
    # 5,6,7,8,9 mixed suits
    hand = (card(5, SUIT_S), card(6, SUIT_H), card(7, SUIT_D),
            card(8, SUIT_C), card(9, SUIT_S))
    mask, draw_type = _straight_features(hand)
    assert draw_type == DRAW_STRAIGHT_MADE
    assert mask == (1, 1, 1, 1, 1)


def test_draw_type_made_wheel():
    # A,2,3,4,5 mixed
    hand = (card("A", SUIT_S), card(2, SUIT_H), card(3, SUIT_D),
            card(4, SUIT_C), card(5, SUIT_S))
    mask, draw_type = _straight_features(hand)
    assert draw_type == DRAW_STRAIGHT_MADE


def test_draw_type_none():
    # 2,5,7,9,K - no 4-in-straight possible
    hand = (card(2, SUIT_S), card(5, SUIT_H), card(7, SUIT_D),
            card(9, SUIT_C), card("K", SUIT_S))
    mask, draw_type = _straight_features(hand)
    assert draw_type == DRAW_NONE


# ============ suited mask ============

def test_suited_mask_4_flush_draw():
    hand = (card(2, SUIT_S), card(5, SUIT_S), card(7, SUIT_S),
            card(9, SUIT_S), card("K", SUIT_H))
    dominant, mask = _suited_mask(hand)
    assert dominant == SUIT_S
    assert mask == (1, 1, 1, 1, 0)


def test_suited_mask_3_flush_draw():
    hand = (card(2, SUIT_S), card(5, SUIT_S), card(7, SUIT_S),
            card(9, SUIT_H), card("K", SUIT_D))
    dominant, mask = _suited_mask(hand)
    assert dominant == SUIT_S
    assert mask == (1, 1, 1, 0, 0)


def test_suited_mask_below_threshold():
    hand = (card(2, SUIT_S), card(5, SUIT_S), card(7, SUIT_H),
            card(9, SUIT_D), card("K", SUIT_C))
    dominant, mask = _suited_mask(hand)
    assert dominant == -1
    assert mask == (0, 0, 0, 0, 0)


def test_suited_mask_made_flush():
    hand = (card(2, SUIT_H), card(5, SUIT_H), card(7, SUIT_H),
            card(9, SUIT_H), card("K", SUIT_H))
    dominant, mask = _suited_mask(hand)
    assert dominant == SUIT_H
    assert mask == (1, 1, 1, 1, 1)


# ============ determinism ============

def test_encoding_deterministic_repeated_calls():
    obs = make_obs([card(5, SUIT_S), card(7, SUIT_H), card(9, SUIT_D),
                    card("J", SUIT_C), card("K", SUIT_S)])
    assert encode_info_set(obs) == encode_info_set(obs)


def test_encoding_deterministic_across_obs_instances():
    h = [card(5, SUIT_S), card(7, SUIT_H), card(9, SUIT_D),
         card("J", SUIT_C), card("K", SUIT_S)]
    obs1 = make_obs(h)
    obs2 = make_obs(h)
    assert encode_info_set(obs1) == encode_info_set(obs2)


# ============ canonicality (equivalent hands map to same key) ============

def test_canonical_suit_permutation_no_flush_potential():
    # Both hands: HighCard K, no flush potential, no straight potential
    # → 7-tuple features identical despite different physical suits
    hand_a = (card(2, SUIT_S), card(5, SUIT_H), card(7, SUIT_D),
              card(9, SUIT_C), card("K", SUIT_S))
    hand_b = (card(2, SUIT_H), card(5, SUIT_S), card(7, SUIT_C),
              card(9, SUIT_D), card("K", SUIT_H))
    assert encode_info_set(make_obs(hand_a)) == encode_info_set(make_obs(hand_b))


def test_canonical_same_features_different_card_identity():
    # Two hands both: HighCard K with 4 hearts + 1 spade K at position 4
    hand_a = (card(2, SUIT_H), card(5, SUIT_H), card(7, SUIT_H),
              card(9, SUIT_H), card("K", SUIT_S))
    hand_b = (card(2, SUIT_H), card(5, SUIT_H), card(7, SUIT_H),
              card(9, SUIT_H), card("K", SUIT_D))  # different K suit
    # SuitedMask same: (1,1,1,1,0). Top card same: K. Hand rank same: HighCard.
    # Straight features same (no straight potential).
    # Suit alignment for opp face-up depends on dominant; both hands have dominant=H.
    # No opp face-up in obs → no difference.
    assert encode_info_set(make_obs(hand_a)) == encode_info_set(make_obs(hand_b))


# ============ completeness (different strategic situations -> different keys) ============

def test_open_ended_vs_gutshot_different_keys():
    hand_open = (card(5, SUIT_H), card(6, SUIT_D), card(7, SUIT_C),
                 card(8, SUIT_S), card("K", SUIT_S))
    hand_gut = (card(7, SUIT_S), card(8, SUIT_H), card("T", SUIT_D),
                card("J", SUIT_C), card(3, SUIT_S))
    assert encode_info_set(make_obs(hand_open)) != encode_info_set(make_obs(hand_gut))


def test_different_phases_different_keys():
    h = [card(5, SUIT_S), card(7, SUIT_H), card(9, SUIT_D),
         card("J", SUIT_C), card("K", SUIT_S)]
    assert encode_info_set(make_obs(h, phase="fold")) != \
        encode_info_set(make_obs(h, phase="redraw"))


def test_different_remaining_redraws_different_keys():
    h = [card(5, SUIT_S), card(7, SUIT_H), card(9, SUIT_D),
         card("J", SUIT_C), card("K", SUIT_S)]
    assert encode_info_set(make_obs(h, my_remaining_redraws=7)) != \
        encode_info_set(make_obs(h, my_remaining_redraws=3))


def test_different_round_wins_different_keys():
    h = [card(5, SUIT_S), card(7, SUIT_H), card(9, SUIT_D),
         card("J", SUIT_C), card("K", SUIT_S)]
    assert encode_info_set(make_obs(h, my_round_wins=0, opp_round_wins=0)) != \
        encode_info_set(make_obs(h, my_round_wins=1, opp_round_wins=0))


# ============ opp face-up encoding ============

def test_opp_face_up_alignment_affects_key():
    # I have 4 hearts (dominant = H)
    my_hand = [card(2, SUIT_H), card(5, SUIT_H), card(7, SUIT_H),
               card(9, SUIT_H), card("K", SUIT_S)]
    opp_aligned = (None, None, card(3, SUIT_H), card(8, SUIT_H), card("J", SUIT_H))
    opp_not_aligned = (None, None, card(3, SUIT_C), card(8, SUIT_C), card("J", SUIT_C))
    assert encode_info_set(make_obs(my_hand, opp_visible_cards=opp_aligned)) != \
        encode_info_set(make_obs(my_hand, opp_visible_cards=opp_not_aligned))


def test_opp_face_up_order_independent():
    # Same 3 face-up cards, different positions in opp_visible_cards
    # (positions 2,3,4 are face-up — reorder within them)
    my_hand = [card(2, SUIT_S), card(5, SUIT_H), card(7, SUIT_D),
               card(9, SUIT_C), card("K", SUIT_S)]
    opp_v1 = (None, None, card(3, SUIT_H), card(8, SUIT_C), card("J", SUIT_D))
    opp_v2 = (None, None, card("J", SUIT_D), card(3, SUIT_H), card(8, SUIT_C))
    assert encode_info_set(make_obs(my_hand, opp_visible_cards=opp_v1)) == \
        encode_info_set(make_obs(my_hand, opp_visible_cards=opp_v2))


# ============ hidden info not leaked ============

def test_opp_hole_none_does_not_break_encoding():
    # Standard observation has opp_visible_cards[0]/[1] = None (hole hidden)
    my_hand = [card(2, SUIT_S), card(5, SUIT_H), card(7, SUIT_D),
               card(9, SUIT_C), card("K", SUIT_S)]
    opp_v = (None, None, card(3, SUIT_H), card(8, SUIT_C), card("J", SUIT_D))
    key = encode_info_set(make_obs(my_hand, opp_visible_cards=opp_v))
    assert isinstance(key, str) and len(key) > 0


def test_encoding_via_real_game():
    """Smoke test: real PokerGame Observation produces valid info set string."""
    game = PokerGame(seed=42)
    for p in (0, 1):
        obs = game.observation(p)
        key = encode_info_set(obs)
        assert isinstance(key, str) and len(key) > 0


def test_two_games_different_seeds_different_keys():
    g1 = PokerGame(seed=1)
    g2 = PokerGame(seed=2)
    k1 = encode_info_set(g1.observation(0))
    k2 = encode_info_set(g2.observation(0))
    # Different deal -> different my_hand -> different key (high probability)
    assert k1 != k2


# ============ v4 coarse abstraction (convergence redesign) ============

from cfr.agent.info_set import _opp_face_up_bucket


def test_opp_bucket_counts_aligned_and_max_rank():
    # my dominant suit = Spades(0). opp face-up: 2 spades + 1 club, max rank K(post-adj=11).
    fu = (None, None, card(2, SUIT_S), card(7, SUIT_S), card("K", SUIT_C))
    assert _opp_face_up_bucket(fu, my_dominant_suit=SUIT_S) == (2, 11 // 4)


def test_opp_bucket_no_visible_cards():
    assert _opp_face_up_bucket((None, None, None, None, None), my_dominant_suit=0) == (0, -1)


def test_top_card_no_longer_splits_keys():
    # Two HighCard hands, same hand_rank + same draw_type, different top card.
    # v4 drops top_card → SAME key.
    h_kicker_k = [card(2, SUIT_S), card(5, SUIT_H), card(7, SUIT_D),
                  card(9, SUIT_C), card("K", SUIT_S)]
    h_kicker_q = [card(2, SUIT_S), card(5, SUIT_H), card(7, SUIT_D),
                  card(9, SUIT_C), card("Q", SUIT_S)]
    assert encode_info_set(make_obs(h_kicker_k)) == encode_info_set(make_obs(h_kicker_q))


def test_my_remaining_redraws_kept_exact():
    # legal-action consistency: my budget 7 vs 4 must stay distinct keys.
    h = [card(2, SUIT_S), card(5, SUIT_H), card(7, SUIT_D),
         card(9, SUIT_C), card("K", SUIT_S)]
    assert encode_info_set(make_obs(h, my_remaining_redraws=7)) != \
        encode_info_set(make_obs(h, my_remaining_redraws=4))


def test_opp_remaining_redraws_capped_at_4():
    # opp budget does NOT affect my legal actions → capped: 5,6,7 collapse.
    h = [card(2, SUIT_S), card(5, SUIT_H), card(7, SUIT_D),
         card(9, SUIT_C), card("K", SUIT_S)]
    assert encode_info_set(make_obs(h, opp_remaining_redraws=7)) == \
        encode_info_set(make_obs(h, opp_remaining_redraws=5))
    # but below the cap still splits:
    assert encode_info_set(make_obs(h, opp_remaining_redraws=7)) != \
        encode_info_set(make_obs(h, opp_remaining_redraws=3))


def test_redraw_history_does_not_split_keys():
    # v4 drops this-round redraw detail → these now collapse.
    h = [card(2, SUIT_S), card(5, SUIT_H), card(7, SUIT_D),
         card(9, SUIT_C), card("K", SUIT_S)]
    assert encode_info_set(make_obs(h, my_redraw_actions_this_round=())) == \
        encode_info_set(make_obs(h, my_redraw_actions_this_round=(16,)))
