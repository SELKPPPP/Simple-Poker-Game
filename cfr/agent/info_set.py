"""
Info set encoding: Observation -> canonical string key for CFR tables.

Design: 7-tuple skeleton from v1 (HandRank, TopCard, SuitedMask, StraightMask)
+ v2/v3 extensions. StraightMask augmented with draw_type bit distinguishing
open-ended (~17% hit) vs gutshot (~8.5% hit) for chase-depth strategy.

Opp face-up suit encoded relative to my dominant suit (same/other) to preserve
"opponent is also chasing flush" signal.

Hidden info (opp hole cards, opp's pending current-iter action) is already
stripped at the Observation layer; this module trusts that stripping.
"""

from collections import Counter
from typing import List, Optional, Tuple

from ..env.observation import Observation
from ..env.poker_rules import evaluate_hand, get_rank_suit

DRAW_NONE = 0
DRAW_OPEN_ENDED = 1
DRAW_GUTSHOT = 2
DRAW_STRAIGHT_MADE = 3

# All 10 5-card straight rank sets (post-adjustment ranks: 2=0..A=12)
_STRAIGHT_SETS: List[frozenset] = [frozenset(range(i, i + 5)) for i in range(9)]
_STRAIGHT_SETS.append(frozenset({12, 0, 1, 2, 3}))  # wheel: A,2,3,4,5


def encode_info_set(obs: Observation) -> str:
    """v4 coarse abstraction (~35K reachable info sets).

    Collapses hand to (hand_rank, draw_type), buckets opp face-up to
    (num_aligned, max_rank//4), caps opp budget at 4. my_remaining_redraws
    stays exact for legal-action consistency. See Improvement/收敛优化设计.md.
    """
    hand_rank, _ = _hand_rank_and_top(obs.my_hand)
    dominant_suit, _ = _suited_mask(obs.my_hand)
    _, draw_type = _straight_features(obs.my_hand)
    opp_bucket = _opp_face_up_bucket(obs.opp_visible_cards, dominant_suit)

    key = (
        hand_rank, draw_type,
        opp_bucket,
        obs.my_remaining_redraws,            # exact (legal-action consistency)
        min(obs.opp_remaining_redraws, 4),   # capped (opp budget not in my legal set)
        obs.my_round_wins, obs.opp_round_wins,
        obs.phase, obs.round_num,
    )
    return repr(key)


def _hand_rank_and_top(hand: Tuple[int, ...]) -> Tuple[int, int]:
    score = evaluate_hand(list(hand))
    return score[0], (score[1][0] if score[1] else 0)


def _suited_mask(hand: Tuple[int, ...]) -> Tuple[int, Tuple[int, ...]]:
    """>=3 cards same suit -> mark those positions. Returns (dominant_suit, mask).
    dominant_suit = -1 if no suit reaches 3.
    """
    suits = [get_rank_suit(c)[1] for c in hand]
    counts = Counter(suits)
    most = counts.most_common(1)
    if not most or most[0][1] < 3:
        return -1, (0, 0, 0, 0, 0)
    dominant = most[0][0]
    mask = tuple(1 if s == dominant else 0 for s in suits)
    return dominant, mask


def _straight_features(hand: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """Return (straight_mask, draw_type).

    straight_mask: 5-tuple marking positions whose rank lies in best partial straight set.
    draw_type: 0=NONE, 1=OPEN_ENDED, 2=GUTSHOT, 3=STRAIGHT_MADE.

    OPEN_ENDED iff >=2 distinct ranks would complete some 5-card straight.
    """
    ranks = [get_rank_suit(c)[0] for c in hand]
    rank_set = set(ranks)

    # Made straight: any straight set fully covered
    for s_set in _STRAIGHT_SETS:
        if len(rank_set & s_set) == 5:
            mask = tuple(1 if r in s_set else 0 for r in ranks)
            return mask, DRAW_STRAIGHT_MADE

    # Best partial straight (max unique hits)
    best_mask: Tuple[int, ...] = (0, 0, 0, 0, 0)
    best_unique = 0
    for s_set in _STRAIGHT_SETS:
        hits = len(rank_set & s_set)
        if hits > best_unique:
            best_unique = hits
            best_mask = tuple(1 if r in s_set else 0 for r in ranks)

    if best_unique < 4:
        return (0, 0, 0, 0, 0) if best_unique < 3 else best_mask, DRAW_NONE

    # 4 in some set -> count distinct completing ranks across all sets
    completing: set = set()
    for s_set in _STRAIGHT_SETS:
        missing = s_set - rank_set
        if len(missing) == 1:
            completing.update(missing)
    return best_mask, (DRAW_OPEN_ENDED if len(completing) >= 2 else DRAW_GUTSHOT)


def _opp_face_up_bucket(
    opp_visible_cards: Tuple[Optional[int], ...],
    my_dominant_suit: int,
) -> Tuple[int, int]:
    """Coarse bucket of opp face-up cards: (num_aligned, max_rank // 4).

    num_aligned = count of visible cards sharing my dominant suit (flush-chase
    signal). max_rank bucketed to 4 bands; -1 when nothing visible.
    """
    num_aligned = 0
    max_rank = -1
    for c in opp_visible_cards:
        if c is None:
            continue
        rank, suit = get_rank_suit(c)
        if my_dominant_suit != -1 and suit == my_dominant_suit:
            num_aligned += 1
        if rank > max_rank:
            max_rank = rank
    rank_bucket = max_rank // 4 if max_rank >= 0 else -1
    return (num_aligned, rank_bucket)
