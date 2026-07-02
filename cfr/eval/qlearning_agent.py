"""Legacy Q-learning bot adapter for the new imperfect-information PokerGame.

The legacy bot (backend/model_lib/q_table.pkl) was trained on the old
perfect-information game: one redraw decision per round, no fold phase, no
iterative STOP. Adapter policy:
    - FOLD phase: always PLAY (the old game had no fold concept).
    - REDRAW phase: first decision of the round -> greedy argmax over the
      LEGAL subset of its 32 q-values (unseen state -> STOP); afterwards ->
      STOP (the old game allowed exactly one redraw decision per round).

State encoding replicates backend/utils_train.py::get_state exactly; the two
poker_rules modules are identical, so features match bit-for-bit.
"""

import pickle
from collections import Counter
from typing import List

from ..env.game import PokerGame, Phase, FOLD_ACTION_PLAY, REDRAW_ACTION_STOP
from ..env.poker_rules import evaluate_hand, get_rank_suit

DEFAULT_Q_TABLE_PATH = "backend/model_lib/q_table.pkl"


def legacy_state(hand: List[int], remaining_redraws: int,
                 my_score: int, opp_score: int) -> tuple:
    """Replica of backend/utils_train.py::get_state.

    The original was called as get_state(hand, redraws, p1_score, p2_score)
    with the agent seated as P2 and returns (..., p2_score, p1_score) — i.e.
    the tail is (remaining_redraws, my_score, opp_score).
    """
    score = evaluate_hand(hand)
    hand_rank = score[0]
    top_card = score[1][0] if score[1] else 0

    suits = []
    ranks = []
    for card in hand:
        r, s = get_rank_suit(card)
        suits.append(s)
        ranks.append(r)

    suit_counts = Counter(suits)
    most_common = suit_counts.most_common()
    dominant_suit = -1
    max_count = 0
    if most_common:
        dominant_suit = most_common[0][0]
        max_count = most_common[0][1]

    suited_mask = []
    for s in suits:
        if s == dominant_suit and max_count >= 3:
            suited_mask.append(1)
        else:
            suited_mask.append(0)

    # Calculate Straight Mask
    straight_sets = [set(range(i, i + 5)) for i in range(9)]
    straight_sets.append({12, 0, 1, 2, 3})  # Wheel: A, 2, 3, 4, 5

    best_straight_mask = [0] * 5
    max_straight_count = 0

    for s_set in straight_sets:
        unique_hits = set()
        for r in ranks:
            if r in s_set:
                unique_hits.add(r)

        unique_count = len(unique_hits)

        if unique_count > max_straight_count:
            max_straight_count = unique_count
            best_straight_mask = [1 if r in s_set else 0 for r in ranks]

    if max_straight_count < 3:
        best_straight_mask = [0] * 5

    return (hand_rank, top_card, tuple(suited_mask),
            tuple(best_straight_mask), remaining_redraws, my_score, opp_score)


class QTableAgent:
    """Greedy legacy Q-learning bot playing through the new game's interface."""
    name = "qlearning"

    def __init__(self, q_table_path: str = DEFAULT_Q_TABLE_PATH):
        with open(q_table_path, "rb") as f:
            self._q = pickle.load(f)

    def act(self, game: PokerGame, player: int) -> int:
        if game.phase == Phase.FOLD:
            return FOLD_ACTION_PLAY

        obs = game.observation(player)
        if obs.my_redraw_actions_this_round:
            return REDRAW_ACTION_STOP  # old game: one decision per round

        state = legacy_state(list(obs.my_hand), obs.my_remaining_redraws,
                             obs.my_round_wins, obs.opp_round_wins)
        row = self._q.get(state)
        if row is None:
            return REDRAW_ACTION_STOP  # unseen state
        # np.argmax semantics (first max) over the legal subset only
        return max(game.legal_actions(), key=lambda a: float(row[a]))
