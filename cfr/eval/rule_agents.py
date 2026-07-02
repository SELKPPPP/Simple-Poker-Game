"""Rule-based baseline agents for evaluating the CFR agent.

All agents implement act(game, player) -> int and return a legal action.
They read only from game.observation(player) (the same hidden-info-safe view
CFR sees), so they cannot accidentally peek at the opponent's hole cards.

Design notes (per Phase 3 fold-dominance finding, see 项目知识库 §4.5.5):
    fold is weakly dominated by play+STOP in this game. Tight and AlwaysCall
    deliberately expose this exploit so that vs-CFR win rate becomes a
    sanity check for whether CFR learned to never-fold.
"""

import random
from collections import Counter
from typing import List

from ..env.game import (
    PokerGame, Phase,
    FOLD_ACTION_PLAY, FOLD_ACTION_FOLD,
    REDRAW_ACTION_STOP, HAND_SIZE,
)
from ..env.poker_rules import evaluate_hand, get_rank_suit


def _junk_indices(hand: List[int]) -> List[int]:
    """Indices of cards not in any pair/3oak and not in 4+ same-suit group."""
    ranks = [get_rank_suit(c)[0] for c in hand]
    suits = [get_rank_suit(c)[1] for c in hand]
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)
    dom_suit, dom_count = suit_counts.most_common(1)[0]
    return [
        i for i in range(HAND_SIZE)
        if rank_counts[ranks[i]] < 2 and not (dom_count >= 4 and suits[i] == dom_suit)
    ]


def _redraw_action_from_indices(indices: List[int], budget: int) -> int:
    """Build legal bitmask from desired indices, capped at budget. 0 means STOP."""
    if not indices or budget <= 0:
        return REDRAW_ACTION_STOP
    return sum(1 << i for i in indices[:budget])


class RandomAgent:
    """Uniform random over legal actions. Weakest baseline."""
    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def act(self, game: PokerGame, player: int) -> int:
        return self._rng.choice(game.legal_actions())


class AlwaysCallAgent:
    """Never fold, never redraw — always action 0 (PLAY in FOLD, STOP in REDRAW).

    Per §4.5.5 this is actually a STRONG baseline against fold-heavy opponents.
    """
    name = "always_call"

    def act(self, game: PokerGame, player: int) -> int:
        return 0


class TightAgent:
    """Fold high-card hands; in REDRAW, throw only junk cards."""
    name = "tight"

    def act(self, game: PokerGame, player: int) -> int:
        obs = game.observation(player)
        hand_rank = evaluate_hand(list(obs.my_hand))[0]

        if game.phase == Phase.FOLD:
            return FOLD_ACTION_FOLD if hand_rank == 0 else FOLD_ACTION_PLAY

        if hand_rank >= 1:  # already have a pair -> sit on it
            return REDRAW_ACTION_STOP
        return _redraw_action_from_indices(
            _junk_indices(list(obs.my_hand)),
            obs.my_remaining_redraws,
        )


class LooseAgent:
    """Never fold, redraw aggressively until 3-of-a-kind+."""
    name = "loose"

    def act(self, game: PokerGame, player: int) -> int:
        if game.phase == Phase.FOLD:
            return FOLD_ACTION_PLAY

        obs = game.observation(player)
        hand_rank = evaluate_hand(list(obs.my_hand))[0]
        if hand_rank >= 3:
            return REDRAW_ACTION_STOP
        return _redraw_action_from_indices(
            _junk_indices(list(obs.my_hand)),
            obs.my_remaining_redraws,
        )


class HeuristicAgent:
    """Most human-like: fold only when very weak AND opp shows strength;
    in REDRAW, draw-aware + budget-conserving (keep 1 redraw for later rounds).
    """
    name = "heuristic"

    def act(self, game: PokerGame, player: int) -> int:
        obs = game.observation(player)
        my_hand = list(obs.my_hand)
        hand_rank = evaluate_hand(my_hand)[0]

        if game.phase == Phase.FOLD:
            opp_visible_ranks = [
                get_rank_suit(c)[0] for c in obs.opp_visible_cards if c is not None
            ]
            opp_max = max(opp_visible_ranks) if opp_visible_ranks else 0
            if hand_rank == 0 and opp_max >= 10:  # opp shows Q+
                return FOLD_ACTION_FOLD
            return FOLD_ACTION_PLAY

        if hand_rank >= 2:  # two pair or better -> hold
            return REDRAW_ACTION_STOP
        # Save 1 redraw for future rounds if we have multiple rounds left
        rounds_left = 3 - (obs.my_round_wins + obs.opp_round_wins) - 1
        reserve = 1 if rounds_left >= 1 else 0
        usable = max(0, obs.my_remaining_redraws - reserve)
        junk = _junk_indices(my_hand)
        return _redraw_action_from_indices(junk[:2], usable)
