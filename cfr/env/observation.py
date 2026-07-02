from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Observation:
    """
    What a player can see at a given decision point.

    Hidden information is physically absent (None / masked) — never just flagged.
    This is the only safe way to prevent CFR info-set construction from leaking
    opponent's private state.

    Iterative redraw note:
        my_redraw_actions_this_round  — my own bitmask sequence this round
                                        (I see full bitmasks since I chose them)
        opp_redraw_counts_this_round  — opp's count sequence this round, only
                                        finalized iterations included
                                        (pending current-iter action is hidden)
    """

    # Self — full visibility
    my_hand: Tuple[int, ...]                                   # 5 cards
    my_remaining_redraws: int
    my_round_wins: int
    my_fold_history: Tuple[bool, ...]                          # past completed rounds
    my_redraw_history: Tuple[Tuple[int, ...], ...]             # past rounds: iter bitmask sequences
    my_redraw_actions_this_round: Tuple[int, ...]              # current round committed + my pending
    my_redraw_done: bool                                       # already STOP this round

    # Opponent — only what's public
    opp_visible_cards: Tuple[Optional[int], ...]               # length 5, None at hidden slots
    opp_remaining_redraws: int
    opp_round_wins: int
    opp_fold_history: Tuple[bool, ...]
    opp_redraw_history: Tuple[Tuple[int, ...], ...]            # past rounds: opp's iter count sequences
    opp_redraw_counts_this_round: Tuple[int, ...]              # current round committed counts only
    opp_redraw_done: bool
    opp_fold_this_round: Optional[bool]                        # None until both committed fold

    # Game state
    round_num: int                                             # 1-indexed
    phase: str                                                 # 'fold' or 'redraw'
    current_player: int                                        # who decides right now (0 or 1)
