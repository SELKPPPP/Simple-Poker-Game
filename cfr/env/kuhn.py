"""Kuhn Poker (3-card toy poker) — used as MCCFR correctness validation.

Rules:
    Deck: J, Q, K (ranks 0, 1, 2)
    Each player gets 1 card; antes 1 each (pot starts at 2 chips).
    Action sequence:
        P0 acts: check (0) or bet (1)
        P1 acts: check/fold (0) or bet/call (1)
        If history is [check, bet]: P0 acts again: fold (0) or call (1)
    Showdown: higher card wins the pot.

Net utility per player (excluding ante since both forfeit it):
    Both check       -> winner gets +1, loser -1
    One folds        -> non-folder +1, folder -1
    Both call (bet)  -> winner +2, loser -2

Info sets total: 12 (3 cards × 4 history prefixes: '', '0', '1', '01').
Known Nash exploitability = 0 with game value -1/18 to P0.
"""

import random
from typing import List, Optional, Tuple

CHECK_FOLD = 0
BET_CALL = 1

JACK = 0
QUEEN = 1
KING = 2
ALL_CARDS = (JACK, QUEEN, KING)

_TERMINAL_HISTORIES = {
    (CHECK_FOLD, CHECK_FOLD): "showdown_1",
    (CHECK_FOLD, BET_CALL, CHECK_FOLD): "p0_fold",
    (CHECK_FOLD, BET_CALL, BET_CALL): "showdown_2",
    (BET_CALL, CHECK_FOLD): "p1_fold",
    (BET_CALL, BET_CALL): "showdown_2",
}


class KuhnGame:
    def __init__(
        self,
        hands: Optional[Tuple[int, int]] = None,
        seed: Optional[int] = None,
    ):
        if hands is None:
            rng = random.Random(seed)
            cards = list(ALL_CARDS)
            rng.shuffle(cards)
            hands = (cards[0], cards[1])
        self.hands: List[int] = list(hands)
        self.history: List[int] = []
        self._terminal = False

    def current_player(self) -> int:
        return len(self.history) % 2

    def is_terminal(self) -> bool:
        return self._terminal

    def is_chance(self) -> bool:
        return False

    def legal_actions(self) -> List[int]:
        if self._terminal:
            return []
        return [CHECK_FOLD, BET_CALL]

    def apply(self, action: int) -> None:
        if self._terminal:
            raise RuntimeError("apply() on terminal state")
        if action not in (CHECK_FOLD, BET_CALL):
            raise ValueError(f"illegal action {action}")
        self.history.append(action)
        if tuple(self.history) in _TERMINAL_HISTORIES:
            self._terminal = True

    def utility(self, player: int) -> float:
        if not self._terminal:
            raise RuntimeError("utility() on non-terminal state")
        kind = _TERMINAL_HISTORIES[tuple(self.history)]
        if kind == "showdown_1":
            winner = 0 if self.hands[0] > self.hands[1] else 1
            return 1.0 if winner == player else -1.0
        if kind == "showdown_2":
            winner = 0 if self.hands[0] > self.hands[1] else 1
            return 2.0 if winner == player else -2.0
        if kind == "p0_fold":
            return -1.0 if player == 0 else 1.0
        if kind == "p1_fold":
            return 1.0 if player == 0 else -1.0
        raise RuntimeError(f"unhandled terminal kind: {kind}")

    def info_set_key(self, player: int) -> str:
        return f"{self.hands[player]}:{''.join(map(str, self.history))}"
