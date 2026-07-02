"""
PokerGame: 5-card draw poker, BO3, hidden hole cards, fold-before-redraw,
iterative redraw with explicit STOP.

Game tree (per round):
    FOLD phase (simultaneous, modeled sequential P1→P2 with hidden info)
        P1 picks play/fold → stored, not visible to P2
        P2 picks play/fold → both reveal
    If any fold: resolve immediately, no redraw, no budget spent
    Else REDRAW phase (iterative until both STOP):
        loop until both done:
            if P1 active: P1 picks action (0=STOP, 1..31=bitmask)
            if P2 active: P2 picks action; in same iter doesn't see P1's pending
            iter finalizes: cards drawn, counts revealed
        Showdown: compare hands
"""

import random
from enum import Enum
from typing import List, Optional, Tuple

from .observation import Observation
from .poker_rules import compare_hands

HOLE_INDICES = (0, 1)
FACE_UP_INDICES = (2, 3, 4)
NUM_PLAYERS = 2
HAND_SIZE = 5
DECK_SIZE = 52
INITIAL_REDRAWS = 7
MAX_ROUNDS = 3
REDRAW_ACTION_SPACE = 32                  # 2^HAND_SIZE
REDRAW_ACTION_STOP = 0                    # in REDRAW phase, 0 means STOP
FOLD_ACTION_PLAY = 0
FOLD_ACTION_FOLD = 1


class Phase(Enum):
    FOLD = 'fold'
    REDRAW = 'redraw'
    TERMINAL = 'terminal'


class PokerGame:
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self.reset()

    # ---------- lifecycle ----------

    def reset(self) -> None:
        self.round_num = 1
        self.round_wins = [0, 0]
        self.remaining_redraws = [INITIAL_REDRAWS, INITIAL_REDRAWS]
        # Past rounds (completed):
        self.fold_history: List[List[bool]] = [[], []]
        # Per round, list of bitmask actions (full sequence including STOP at end)
        self.redraw_history: List[List[Tuple[int, ...]]] = [[], []]
        self._terminal = False
        self._start_round()

    def _start_round(self) -> None:
        deck = list(range(DECK_SIZE))
        self._rng.shuffle(deck)
        self.hands = [deck[0:HAND_SIZE], deck[HAND_SIZE:2 * HAND_SIZE]]
        self._deck = deck
        self._deck_pointer = 2 * HAND_SIZE

        self.fold_this_round: List[Optional[bool]] = [None, None]
        # Public, finalized iter actions for this round (bitmasks)
        self.redraw_actions_committed: List[List[int]] = [[], []]
        self.redraw_done: List[bool] = [False, False]
        # Pending current iter (hidden from opp until finalized)
        self._pending_redraw: List[Optional[int]] = [None, None]

        self._fold_revealed = False
        self._showdown_revealed = False

        self._waiting_for = 0  # P1 always first
        self.phase = Phase.FOLD

    # ---------- CFR interface ----------

    def current_player(self) -> int:
        if self._terminal:
            return -1
        return self._waiting_for

    def is_terminal(self) -> bool:
        return self._terminal

    def is_chance(self) -> bool:
        return False  # randomness baked into reset / _start_round / _finalize_iter

    def legal_actions(self) -> List[int]:
        if self._terminal:
            return []
        player = self.current_player()
        if self.phase == Phase.FOLD:
            return [FOLD_ACTION_PLAY, FOLD_ACTION_FOLD]
        if self.phase == Phase.REDRAW:
            cap = self.remaining_redraws[player]
            # action 0 (STOP) always legal; redraws filtered by budget
            return [a for a in range(REDRAW_ACTION_SPACE)
                    if bin(a).count('1') <= cap]
        return []

    def apply(self, action: int) -> None:
        if self._terminal:
            raise RuntimeError("apply() on terminal state")
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action} in phase {self.phase}")

        player = self.current_player()

        if self.phase == Phase.FOLD:
            self.fold_this_round[player] = (action == FOLD_ACTION_FOLD)
            if self._waiting_for == 0:
                self._waiting_for = 1
            else:
                self._fold_revealed = True
                if any(self.fold_this_round):
                    self._resolve_round()
                else:
                    self.phase = Phase.REDRAW
                    self._start_redraw_iter()

        elif self.phase == Phase.REDRAW:
            self._pending_redraw[player] = action

            other = 1 - player
            if (not self.redraw_done[other]) and self._pending_redraw[other] is None:
                # Wait for the other player in this iter
                self._waiting_for = other
            else:
                # Iter complete (other already committed this iter, or other is done)
                self._finalize_iter()

    def utility(self, player: int) -> float:
        if not self._terminal:
            raise RuntimeError("utility() on non-terminal state")
        opp = 1 - player
        if self.round_wins[player] > self.round_wins[opp]:
            return 1.0
        if self.round_wins[player] < self.round_wins[opp]:
            return -1.0
        return 0.0

    # ---------- internal transitions ----------

    def _start_redraw_iter(self) -> None:
        """Begin a new redraw iteration. Pick whose turn first."""
        self._pending_redraw = [None, None]
        if not self.redraw_done[0]:
            self._waiting_for = 0
        else:
            self._waiting_for = 1  # P0 done, P1 must still be active

    def _finalize_iter(self) -> None:
        """Apply both players' pending actions for this iter, reveal, advance."""
        for p in range(NUM_PLAYERS):
            action = self._pending_redraw[p]
            if action is None:
                continue  # this player was already done before this iter

            self.redraw_actions_committed[p].append(action)

            if action == REDRAW_ACTION_STOP:
                self.redraw_done[p] = True
            else:
                indices = [i for i in range(HAND_SIZE) if (action >> i) & 1]
                for idx in indices:
                    self.hands[p][idx] = self._deck[self._deck_pointer]
                    self._deck_pointer += 1
                self.remaining_redraws[p] -= len(indices)

        self._pending_redraw = [None, None]

        if all(self.redraw_done):
            self._showdown_revealed = True
            self._resolve_round()
        else:
            self._start_redraw_iter()

    def _resolve_round(self) -> None:
        f0, f1 = self.fold_this_round
        if f0 and f1:
            pass  # push
        elif f0:
            self.round_wins[1] += 1
        elif f1:
            self.round_wins[0] += 1
        else:
            result = compare_hands(self.hands[0], self.hands[1])
            if result == 1:
                self.round_wins[0] += 1
            elif result == 2:
                self.round_wins[1] += 1

        for p in range(NUM_PLAYERS):
            self.fold_history[p].append(self.fold_this_round[p])
            self.redraw_history[p].append(tuple(self.redraw_actions_committed[p]))

        if (self.round_wins[0] >= 2 or self.round_wins[1] >= 2
                or self.round_num >= MAX_ROUNDS):
            self._terminal = True
            self.phase = Phase.TERMINAL
        else:
            self.round_num += 1
            self._start_round()

    # ---------- observation ----------

    def observation(self, player: int) -> Observation:
        opp = 1 - player

        opp_cards: List[Optional[int]] = []
        for i in range(HAND_SIZE):
            if i in FACE_UP_INDICES:
                opp_cards.append(self.hands[opp][i])
            elif self._terminal and self._showdown_revealed:
                opp_cards.append(self.hands[opp][i])
            else:
                opp_cards.append(None)

        # My this-round actions: committed + pending (I know my own)
        my_actions = list(self.redraw_actions_committed[player])
        if self._pending_redraw[player] is not None:
            my_actions.append(self._pending_redraw[player])

        # Opp this-round: only finalized (no pending leak)
        opp_counts = tuple(bin(a).count('1') for a in self.redraw_actions_committed[opp])

        # Past rounds: opp history shown as counts only
        opp_past_history = tuple(
            tuple(bin(a).count('1') for a in round_seq)
            for round_seq in self.redraw_history[opp]
        )
        my_past_history = tuple(self.redraw_history[player])

        opp_fold = self.fold_this_round[opp] if self._fold_revealed else None

        return Observation(
            my_hand=tuple(self.hands[player]),
            my_remaining_redraws=self.remaining_redraws[player],
            my_round_wins=self.round_wins[player],
            my_fold_history=tuple(self.fold_history[player]),
            my_redraw_history=my_past_history,
            my_redraw_actions_this_round=tuple(my_actions),
            my_redraw_done=self.redraw_done[player],
            opp_visible_cards=tuple(opp_cards),
            opp_remaining_redraws=self.remaining_redraws[opp],
            opp_round_wins=self.round_wins[opp],
            opp_fold_history=tuple(self.fold_history[opp]),
            opp_redraw_history=opp_past_history,
            opp_redraw_counts_this_round=opp_counts,
            opp_redraw_done=self.redraw_done[opp],
            opp_fold_this_round=opp_fold,
            round_num=self.round_num,
            phase=self.phase.value,
            current_player=self.current_player(),
        )
