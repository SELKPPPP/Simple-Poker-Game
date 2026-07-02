"""Interactive CLI: human vs CFR agent.

Usage:
    python -m cfr.scripts.play_human --checkpoint cfr/checkpoints/<file>.pkl
    python -m cfr.scripts.play_human                      # CFR plays from empty table (uniform random)

Run with --help for all options.
"""

import argparse
import os
import sys
from typing import List, Optional

# Allow `python cfr/scripts/play_human.py ...` without -m.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cfr.agent.regret_table import RegretTable
from cfr.env.game import (
    PokerGame, Phase,
    FOLD_ACTION_PLAY, FOLD_ACTION_FOLD,
    REDRAW_ACTION_STOP, HAND_SIZE,
)
from cfr.env.poker_rules import get_rank_suit
from cfr.eval.cfr_agent import CFRAgent

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS_ASCII = ["S", "H", "D", "C"]


def card_str(card: Optional[int]) -> str:
    if card is None:
        return "[??]"
    suit = card // 13
    raw_rank = card % 13
    return f"{RANKS[raw_rank]:>2}{SUITS_ASCII[suit]}"


def render_hand(cards, show_indices: bool = False) -> str:
    parts = [card_str(c) for c in cards]
    if show_indices:
        labels = [f"[{i}]" for i in range(len(cards))]
        return "  ".join(f"{l}{p}" for l, p in zip(labels, parts))
    return "  ".join(parts)


def print_state(game: PokerGame, human_seat: int) -> None:
    obs = game.observation(human_seat)
    print()
    print("=" * 60)
    print(f"Round {obs.round_num}    Phase: {obs.phase}    "
          f"Score: you {obs.my_round_wins} - {obs.opp_round_wins} AI")
    print(f"Budget: you {obs.my_remaining_redraws} redraws, "
          f"AI {obs.opp_remaining_redraws} redraws")
    print(f"Opp face-up: {render_hand(obs.opp_visible_cards)}")
    print(f"Your hand:   {render_hand(obs.my_hand, show_indices=True)}")
    if obs.my_redraw_actions_this_round:
        seq = [bin(a).count('1') for a in obs.my_redraw_actions_this_round]
        print(f"Your redraws this round: {seq}  (last=0 means STOP)")
    if obs.opp_redraw_counts_this_round:
        print(f"AI redraws this round:   {list(obs.opp_redraw_counts_this_round)}")


def prompt_fold_action() -> int:
    while True:
        ans = input("Action [p]lay or [f]old? ").strip().lower()
        if ans in ("p", "play", ""):
            return FOLD_ACTION_PLAY
        if ans in ("f", "fold"):
            return FOLD_ACTION_FOLD
        print("  -> enter 'p' or 'f'.")


def prompt_redraw_action(budget: int) -> int:
    print(f"  Pick card indices to redraw (0-{HAND_SIZE-1}), comma-separated;")
    print(f"  blank or 'stop' to STOP. Budget remaining: {budget}.")
    while True:
        ans = input("Redraw> ").strip().lower()
        if ans in ("", "stop", "s"):
            return REDRAW_ACTION_STOP
        try:
            indices = sorted({int(x) for x in ans.replace(" ", "").split(",") if x})
        except ValueError:
            print("  -> invalid, use e.g. '0,2,4' or 'stop'.")
            continue
        if any(i < 0 or i >= HAND_SIZE for i in indices):
            print(f"  -> indices must be 0..{HAND_SIZE-1}.")
            continue
        if len(indices) > budget:
            print(f"  -> not enough budget ({budget} left, {len(indices)} requested).")
            continue
        return sum(1 << i for i in indices)


def human_act(game: PokerGame, human_seat: int) -> int:
    print_state(game, human_seat)
    obs = game.observation(human_seat)
    if game.phase == Phase.FOLD:
        return prompt_fold_action()
    return prompt_redraw_action(obs.my_remaining_redraws)


def run_match(cfr_agent: CFRAgent, human_seat: int, game_seed: int) -> None:
    game = PokerGame(seed=game_seed)
    cfr_seat = 1 - human_seat
    print(f"\n>>> You are seat {human_seat}, AI is seat {cfr_seat}.")

    while not game.is_terminal():
        cur = game.current_player()
        if cur == human_seat:
            a = human_act(game, human_seat)
        else:
            a = cfr_agent.act(game, cfr_seat)
            phase_name = game.phase.value
            if phase_name == "fold":
                a_str = "PLAY" if a == FOLD_ACTION_PLAY else "FOLD"
            else:
                if a == 0:
                    a_str = "STOP"
                else:
                    idx = [i for i in range(HAND_SIZE) if (a >> i) & 1]
                    a_str = f"redraw {idx}"
            print(f"  [AI acts: {a_str}]")
        game.apply(a)

    # Reveal both hands at terminal
    print("\n" + "=" * 60)
    print("GAME OVER")
    print(f"  AI hand revealed: {render_hand(game.hands[cfr_seat])}")
    print(f"  Your hand:        {render_hand(game.hands[human_seat])}")
    u = game.utility(human_seat)
    if u > 0:
        print("  *** YOU WIN ***")
    elif u < 0:
        print("  *** AI WINS ***")
    else:
        print("  *** TIE ***")


def main() -> None:
    parser = argparse.ArgumentParser(description="Human vs CFR PokerGame")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint .pkl (omit = uniform random AI)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Game seed (random by default)")
    parser.add_argument("--human-seat", type=int, choices=[0, 1], default=1,
                        help="Which seat the human plays (default: 1)")
    args = parser.parse_args()

    if args.checkpoint:
        cfr = CFRAgent.from_checkpoint(args.checkpoint)
        print(f"Loaded CFR agent from {args.checkpoint}")
    else:
        cfr = CFRAgent(RegretTable())
        print("No checkpoint — AI plays uniform random.")

    import random as _r
    seed = args.seed if args.seed is not None else _r.randrange(2**31)
    run_match(cfr, human_seat=args.human_seat, game_seed=seed)


if __name__ == "__main__":
    main()
