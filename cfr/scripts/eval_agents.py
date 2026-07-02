"""Evaluate an agent against the 5 rule baselines (and optionally head-to-head).

Usage:
    python -m cfr.scripts.eval_agents --agent qlearning
    python -m cfr.scripts.eval_agents --agent cfr --checkpoint cfr/checkpoints/<f>.pkl
    python -m cfr.scripts.eval_agents --head-to-head --checkpoint <f>.pkl
"""

import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from cfr.eval.play import play_match
from cfr.eval.qlearning_agent import QTableAgent
from cfr.eval.rule_agents import (
    RandomAgent, AlwaysCallAgent, TightAgent, LooseAgent, HeuristicAgent,
)

BASELINES = [
    ("random",      lambda: RandomAgent(seed=1)),
    ("always_call", lambda: AlwaysCallAgent()),
    ("tight",       lambda: TightAgent()),
    ("loose",       lambda: LooseAgent()),
    ("heuristic",   lambda: HeuristicAgent()),
]


def eval_vs_baselines(agent, num_games: int, seed: int) -> None:
    for name, make_opp in BASELINES:
        r = play_match(agent, make_opp(), num_games=num_games, seed=seed)
        print(f"{name:<14} avg_u={r['a_avg_utility']:+.3f}  "
              f"W-T-L: {r['a_wins']}-{r['ties']}-{r['b_wins']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["qlearning", "cfr"], default=None)
    p.add_argument("--checkpoint", default=None, help="CFR checkpoint .pkl")
    p.add_argument("--head-to-head", action="store_true",
                   help="CFR (from --checkpoint) vs legacy Q-learning bot")
    p.add_argument("--num-games", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.head_to_head:
        from cfr.eval.cfr_agent import CFRAgent
        cfr = CFRAgent.from_checkpoint(args.checkpoint)
        ql = QTableAgent()
        r = play_match(cfr, ql, num_games=args.num_games, seed=args.seed)
        print(f"cfr vs qlearning  avg_u={r['a_avg_utility']:+.3f}  "
              f"W-T-L: {r['a_wins']}-{r['ties']}-{r['b_wins']}")
        return

    if args.agent == "qlearning":
        agent = QTableAgent()
    else:
        from cfr.eval.cfr_agent import CFRAgent
        agent = CFRAgent.from_checkpoint(args.checkpoint)
    eval_vs_baselines(agent, args.num_games, args.seed)


if __name__ == "__main__":
    main()
